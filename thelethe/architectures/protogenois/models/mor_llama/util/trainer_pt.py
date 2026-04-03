import os
import time
import glob
import pickle
import shutil
import warnings
import functools
import contextlib
from packaging import version
from typing import Dict, Union, Any, Optional

import torch
from torch import nn 
import torch.distributed as dist
from transformers import Trainer 
from transformers.trainer import _is_peft_model, _get_fsdp_ckpt_kwargs
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.trainer_utils import TrainOutput, SaveStrategy, speed_metrics
from transformers.trainer_pt_utils import get_model_param_count, reissue_pt_warnings
from transformers.optimization import get_scheduler
from transformers.utils import (
    is_torch_xla_available,
    is_sagemaker_mp_enabled,
    is_apex_available,
    is_accelerate_available,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    logging,
)
from transformers.trainer_callback import (
    ExportableState,
    TrainerState,
)
from transformers.debug_utils import DebugOption
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION
    from transformers.trainer_pt_utils import smp_forward_backward
    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False
if is_apex_available():
    from apex import amp
if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate.utils import DistributedType, load_fsdp_optimizer
    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper
    
from util.losses import DISTILL_LOSSES
from util.misc import get_iterator
from util.callback import MoRCallbackHandler, MorSaveCallback

TRAINER_STATE_NAME = "trainer_state.json"

logger = logging.get_logger(__name__)


class MoRTrainer(Trainer):
    def __init__(self, *args, cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam_tr_loss = torch.tensor(0.0).to(self.args.device)  # for logging sampling_loss from MoR
        self.sam_tr_acc = torch.tensor(0.0).to(self.args.device)  # for logging sampling_acc from MoR
        self.sam_tr_topk_acc = torch.tensor(0.0).to(self.args.device)  # for logging sampling_topk_acc from MoR
        self.bal_tr_loss = torch.tensor(0.0).to(self.args.device)  # for logging balancing_loss from MoR
        self.bal_tr_ratio = torch.tensor([0.0] * cfg.recursive.num_recursion).to(self.args.device)  # for logging balancing_ratio from MoR
        self.bal_tr_entropy = torch.tensor(0.0).to(self.args.device)  # for logging balancing_entropy from MoR
        self.router_z_loss = torch.tensor(0.0).to(self.args.device)  # for logging router_z_loss from MoR
        self.cfg = cfg
        
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and "mlp_router" not in n and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and "mlp_router" not in n and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
        
    def create_sampling_optimizer_and_scheduler(self, num_training_steps):
        self.create_sampling_optimizer()
        if IS_SAGEMAKER_MP_POST_1_10 and smp.state.cfg.fp16:
            # If smp >= 1.10 and fp16 is enabled, we unwrap the optimizer
            optimizer = self.sam_optimizer.optimizer
        else:
            optimizer = self.sam_optimizer
        self.create_sampling_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)
        
    def create_sampling_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if not hasattr(self, "sam_optimizer"):
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if ("mlp_router" in n and p.requires_grad)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * 10.0,
                },
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.sam_optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.sam_optimizer = smp.DistributedOptimizer(self.sam_optimizer)
            
        self.sam_optimizer = self.accelerator.prepare_optimizer(self.sam_optimizer)
        return self.sam_optimizer
    
    def create_sampling_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if not hasattr(self, "sam_lr_scheduler"):
            self.sam_lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.sam_optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
        self.sam_lr_scheduler = self.accelerator.prepare_scheduler(self.sam_lr_scheduler)
        return self.sam_lr_scheduler
    
    def _load_sam_optimizer_and_sam_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        if self.is_deepspeed_enabled:
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            if not isinstance(self.sam_lr_scheduler, DeepSpeedSchedulerWrapper):
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.sam_lr_scheduler.load_state_dict(
                        torch.load(os.path.join(checkpoint, "sam_lr_scheduler.pt"), weights_only=True)
                    )
                reissue_pt_warnings(caught_warnings)
            return

        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, "sam_optimizer.pt") + "_*")
            if is_sagemaker_mp_enabled()
            else (
                os.path.isfile(os.path.join(checkpoint, "sam_optimizer.pt"))
                or os.path.isfile(os.path.join(checkpoint, "sam_optimizer.bin"))
                or (
                    os.path.isdir(checkpoint)
                    and any(
                        "sam_optimizer.bin".split(".")[0] in folder_name
                        for folder_name in os.listdir(checkpoint)
                        if os.path.isdir(os.path.join(checkpoint, folder_name))
                    )
                )
            )
        )
        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, f"rank*-of-{self.args.world_size}-sam_optimizer.pt"))
            if self.is_fsdp_xla_v1_enabled
            else checkpoint_file_exists
        )
        if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, "sam_lr_scheduler.pt")):
            # Load in optimizer and scheduler states
            if is_torch_xla_available():
                # On TPU we have to take some extra precautions to properly load the states on the right device.
                if self.is_fsdp_xla_v1_enabled:
                    optimizer_state = torch.load(
                        os.path.join(
                            checkpoint, f"rank{self.args.process_index}-of-{self.args.world_size}-sam_optimizer.pt"
                        ),
                        map_location="cpu",
                        weights_only=True,
                    )
                    # We only need `optimizer` when resuming from checkpoint
                    optimizer_state = optimizer_state["optimizer"]
                else:
                    optimizer_state = torch.load(
                        os.path.join(checkpoint, "sam_optimizer.pt"), map_location="cpu", weights_only=True
                    )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    lr_scheduler_state = torch.load(
                        os.path.join(checkpoint, "sam_lr_scheduler.pt"), map_location="cpu", weights_only=True
                    )
                reissue_pt_warnings(caught_warnings)

                xm.send_cpu_data_to_device(optimizer_state, self.args.device)
                xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)

                self.sam_optimizer.load_state_dict(optimizer_state)
                self.sam_lr_scheduler.load_state_dict(lr_scheduler_state)
            else:
                if is_sagemaker_mp_enabled():
                    if os.path.isfile(os.path.join(checkpoint, "user_content.pt")):
                        # Optimizer checkpoint was saved with smp >= 1.10
                        def opt_load_hook(mod, opt):
                            opt.load_state_dict(smp.load(os.path.join(checkpoint, "sam_optimizer.pt"), partial=True))

                    else:
                        # Optimizer checkpoint was saved with smp < 1.10
                        def opt_load_hook(mod, opt):
                            if IS_SAGEMAKER_MP_POST_1_10:
                                opt.load_state_dict(
                                    smp.load(os.path.join(checkpoint, "sam_optimizer.pt"), partial=True, back_compat=True)
                                )
                            else:
                                opt.load_state_dict(smp.load(os.path.join(checkpoint, "sam_optimizer.pt"), partial=True))

                    self.model_wrapped.register_post_step_hook(opt_load_hook)
                else:
                    # We use the CPU when training on one GPU to avoid OOM for GPU RAM when training big models.
                    # In distributed training however, we load directly on each GPU and risk the GPU OOM as it's more
                    # likely to get OOM on CPU (since we load num_gpu times the optimizer state
                    map_location = self.args.device if self.args.world_size > 1 else "cpu"
                    if self.is_fsdp_enabled:
                        load_fsdp_optimizer(
                            self.accelerator.state.fsdp_plugin,
                            self.accelerator,
                            self.sam_optimizer,
                            self.model,
                            checkpoint,
                            **_get_fsdp_ckpt_kwargs(),
                        )
                    else:
                        self.sam_optimizer.load_state_dict(
                            torch.load(
                                os.path.join(checkpoint, "sam_optimizer.pt"), map_location=map_location, weights_only=True
                            )
                        )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.sam_lr_scheduler.load_state_dict(
                        torch.load(os.path.join(checkpoint, "sam_lr_scheduler.pt"), weights_only=True)
                    )
                reissue_pt_warnings(caught_warnings)
    
    def _load_training_step(self, checkpoint):
        if checkpoint is None:
            return
        
        if os.path.isfile(os.path.join(checkpoint, "training_step.pickle")):
            with open(os.path.join(checkpoint, "training_step.pickle"), 'rb') as f:
                training_step = pickle.load(f)
                
            for layer_idx in range(len(self.model.model.layers)):
                if hasattr(self.model.model.layers[layer_idx], "training_step"):
                    self.model.model.layers[layer_idx].training_step = training_step
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
            if self.cfg.mor.type == "expert":
                if self.cfg.mor.expert.sampling == "aux_router":
                    self.create_sampling_optimizer_and_scheduler(num_training_steps=max_steps)
                else:
                    self.sam_optimizer, self.sam_lr_scheduler = None, None
                self.mor_callback_handler = MoRCallbackHandler([MorSaveCallback(self.cfg.save_steps)], self.model, self.processing_class, self.optimizer, self.lr_scheduler, self.sam_optimizer, self.sam_lr_scheduler)
                
        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)
        self._load_training_step(resume_from_checkpoint)
        if self.cfg.mor.type == "expert":
            if self.cfg.mor.expert.sampling == "aux_router":
                self._load_sam_optimizer_and_sam_scheduler(resume_from_checkpoint)
            
        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            if args.gradient_accumulation_steps == 1:
                total_updates -= 1
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += (
                                self.accelerator.gather(input_tokens).sum().cpu().item()
                            )
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                param_dict = {n : p for n, p in model.named_parameters() if 'mlp_router' not in n}
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    get_iterator(param_dict),
                                    args.max_grad_norm,
                                )
                                
                                param_dict = {n : p for n, p in model.named_parameters() if 'mlp_router' in n}
                                _ = self.accelerator.clip_grad_norm_(
                                    get_iterator(param_dict),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self.optimizer.step()
                        if hasattr(self, "sam_optimizer") and self.sam_optimizer is not None:
                            self.sam_optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()
                                if hasattr(self, "sam_lr_scheduler") and self.sam_lr_scheduler is not None:
                                    self.sam_lr_scheduler.step()

                        if self.cfg.mor.type == "token":
                            self.bal_tr_ratio = self.bal_tr_ratio / self.args.gradient_accumulation_steps
                            
                            if self.cfg.mor.token.balancing == "loss_free":
                                with torch.no_grad():
                                    avg_expert_ratio = torch.mean(self.bal_tr_ratio).expand(self.bal_tr_ratio.numel())
                                    eps_i = avg_expert_ratio - self.bal_tr_ratio
                                        
                                    if self.cfg.recursive.sharing == "cycle":
                                        self.model.model.layers.router_bias.data += self.cfg.mor.token.u * torch.sign(eps_i)
                                    elif self.cfg.recursive.sharing == "middle_cycle":
                                        self.model.model.layers[1].router_bias.data += self.cfg.mor.token.u * torch.sign(eps_i)
                                
                            self.bal_tr_entropy += -torch.sum(self.bal_tr_ratio * torch.log(self.bal_tr_ratio + 1e-10))
                            self.bal_tr_ratio -= self.bal_tr_ratio

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
                        )
                        
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    def update_metric(self, metric, key=""):
        metric_step = metric.detach()
        # Retrieve the current value associated with the key.
        current_metric = getattr(self, key)
        
        if (
            self.args.logging_nan_inf_filter
            and not is_torch_xla_available()
            and (torch.isnan(metric_step.sum()) or torch.isinf(metric_step).sum())
        ):
            # if the metric is NaN or Inf, add the average of previous logged losses
            new_metric = current_metric + current_metric / (1 + self.state.global_step - self._globalstep_last_logged)
        else:
            if current_metric.device != metric_step.device:
                raise ValueError(
                    f"Calculated loss must be on the original device: {current_metric.device} but device in use is {metric_step.device}"
                )
            new_metric = current_metric + metric_step
        
        # Update the attribute using setattr.
        setattr(self, key, new_metric)
        
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        if hasattr(self, "sam_optimizer") and self.sam_optimizer is not None \
            and hasattr(self.sam_optimizer, "train") and callable(self.sam_optimizer.train):
                self.sam_optimizer.train()
            
        inputs = self._prepare_inputs(inputs)        

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        
        with self.compute_loss_context_manager():
            loss, sampling_loss, sampling_acc, sampling_topk_acc, uniformity, dead_token_seq, balancing_loss, balancing_ratio, router_z_loss \
                = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            
        if self.cfg.mor.type == "expert":
            if self.cfg.mor.expert.sampling == "aux_loss":
                coeff = self.cfg.mor.expert.get("coeff", 0.01)
                total_loss = loss + coeff * sampling_loss
            elif self.cfg.mor.expert.sampling == "aux_router": 
                total_loss = loss
        elif self.cfg.mor.type == "token":
            if self.cfg.mor.token.balancing == "loss":
                coeff = self.cfg.mor.token.get("coeff", 0.01)
                total_loss = loss + coeff * balancing_loss
            elif self.cfg.mor.token.balancing == "loss_free":
                total_loss = loss
        else:
            raise ValueError(f"Invalid MOR type: {self.cfg.mor.type}")
            
        if "z_loss" in self.cfg.mor and self.cfg.mor.z_loss:
            z_coeff = self.cfg.mor.get("z_coeff", 0.001)
            total_loss = total_loss + z_coeff * router_z_loss
            
        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning(
                    "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                )
            else:
                torch.cuda.empty_cache()
        
        # Handle main loss optimization
        kwargs = {}
        
        # For LOMO optimizers you need to explicitly use the learning rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()
        
        # Backward pass for LM loss
        if self.args.n_gpu > 1:
            total_loss = total_loss.mean()  # mean() to average on multi-gpu parallel training
            
        if self.use_apex:
            with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()  # retain graph for aux loss
        else:
            # Normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                total_loss = total_loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(total_loss, **kwargs, retain_graph=True) # retain_graph
            
        if self.cfg.mor.type == "expert" and self.cfg.mor.expert.sampling == "aux_router":
            # Backward pass for auxiliary loss
            if self.args.n_gpu > 1:
                sampling_loss = sampling_loss.mean()
                
            # Backward pass for auxiliary loss
            if self.use_apex:
                with amp.scale_loss(sampling_loss, self.sam_optimizer) as scaled_sampling_loss:
                    scaled_sampling_loss.backward()
            else:
                # Normalize aux loss if needed
                if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                    sampling_loss = sampling_loss / self.args.gradient_accumulation_steps
    
                self.accelerator.backward(sampling_loss, **kwargs)
            
        if self.cfg.mor.type == "expert":
            self.update_metric(sampling_loss, key="sam_tr_loss")
            self.update_metric(sampling_acc, key="sam_tr_acc")
            self.update_metric(sampling_topk_acc, key="sam_tr_topk_acc")
        
        if self.cfg.mor.type == "token":
            self.update_metric(balancing_loss, key="bal_tr_loss")
            self.update_metric(balancing_ratio, key="bal_tr_ratio")
        
        self.update_metric(router_z_loss, key="router_z_loss")
        
        return loss.detach()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # How the loss is computed by Trainer. By default, all models return the loss in the first element.
        # Subclass and override for custom behavior.
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        
        outputs = model(**inputs)
        
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        sampling_loss = outputs.sampling_loss if isinstance(outputs, dict) else outputs[-8]
        sampling_acc = outputs.sampling_acc if isinstance(outputs, dict) else outputs[-7]
        sampling_topk_acc = outputs.sampling_topk_acc if isinstance(outputs, dict) else outputs[-6]
        uniformity = outputs.uniformity if isinstance(outputs, dict) else outputs[-5]
        dead_token_seq = outputs.dead_token_seq if isinstance(outputs, dict) else outputs[-4]
        balancing_loss = outputs.balancing_loss if isinstance(outputs, dict) else outputs[-3]
        balancing_ratio = outputs.balancing_ratio if isinstance(outputs, dict) else outputs[-2]
        router_z_loss = outputs.router_z_loss if isinstance(outputs, dict) else outputs[-1]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes
            sampling_loss *= self.accelerator.num_processes
            balancing_loss *= self.accelerator.num_processes
            router_z_loss *= self.accelerator.num_processes
            
        return (loss, sampling_loss, sampling_acc, sampling_topk_acc, uniformity, dead_token_seq, \
            balancing_loss, balancing_ratio, router_z_loss)

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()
                
            logs: Dict[str, float] = {}
            
            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            router_z_loss_scalar = self._nested_gather(self.router_z_loss).mean().item()
            
            if self.cfg.mor.type == "expert":
                sam_tr_loss_scalar = self._nested_gather(self.sam_tr_loss).mean().item()
                sam_tr_acc_scalar = self._nested_gather(self.sam_tr_acc).mean().item()
                sam_tr_topk_acc_scalar = self._nested_gather(self.sam_tr_topk_acc).mean().item()
                
            elif self.cfg.mor.type == "token":
                bal_tr_loss_scalar = self._nested_gather(self.bal_tr_loss).mean().item()
                bal_tr_entropy_scalar = self._nested_gather(self.bal_tr_entropy).mean().item()
                
            # reset tr_loss to zero
            tr_loss -= tr_loss
            self.router_z_loss -= self.router_z_loss
            
            if self.cfg.mor.type == "expert":
                self.sam_tr_loss -= self.sam_tr_loss
                self.sam_tr_acc -= self.sam_tr_acc
                self.sam_tr_topk_acc -= self.sam_tr_topk_acc
                
            elif self.cfg.mor.type == "token":
                self.bal_tr_loss -= self.bal_tr_loss
                self.bal_tr_entropy -= self.bal_tr_entropy
                
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            
            if self.cfg.mor.type == "expert":
                denominator = self.cfg.recursive.num_recursion - 1 if ("include_first" in self.cfg.mor.expert and not self.cfg.mor.expert.include_first) else self.cfg.recursive.num_recursion
                logs["sampling_loss"] = round(sam_tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
                logs["sampling_loss"] = round(logs["sampling_loss"] / denominator / self.args.gradient_accumulation_steps, 4) 
                logs["sampling_acc"] = round(sam_tr_acc_scalar / self.args.gradient_accumulation_steps / (self.state.global_step - self._globalstep_last_logged), 4)
                logs["sampling_topk_acc"] = round(sam_tr_topk_acc_scalar / self.args.gradient_accumulation_steps / (self.state.global_step - self._globalstep_last_logged), 4)
                logs["router_z_loss"] = round(router_z_loss_scalar / self.cfg.recursive.num_recursion / (self.state.global_step - self._globalstep_last_logged), 4)
            
            elif self.cfg.mor.type == "token":
                logs["balancing_loss"] = round(bal_tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
                logs["balancing_entropy"] = round(bal_tr_entropy_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
                logs["router_z_loss"] = round(router_z_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()
            
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            
            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            if hasattr(self, "mor_callback_handler"):
                self.mor_callback_handler.on_save(self.args, self.state, self.control)