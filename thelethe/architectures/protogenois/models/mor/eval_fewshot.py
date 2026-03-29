import os
os.environ["HYDRA_FULL_ERROR"] = "1"
from paths import PROJECT_ROOT, SAVE_DIR, HF_CACHE_DIR; os.environ["HF_HOME"] = HF_CACHE_DIR
import glob
import json
import pickle
import logging
import re
import sys
import wandb
from pathlib import Path
from collections import defaultdict

import string
import random
import numpy as np
import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from omegaconf import OmegaConf, open_dict
from safetensors import safe_open
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from transformers import AutoConfig

import lm_eval
from lm_eval import evaluator, utils
from lm_eval.evaluator import request_caching_arg_to_dict
from lm_eval.loggers import EvaluationTracker
from lm_eval.tasks import TaskManager
from lm_eval.api.registry import get_model
from lm_eval.utils import handle_non_serializable, make_table, simple_parse_args_string
from model.util import MODEL_CLS, RECURSIVE_MODEL_CLS, MOR_MODEL_CLS
from util.tokenizer import load_tokenizer_from_config
from util.config import overwrite_eval_config
from util.misc import convert_to_serializable, print_rank_zero; print_rank_zero()


def init_wandb(cfg):                    
    os.environ["WANDB_ENTITY"] = cfg.wandb_entity
    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    if cfg.get("wandb_watch") is not None:
        os.environ["WANDB_WATCH"] = cfg.get("wandb_watch")
    os.environ ["WANDB_RESUME"] = "allow"
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = cfg.get("WANDB_MODE", "online")
    if os.environ["WANDB_MODE"] == "offline":
        os.environ["WANDB_DIR"] = PROJECT_ROOT
    os.environ["WANDB_SAVE_CODE"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "false"
    if cfg.get("wandb_run_id") is None:
        fpath = os.path.join(PROJECT_ROOT, "conf/pretrain", f"{cfg.name}.yaml")
        if os.path.exists(fpath):
            train_cfg = OmegaConf.load(fpath)
            with open_dict(cfg):
                cfg.wandb_run_id = train_cfg.wandb_run_id
        else:
            characters = string.ascii_letters + string.digits
            random.seed(os.urandom(16))
            wandb_run_id = "".join(random.choices(characters, k=8))
            with open_dict(cfg):
                cfg.wandb_run_id = wandb_run_id                
    os.environ["WANDB_RUN_ID"] = cfg.wandb_run_id
    wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=cfg.name)
    wandb.define_metric("eval_fewshot/step")
    

@hydra.main(config_path="conf/eval_fewshot", config_name="yymmdd_eval_fewshot")
def main(cfg: DictConfig):
    
    cfg = overwrite_eval_config(cfg).eval_fewshot
    
    # preprocess config
    cfg.output_path = os.path.join(SAVE_DIR, "eval_fewshot", cfg.name)
        
    if cfg.get("tensorboard"):
        writer = SummaryWriter(os.path.join(SAVE_DIR, "tensorboard", cfg.name))
    else:
        writer = None
        
    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{cfg.verbosity}"))
    eval_logger.info(f"Verbosity set to {cfg.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if cfg.get("eval_multiple_checkpoints", False):
        print("eval_multiple_checkpoints only supports for checkpoints that we trained")
        evaluate_multiple_checkpoints(cfg, eval_logger, writer)
    else:
        evaluate_model(cfg, eval_logger, writer)
       

def evaluate_model(cfg: DictConfig, eval_logger=None, writer=None, step=None):
    if cfg.model in ["hf-auto", "hf", "huggingface"]:
        print("Preparing evaluation of vanilla Transformer")
        assert cfg.model_args, "Must specify --model_args"
        
        # check dtype
        if cfg.precision == "fp32":
            dtype, torch_dtype = "float32", torch.float32
        elif cfg.precision == "fp16":
            dtype, torch_dtype = "float16", torch.float16
        elif cfg.precision == "bf16":
            dtype, torch_dtype = "bfloat16", torch.bfloat16
        else:
            raise ValueError
        
        print("Using huggingface model")
        model = cfg.model
        
        if "dtype" not in cfg.model_args:
            cfg.model_args += f",dtype={dtype}"
        else:
            pattern = r"dtype=\w+"
            cfg.model_args = re.sub(pattern, f"dtype={dtype}", cfg.model_args)
    
    elif cfg.model in ["recursive_lm", "recursive_transformer"]:
        print("Preparing evaluation of Recursive Transformer")
        assert cfg.model_args, "Must specify --model_args"
        
        train_cfg_path = os.path.join(PROJECT_ROOT, cfg.train_cfg_fpath, f"{cfg.model_name_or_path.split('/')[-1]}.yaml")
        if os.path.exists(train_cfg_path):
            # in case of pretrained model, they need tokenizer config files
            tokenizer = load_tokenizer_from_config(cfg)
            tokenizer.save_pretrained(cfg.model_args.split("=")[1])
        
        # check dtype
        if cfg.precision == "fp32":
            dtype, torch_dtype = "float32", torch.float32
        elif cfg.precision == "fp16":
            dtype, torch_dtype = "float16", torch.float16
        elif cfg.precision == "bf16":
            dtype, torch_dtype = "bfloat16", torch.bfloat16
        else:
            raise ValueError
        
        if "mor" in cfg and cfg.mor.enable:
            model_cls = MOR_MODEL_CLS[cfg.model_arch]
        elif cfg.recursive.enable or ("kv_sharing" in cfg and cfg.kv_sharing.enable):
            model_cls = RECURSIVE_MODEL_CLS[cfg.model_arch]
        else:
            model_cls = MODEL_CLS[cfg.model_arch]
            
        attn_implementation = cfg.get("attn_implementation", "flash_attention_2")
        print("Loading model from pretrained weights...")
        print(f"Loading model with {attn_implementation}...")
                
        if cfg.relaxation.enable:
            # assert os.path.exists(os.path.join(cfg.model_args.split("=")[1], "adapter_config.json"))
            if cfg.relaxation.method in ["lora", "dora", "recursion_encoding"]:
                model = model_cls.from_pretrained(
                    cfg.model_args.split("=")[1],
                    attn_implementation=attn_implementation, 
                    torch_dtype=torch_dtype,
                )
                
                if cfg.relaxation.method == "recursion_encoding":
                    from model.relaxation.util import relax_weight_sharing
                    model = relax_weight_sharing(cfg, model)
                
                try:
                    state_dict = torch.load(os.path.join(cfg.model_args.split("=")[1], "pytorch_model.bin"))
                except FileNotFoundError:
                    from safetensors.torch import load_file
                    state_dict = load_file(os.path.join(cfg.model_args.split("=")[1], "model.safetensors"))
                    
                model.load_state_dict(state_dict)
                
            elif cfg.relaxation.method == "adaption_prompt":
                model = model_cls.from_pretrained(
                    OmegaConf.load(train_cfg_path).model_name_or_path,
                    attn_implementation=attn_implementation,
                    torch_dtype=torch_dtype,
                )
                
                from model.sharing_strategy import SHARING_STRATEGY
                model, _ = SHARING_STRATEGY[cfg.model_arch](cfg, model)
                
                from model.relaxation.util import relax_weight_sharing
                model = relax_weight_sharing(cfg, model)
                
                state_dict = torch.load(os.path.join(cfg.model_args.split("=")[1], "pytorch_model.bin"))
                if cfg.relaxation.method == "adaption_prompt":
                    model.get_base_model().load_state_dict(state_dict)
                    
                    adapter_state_dict = torch.load(os.path.join(cfg.model_args.split("=")[1], "adapter_model.bin"))
                    
                    if "prompt_embeddings" in adapter_state_dict:
                        adapter_state_dict["prompt_encoder.default.embedding.weight"] = adapter_state_dict["prompt_embeddings"]
                        del adapter_state_dict["prompt_embeddings"]
                        
                    model.load_state_dict(adapter_state_dict, strict=False)
            
        else:
            if "mor" in cfg and cfg.mor.get("enable"):
                config = AutoConfig.from_pretrained(
                        OmegaConf.load(train_cfg_path).model_name_or_path,
                        attn_implementation=attn_implementation, 
                        torch_dtype=torch_dtype,
                    )
                                
                if cfg.get("model_config") is not None:
                    print("Using custom config for vanilla model...")
                    for k, v in cfg.model_config.items():
                        if not hasattr(config, k):
                            raise ValueError(f"Config key {k} not found in model config.")
                        print(f" {k}: {v}")
                        setattr(config, k, v)
                model =  model_cls._from_config(
                    config, attn_implementation=attn_implementation, torch_dtype=torch_dtype,)
                        
                if cfg.mor.type == "expert":
                    model.transform_layer_to_mor_expert(cfg)
                elif cfg.mor.type == "token":
                    model.transform_layer_to_mor_token(cfg)
                
                state_dict = torch.load(os.path.join(cfg.model_args.split("=")[1], "pytorch_model.bin"))
                model.load_state_dict(state_dict)
                
            else:                       
                model = model_cls.from_pretrained(
                    cfg.model_args.split("=")[1],
                    attn_implementation=attn_implementation, 
                    torch_dtype=torch_dtype,
                )
        
        if "kv_sharing" in cfg and cfg.kv_sharing.get("enable"):
            model.set_kv_sharing_config(cfg)
                
        accelerator = Accelerator()
        model = accelerator.prepare(model)
        
        model = get_model(cfg.model)(
            pretrained=model,
            tokenizer=tokenizer,
            device=cfg.device,
            max_length=cfg.max_length,
            dtype=dtype,
            add_bos_token=cfg.add_bos_token,
            batch_size=cfg.batch_size,
        )
            
    else:
        raise ValueError(f"Model {cfg.model} not supported")
            
    # update the evaluation tracker args with the output path and the HF token
    if cfg.output_path:
        cfg.hf_hub_log_args += f",output_path={cfg.output_path}"
    if os.environ.get("HF_TOKEN", None):
        cfg.hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}"
    evaluation_tracker_args = simple_parse_args_string(cfg.hf_hub_log_args)
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    if cfg.predict_only:
        cfg.log_samples = True
    if (cfg.log_samples or cfg.predict_only) and not cfg.output_path:
        raise ValueError(
            "Specify --output_path if providing --log_samples or --predict_only"
        )

    if cfg.fewshot_as_multiturn and cfg.apply_chat_template is False:
        raise ValueError(
            "When `fewshot_as_multiturn` is selected, `apply_chat_template` must be set (either to `True` or to the chosen template name)."
        )

    if cfg.include_path is not None:
        eval_logger.info(f"Including path: {cfg.include_path}")
    task_manager = TaskManager(cfg.verbosity, include_path=cfg.include_path)

    if "push_samples_to_hub" in evaluation_tracker_args and not cfg.log_samples:
        eval_logger.warning(
            "Pushing samples to the Hub requires --log_samples to be set. Samples will not be pushed to the Hub."
        )

    if cfg.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if cfg.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif cfg.tasks == "list":
        print(task_manager.list_all_tasks())
    else:
        if os.path.isdir(cfg.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(cfg.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = cfg.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task for task in task_list if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                )   
                     
    # Respect user's value passed in via CLI, otherwise default to True and add to comma-separated model args
    if cfg.trust_remote_code:
        eval_logger.info(
            "Passed `--trust_remote_code`, setting environment variable `HF_DATASETS_TRUST_REMOTE_CODE=true`"
        )
        # HACK: import datasets and override its HF_DATASETS_TRUST_REMOTE_CODE value internally,
        # because it's already been determined based on the prior env var before launching our
        # script--`datasets` gets imported by lm_eval internally before these lines can update the env.
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

        cfg.model_args = cfg.model_args + ",trust_remote_code=True"

    eval_logger.info(f"Selected Tasks: {task_names}")

    request_caching_args = request_caching_arg_to_dict(
        cache_requests=cfg.cache_requests
    )

    results = evaluator.simple_evaluate(
        model=model,
        model_args=cfg.model_args,
        tasks=task_names,
        num_fewshot=cfg.num_fewshot,
        batch_size=cfg.batch_size,
        device=cfg.device,
        max_batch_size=cfg.max_batch_size,
        use_cache=cfg.use_cache,
        limit=cfg.limit,
        check_integrity=cfg.check_integrity,
        write_out=cfg.write_out,
        log_samples=cfg.log_samples,
        evaluation_tracker=evaluation_tracker,
        system_instruction=cfg.system_instruction,
        apply_chat_template=cfg.apply_chat_template,
        fewshot_as_multiturn=cfg.fewshot_as_multiturn,
        gen_kwargs=cfg.gen_kwargs,
        task_manager=task_manager,
        verbosity=cfg.verbosity,
        predict_only=cfg.predict_only,
        random_seed=cfg.seed[0],
        numpy_random_seed=cfg.seed[1],
        torch_random_seed=cfg.seed[2],
        fewshot_random_seed=cfg.seed[3],
        **request_caching_args,
    )
    
    if results is not None:
        if cfg.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(
            results, indent=2, default=handle_non_serializable, ensure_ascii=False
        )
        if cfg.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        evaluation_tracker.save_results_aggregated(
            results=results, samples=samples if cfg.log_samples else None
        )

        if cfg.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )

        if (
            evaluation_tracker.push_results_to_hub
            or evaluation_tracker.push_samples_to_hub
        ):
            evaluation_tracker.recreate_metadata_card()

        print(
            f"{cfg.model} ({cfg.model_args}), gen_kwargs: ({cfg.gen_kwargs}), limit: {cfg.limit}, num_fewshot: {cfg.num_fewshot}, "
            f"batch_size: {cfg.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))
            
        if not cfg.get("eval_multiple_checkpoints", False):
            # json file
            save_path = os.path.join(SAVE_DIR, "eval_fewshot", cfg.name, "results_last_ckpt.json")
            with open(save_path, "w") as f:
                results["model_source"] = cfg.model
                json.dump(results, f, indent=4, default=convert_to_serializable)
                
            if cfg.get("wandb"):
                init_wandb(cfg)
            
                api = wandb.Api()
                run = api.run(f"{cfg.wandb_entity}/{cfg.wandb_project}/{cfg.wandb_run_id}")
                history = run.history(keys=["_step"], pandas=False)
                if history:
                    last_step = history[-1]["_step"]
                else:
                    last_step = 0
                                    
                wandb_log = defaultdict(float)
                for task_name, metrics in results["results"].items():
                    for metric_name, value in metrics.items():
                        if "std" not in metric_name:
                            key = f"eval_fewshot/{task_name}_{metric_name.replace('.', '_')}_last_ckpt"
                            wandb_log[key] = value
                            wandb.define_metric(key, step_metric="eval_fewshot/step")
                            
                wandb_log["eval_fewshot/step"] = last_step
                wandb.log(wandb_log)
                wandb.finish()
    return results


def evaluate_multiple_checkpoints(cfg: DictConfig, eval_logger=None, writer=None):
    ckpt_fpath = cfg.model_args.split("=")[1]
    
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        ckpt_dirs = glob.glob(os.path.join(ckpt_fpath, "checkpoint-*"))
        ckpt_dirs.sort(key=lambda x: int(x.split("-")[-1]))
        
        results = defaultdict(dict)        
        if len(ckpt_dirs) != 0:
            print("Evaluating all checkpoints")
            print("=" * 80)
            for ckpt_dir in ckpt_dirs:
                cfg.model_args = f"pretrained={ckpt_dir}"
                
                step = int(ckpt_dir.split("-")[-1])
                _results = evaluate_model(cfg, eval_logger, step=step)
                results[step] = _results
                
    if len(ckpt_dirs) != 0:
        # json file
        save_path = os.path.join(SAVE_DIR, "eval_fewshot", cfg.name, "results_multiple_ckpt.json")
        with open(save_path, "w") as f:
            for step, result in results.items():
                if "model_source" in result:
                    results[step]["model_source"] = cfg.model
            json.dump(results, f, indent=4, default=convert_to_serializable)

        if cfg.get("wandb"):
            init_wandb(cfg)
            
            api = wandb.Api()
            run = api.run(f"{cfg.wandb_entity}/{cfg.wandb_project}/{cfg.wandb_run_id}")
            history = run.history(keys=["_step"], pandas=False)
            
            total_steps = len(history)
            ratio = sorted(results.keys())[-1] / sorted(results.keys())[0]
                
            for idx, (step, _results) in enumerate(results.items()):
                wandb_log = defaultdict(float)
                for task_name, metrics in _results["results"].items():
                    for metric_name, value in metrics.items():
                        if "std" not in metric_name:
                            key = f"eval_fewshot/{task_name}_{metric_name.replace('.', '_')}_multiple_ckpt"
                            wandb_log[key] = value                                
                            wandb.define_metric(key, step_metric="eval_fewshot/step")
                            
                wandb_log["eval_fewshot/step"] = int(total_steps / ratio * (idx + 1)) - 1
                wandb.log(wandb_log)
    return len(ckpt_dirs) != 0


if __name__ == "__main__":
    main()
