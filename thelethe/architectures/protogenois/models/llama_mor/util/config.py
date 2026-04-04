import os
import warnings

import copy
import torch 
from omegaconf import DictConfig, OmegaConf, open_dict

from util.misc import check_saved_checkpoint
from paths import PROJECT_ROOT, SAVE_DIR


def preprocess_config(cfg: DictConfig):
    print("Preprocess Config ".center(80, "-"))
    
    if cfg.precision not in ["fp32", "fp16", "bf16"]:
        raise NotImplementedError(f"Precision {cfg.precision} is not implemented yet")
    elif cfg.precision == "fp16":
        warnings.warn("Are you sure you want to use fp16? We use bf16 by default.")
        
    # Automatically determine batch size and gradient accumulation steps
    n_gpus = torch.cuda.device_count()
    n_gpus = n_gpus or 1
    if cfg.get("total_batch_size") is not None:
        print("Automatically determining batch size based on `total_batch_size`")
        
        # Automatically set per_device_train_batch_size or
        # (if per_device_train_batch_size is already set) gradient_accumulation steps
        if cfg.get("gradient_accumulation_steps") is not None:
            raise ValueError("Cannot specify both total_batch_size and gradient_accumulation_steps")
        if cfg.get("per_device_train_batch_size") is not None:
            cfg.gradient_accumulation_steps = round(cfg.total_batch_size / (cfg.per_device_train_batch_size * n_gpus))
            print(f"total_batch_size              : {cfg.total_batch_size} (given)")
            print(f"torch.cuda.device_count()     : {n_gpus}")
            print(f"per_device_train_batch_size   : {cfg.per_device_train_batch_size} (given)")
            print(f"gradient_accumulation_steps   : {cfg.gradient_accumulation_steps} (computed)")
            actual_total = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * n_gpus
            print(f"actual total batch size       : {actual_total}")
        else:
            cfg.per_device_train_batch_size = round(cfg.total_batch_size / n_gpus)
            print(f"total_batch_size              : {cfg.total_batch_size} (given)")
            print(f"torch.cuda.device_count()     : {n_gpus}")
            print(f"per_device_train_batch_size   : {cfg.per_device_train_batch_size} (computed)")
            print(f"actual total batch size       : {cfg.per_device_train_batch_size * n_gpus}")

    if cfg.get("model_config") is not None:
        with open_dict(cfg):
            if cfg.model_config.get("hidden_size") is not None and cfg.model_config.get("num_attention_heads") is not None:
                cfg.model_config.head_dim = int(cfg.model_config.hidden_size // cfg.model_config.num_attention_heads)
    
    if "resume_step" in cfg and cfg.resume_step is not None:
        cfg.name = f"{cfg.name}_resume_{cfg.resume_step}"
        
        assert cfg.resume_from_checkpoint is not None and type(cfg.resume_from_checkpoint) == str, "resume_from_checkpoint should be a string"
        cfg.resume_from_checkpoint = os.path.join(SAVE_DIR, "pretrain", cfg.resume_from_checkpoint)
                
    if "wandb_run_name" in cfg and cfg.get("wandb_run_name") is None:
        print(f"Setting wandb_run_name: {cfg.name}")
        cfg.wandb_run_name = cfg.name
        
    if "tensorboard_dir" in cfg and cfg.get("tensorboard_dir") is None:
        cfg.tensorboard_dir = os.path.join(SAVE_DIR, "tensorboard", cfg.name)
        
    if cfg.get("add_bos_token") is True:
        if cfg.tokenizer not in ["gemma", "gemma3", "tinyllama"]:
            raise ValueError("Given tokenizer does not support adding bos token")
    else:
        if cfg.tokenizer in ["gemma", "gemma3", "tinyllama"]:
            cfg.add_bos_token = True

    if "output_dir" in cfg and cfg.get("output_dir") is None:
        print(f"Setting output_dir  : {cfg.name}")
        cfg.output_dir = cfg.name
    
    if check_saved_checkpoint(cfg, SAVE_DIR):
        cfg.resume_from_checkpoint = True
        warnings.warn("Resume from latest checkpoint. If you want to train from scratch, please delete the checkpoint directory.")
        
    if cfg.get("num_warmup_steps") is None:
        warning = "num_warmup_steps not found in config, setting to 5% of num_train_steps"
        warnings.warn(warning)
        
        cfg.num_warmup_steps = int(cfg.num_train_steps * 0.05)

    if cfg.get("save_interval") is not None:
        if cfg.get("save_steps") is not None:
            warning = f"save_interval ({cfg.save_interval}) will be used instead of save_steps ({cfg.save_steps})"
            warnings.warn(warning)
            
        cfg.save_steps = max(int(cfg.num_train_steps * cfg.save_interval), 0)
        
    # Check stop_steps and save_steps
    if cfg.get("stop_steps") is not None and cfg.get("save_steps") is not None:
        if cfg.stop_steps % cfg.save_steps != 0:
            warning = f"stop_steps ({cfg.stop_steps}) is not divisible by save_steps ({cfg.save_steps})"
            warnings.warn(warning)
            
    if cfg.get("deepspeed") is not None:
        # Prepend PROJECT_ROOT if not absolute using os.path.isabs
        if not os.path.isabs(cfg.deepspeed):
            cfg.deepspeed = os.path.join(PROJECT_ROOT, cfg.deepspeed)
        print(f"Using deepspeed config = {cfg.deepspeed}")
    
    if cfg.get("max_grad_norm") is None:
        with open_dict(cfg):
            cfg.max_grad_norm = 1.0  # default value

    if cfg.get("recursive") and cfg.recursive.get("enable"):
        assert cfg.recursive.get("num_recursion"), "num_recursion should be specified"
        
        if cfg.recursive.num_recursion == 1:
            assert cfg.recursive.get("base_depth"), "base_depth should be specified"
            assert cfg.recursive.sharing in ["cycle", "sequence"], "sharing should be either cycle or sequence"
            
    if cfg.get("kv_sharing") and cfg.kv_sharing.get("enable"):
        if cfg.get("recursive") and cfg.recursive.get("enable"):
            cfg.kv_sharing.num_recursion = cfg.recursive.num_recursion
            cfg.kv_sharing.sharing = cfg.recursive.sharing
        else:
            assert cfg.kv_sharing.num_recursion, "num_recursion should be specified"
            assert cfg.kv_sharing.sharing, "sharing should be specified"
            
        if "update_cache" in cfg.kv_sharing and cfg.kv_sharing.update_cache:
            assert cfg.mor.type == "expert", "update_cache is only supported for expert type"
                        
    if cfg.get("relaxation") and cfg.relaxation.get("enable"):
        assert cfg.recursive.get("enable"), "Recursive model should be enabled for relaxation"
        
    if "mor" in cfg and cfg.mor.get("enable"):
        if cfg.mor.capacity is None:
            cfg.mor.capacity = ",".join([str((cfg.recursive.num_recursion - i) / cfg.recursive.num_recursion) for i in range(cfg.recursive.num_recursion)])
        
        assert cfg.recursive.num_recursion > 1, "num_recursion should be greater than 1 for moR"        
        assert len(cfg.mor.capacity.split(',')) == cfg.recursive.num_recursion, "capacity should be a list of length num_recursion"
        assert cfg.recursive.sharing in ["cycle", "middle_cycle"], "sharing should be either cycle or middle_cycle"
        
        if cfg.mor.type == "expert":
            capacity = [float(x) for x in cfg.mor.capacity.split(',')]
            for i in range(len(capacity) - 1):
                if capacity[i] < capacity[i + 1]:
                    raise ValueError("capacity should be in descending order")
        
    print ("-" * 80)
    return cfg


def overwrite_eval_config(eval_cfg: DictConfig):
    """
    For evaluation config, overwrite the config if there is the same name of training config file.
    """
    
    overwrite_keys = ["max_length", "add_bos_token", "tokenizer", "model", "attn_implementation", "model_config", "precision", 
                      "kv_sharing", "recursive", "relaxation", "mor", "num_train_steps", "num_warmup_steps", "gradient_accumulation_steps",]
    train_cfg_path = os.path.join(PROJECT_ROOT, eval_cfg.train_cfg_fpath, f"{eval_cfg.model_name_or_path.split('/')[-1]}.yaml")
    
    if os.path.exists(train_cfg_path):
        train_cfg = OmegaConf.load(train_cfg_path)
        preprocess_config(train_cfg)
        
        for key in overwrite_keys:
            if key in train_cfg:
                with open_dict(eval_cfg):
                    eval_cfg[key] = train_cfg[key]
            else:
                print(f"Warning: Key {key} not found in {train_cfg_path}")
    else:
        print(f"Warning: {train_cfg_path} not found")
    
    if eval_cfg.get("eval_fewshot") is not None:
        if eval_cfg.eval_fewshot.get("model_args") is None:
            eval_cfg.eval_fewshot.model_args = f"pretrained={os.path.join(SAVE_DIR, eval_cfg.model_name_or_path)}"
        if eval_cfg.eval_fewshot.get("model_name_or_path") is None:
            eval_cfg.eval_fewshot.model_name_or_path = os.path.join(SAVE_DIR, eval_cfg.model_name_or_path)
            
    # for eval_fewshot.py
    keys_to_copy = set(eval_cfg.keys()) - set(eval_cfg.eval_fewshot.keys()) - set(["eval_fewshot"])
    
    with open_dict(eval_cfg):
        for key in keys_to_copy:
            eval_cfg.eval_fewshot[key] = copy.deepcopy(eval_cfg[key])
            
        eval_cfg.eval_fewshot.model_arch = eval_cfg.model
    
    return eval_cfg