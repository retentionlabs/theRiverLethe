import os
import warnings

from omegaconf import DictConfig
import torch
from transformers import AutoConfig 

from model.base_model.modeling_llama import LlamaForCausalLM
from model.recursive_model.modeling_llama import LlamaForCausalLM as RecursiveLlamaForCausalLM
from model.mor_model.modeling_llama import MoRLlamaForCausalLM

MODEL_CLS = {
    "smollm": LlamaForCausalLM,
    "smollm2": LlamaForCausalLM,
}

RECURSIVE_MODEL_CLS = {
    "smollm": RecursiveLlamaForCausalLM,
    "smollm2": RecursiveLlamaForCausalLM,
}

MOR_MODEL_CLS = {
    "smollm": MoRLlamaForCausalLM,
    "smollm2": MoRLlamaForCausalLM,
}

if "wandb_mode" not in os.environ:
    local_files_only = True
else:
    local_files_only = os.environ["WANDB_MODE"] == "offline"


def get_torch_dtype(cfg: DictConfig):
    if cfg.precision == "bf16":
        return torch.bfloat16
    elif cfg.precision == "fp16":
        return torch.float16
    elif cfg.precision == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Precision {cfg.precision} not supported.")


def load_model_from_config(cfg: DictConfig):
    if "mor" in cfg and cfg.mor.enable:
        model_cls = MOR_MODEL_CLS[cfg.model]
    elif cfg.recursive.enable or ("kv_sharing" in cfg and cfg.kv_sharing.enable):
        model_cls = RECURSIVE_MODEL_CLS[cfg.model]
    else:
        model_cls = MODEL_CLS[cfg.model]
        
    attn_implementation = cfg.get("attn_implementation", "flash_attention_2")
    torch_dtype = get_torch_dtype(cfg)
    
    if cfg.use_pretrained_weights:
        print("Loading model from pretrained weights...")
        print(f"Loading model with {attn_implementation}...")
        return model_cls.from_pretrained(
            cfg.model_name_or_path,
            attn_implementation=attn_implementation, 
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
                
    else:
        print("Initializing model from scratch...")
        config = AutoConfig.from_pretrained(
            cfg.model_name_or_path,
            attn_implementation=attn_implementation, 
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        
        if cfg.get("model_config") is not None:
            print("Using custom config for vanilla model...")
            for k, v in cfg.model_config.items():
                if not hasattr(config, k):
                    raise ValueError(f"Config key {k} not found in model config.")
                print(f" {k}: {v}")
                setattr(config, k, v)
        if cfg.get("max_length") and cfg.max_length != config.max_position_embeddings:
            warnings.warn(f"original max_position_embeddings of {config.max_position_embeddings} is changed to {cfg.max_length}")
            setattr(config, "max_position_embeddings", cfg.max_length)
        return model_cls._from_config(
            config, attn_implementation=attn_implementation, torch_dtype=torch_dtype,)