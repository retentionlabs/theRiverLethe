import torch
import torch.nn as nn
from peft import LoraConfig, TaskType


def check_name(name, module_list):
    for module_name in module_list:
        if module_name in name:
            return True
    return False


def get_lora_config(cfg, model):
    # Get rank patterns
    rank_pattern = {}    
    if cfg.relaxation.lora.get("rank_pattern"):
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in module_name:
                if not check_name(module_name, cfg.relaxation.lora.target_modules):
                    continue
                
                if cfg.relaxation.lora.rank_pattern.get("self_attn") is not None:
                    if cfg.model in ["smollm", "smollm2", "tinyllama", "gemma", "gpt_neo"]:
                        if "q_proj" in module_name or "k_proj" in module_name or "v_proj" in module_name:
                            rank_pattern[module_name] = int(cfg.relaxation.lora.rank_pattern.self_attn)
                        elif cfg.model in ["pythia"]:
                            if "query_key_value" in module_name or "dense" in module_name:
                                rank_pattern[module_name] = int(cfg.relaxation.lora.rank_pattern.self_attn)
                if cfg.relaxation.lora.rank_pattern.get("ffn") is not None:
                    if cfg.model in ["smollm", "smollm2", "tinyllama", "gemma"]:
                        if "gate_proj" in module_name or "up_proj" in module_name or "down_proj" in module_name:
                            rank_pattern[module_name] = int(cfg.relaxation.lora.rank_pattern.ffn)
                        elif cfg.model in ["gpt_neo"]:
                            if "c_fc" in module_name or "c_proj" in module_name:
                                rank_pattern[module_name] = int(cfg.relaxation.lora.rank_pattern.ffn)
                        elif cfg.model in ["pythia"]:
                            if "dense_h_to_4h" in module_name or "dense_4h_to_h" in module_name:
                                rank_pattern[module_name] = int(cfg.relaxation.lora.rank_pattern.ffn)
        
    if cfg.model == "pythia":
        # Three linear weights are combined into one weight in Pythia
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "query_key_value" not in module_name:
                rank = rank_pattern[module_name] if module_name in rank_pattern else cfg.relaxation.lora.r
                rank_pattern[module_name] = int(rank * 1.5)
                
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in module_name:
            if not check_name(module_name, cfg.relaxation.lora.target_modules):
                continue
            rank = rank_pattern[module_name] if module_name in rank_pattern else cfg.relaxation.lora.r
            rank_pattern[module_name] = min(rank, module.in_features, module.out_features)
                
    # Get alpha patterns
    alpha_pattern = {}
    for key, value in rank_pattern.items():
        if not cfg.relaxation.lora.get("alpha_pattern"):
            alpha_pattern[key] = cfg.relaxation.lora.alpha
        else:
            assert type(cfg.relaxation.lora.alpha_pattern) == float, "alpha_pattern should be a float"
            alpha_pattern[key] = value * cfg.relaxation.lora.alpha_pattern
    
    # Get layers to transform
    relaxed_layers = list(range(model.config.num_hidden_layers))
    if cfg.recursive.sharing in ["middle_sequence", "middle_cycle"]:
        relaxed_layers = relaxed_layers[1:-1]
    
    lora_config = LoraConfig(
        use_dora=False if cfg.relaxation.method == "lora" else True,
        r=cfg.relaxation.lora.r, 
        # lora_alpha=cfg.relaxation.lora.alpha, 
        lora_dropout=cfg.relaxation.lora.dropout,
        target_modules=list(cfg.relaxation.lora.target_modules), 
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        layers_to_transform=relaxed_layers,
        bias="none", 
        task_type=TaskType.CAUSAL_LM
    )
    return lora_config, rank_pattern