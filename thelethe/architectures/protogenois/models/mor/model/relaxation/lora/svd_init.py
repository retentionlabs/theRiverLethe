import re

import torch

from util.misc import get_torch_dtype


def svd_init_lora_weights(cfg, peft_model, lora_init_dict, rank_pattern):
    rank = cfg.relaxation.lora.r
    torch_dtype = get_torch_dtype(cfg)
    
    lora_A_updates, lora_B_updates = {}, {}
    
    for key, init_weight in lora_init_dict.items():
        layer_idx, layer_name = key.split("_", 1)
        
        with torch.no_grad():
            u, s, v = torch.linalg.svd(init_weight.float().cuda(), full_matrices=False)
            
            pattern = key.replace("_", "\..*\.", 1)
            found = False
            for target_key, value in rank_pattern.items():
                if re.search(pattern, target_key):
                    rank = value
                    found = True; break                    
            
            if found:
                u, s, v = u[:, :rank], s[:rank], v[:rank, :]
                
                lora_A_updates[f"{layer_idx}_{layer_name}"] = v.cpu().contiguous()
                lora_B_updates[f"{layer_idx}_{layer_name}"] = (u @ torch.diag(s)).cpu().contiguous()
        
    for name, param in peft_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            if cfg.model in ["smollm", "smollm2", "tinyllama", "gemma", "pythia"]:
                match = re.search(r"layers\.(\d+)\.(\w+)\.(\w+)\.lora_[AB]", name)
                if not match:
                    continue
                layer_idx, _, layer_name = match.groups()
                
            key = f"{layer_idx}_{layer_name}"
            if "lora_A" in name and key in lora_A_updates:
                assert param.shape == lora_A_updates[key].shape, f"Shape mismatch: {param.shape} != {lora_A_updates[key].shape}"
                with torch.no_grad():
                    param.data = torch.as_tensor(lora_A_updates[key], dtype=torch_dtype).to(param.device)
            elif "lora_B" in name and key in lora_B_updates:
                assert param.shape == lora_B_updates[key].shape, f"Shape mismatch: {param.shape} != {lora_B_updates[key].shape}"
                with torch.no_grad():
                    param.data = torch.as_tensor(lora_B_updates[key], dtype=torch_dtype).to(param.device)
                 
    return peft_model