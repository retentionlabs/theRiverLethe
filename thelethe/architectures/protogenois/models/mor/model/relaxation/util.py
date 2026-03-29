import torch.nn as nn
from peft import get_peft_model
from peft import PeftType
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING

from model.relaxation.lora.config import get_lora_config
from model.relaxation.lora.svd_init import svd_init_lora_weights
from model.relaxation.prompt.config import AdaptionPromptConfig
from model.relaxation.prompt.model import AdaptionPromptModel_
from model.relaxation.recursion_encoding.layer import EncodedLlamaDecoderLayer

PEFT_TYPE_TO_MODEL_MAPPING[PeftType.ADAPTION_PROMPT] = AdaptionPromptModel_


def relax_weight_sharing(cfg, model, lora_init_dict=None):
    """
    Add adapter to the model.
    In order to apply "skip_first_loop",
    For lora, we use layers_to_transform to specify the layers to transform.
    For adaptation_prompt, we freeze the first loop 
    """
        
    if cfg.relaxation.method in ["lora", "dora"]:
        adapter_config, rank_pattern = get_lora_config(cfg, model)
        model = get_peft_model(model, adapter_config)
                    
    elif cfg.relaxation.method == "adaption_prompt":
        adapter_config = AdaptionPromptConfig(
            adapter_len=cfg.relaxation.prompt.len,
            adapter_layers=model.config.num_hidden_layers,
        )
        model = get_peft_model(model, adapter_config)
    
    elif cfg.relaxation.method == "recursion_encoding":
        num_hidden_layers = model.config.num_hidden_layers
        num_recursion = cfg.recursive.num_recursion
        if cfg.recursive.sharing == "sequence":
            model.model.layers = nn.ModuleList([EncodedLlamaDecoderLayer(model.config, layer) for layer_idx, layer in enumerate(model.model.layers)])
        elif cfg.recursive.sharing == "cycle":
            base_depth = num_hidden_layers // num_recursion
            model.model.layers = nn.ModuleList([EncodedLlamaDecoderLayer(model.config, layer) if (layer_idx + 1) % base_depth == 0 else layer for layer_idx, layer in enumerate(model.model.layers)])
        elif cfg.recursive.sharing == "middle_sequence":
            model.model.layers = nn.ModuleList([model.model.layers[0]] \
                + [EncodedLlamaDecoderLayer(model.config, layer) for layer_idx, layer in enumerate(model.model.layers[1:-1])] \
                + [model.model.layers[-1]])
        elif cfg.recursive.sharing == "middle_cycle":
            base_depth = (num_hidden_layers - 2) // num_recursion
            model.model.layers = nn.ModuleList([model.model.layers[0]] \
                + [EncodedLlamaDecoderLayer(model.config, layer) if (layer_idx + 1) % base_depth == 0 else layer for layer_idx, layer in enumerate(model.model.layers[1:-1])] \
                + [model.model.layers[-1]])
        
    else:
        raise ValueError(f"Invalid method type: {cfg.relaxation.method}")
    
    
    # Enable gradients for all parameters, and disable for unrelaxed layers
    for name, param in model.named_parameters():
        param.requires_grad = True
    
    if cfg.relaxation.method in ["lora", "dora"] and cfg.relaxation.lora.get("svd_init"):
        model = svd_init_lora_weights(cfg, model, lora_init_dict, rank_pattern)
        
    return model
