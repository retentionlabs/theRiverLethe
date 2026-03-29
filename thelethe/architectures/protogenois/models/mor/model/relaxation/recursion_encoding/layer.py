from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.processing_utils import Unpack

from model.recursive_model.modeling_llama import Cache, FlashAttentionKwargs


class EncodedLlamaDecoderLayer(nn.Module):
    def __init__(self, config, block):
        super().__init__()
        self.config = config
        self.block = block
        
        self.recursion_encoding = nn.Parameter(torch.ones(config.hidden_size))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs]):
    
        outputs = self.block(
                hidden_states=hidden_states, 
                attention_mask=attention_mask,
                position_ids=position_ids, 
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs
        )
    
        relaxed_output = outputs[0] * self.recursion_encoding
        return (relaxed_output,) + outputs[1:]