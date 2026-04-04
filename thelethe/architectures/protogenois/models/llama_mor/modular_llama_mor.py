from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from huggingface_hub.dataclasses import strict

from transformers.masking_utils import create_causal_mask
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    ModelOutput,
    auto_docstring,
    can_return_tuple,
    logging
)
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaForCausalLM
)

from .cache_utils import RecursiveDynamicCache
from .routing_llama_mor import LlamaMorExpertRouter, LlamaMorTokenRouter


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="meta-llama/Llama-2-7b-hf")
@strict
class LlamaMorConfig(LlamaConfig):
    r"""
    ```python
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llama_mor"


@dataclass
class LlamaMorBaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    sampling_loss: Optional[torch.FloatTensor] = None
    sampling_acc: Optional[torch.FloatTensor] = None
    sampling_topk_acc: Optional[torch.FloatTensor] = None
    uniformity: Optional[torch.FloatTensor] = None
    dead_token_seq: Optional[torch.FloatTensor] = None
    balancing_loss: Optional[torch.FloatTensor] = None
    balancing_ratio: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None


@dataclass
class LlamaMorCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    sampling_loss: Optional[torch.FloatTensor] = None
    sampling_acc: Optional[torch.FloatTensor] = None
    sampling_topk_acc: Optional[torch.FloatTensor] = None
    uniformity: Optional[torch.FloatTensor] = None
    dead_token_seq: Optional[torch.FloatTensor] = None
    balancing_loss: Optional[torch.FloatTensor] = None
    balancing_ratio: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None


class LlamaMorAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.keep_cache_positions = False

    def forward(self):
        # TODO: implement forward
        pass


class LlamaMorDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaMorAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


@auto_docstring
class LlamaMorModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.experts = nn.ModuleList(

        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def transform_layer_to_mor_expert(self, cfg):
        from model.mor_model.expert_choice_router import MoRLlamaDecoderLayer

        capacity = [float(cap) for cap in cfg.mor.capacity.split(',')]
        # warmup_step for capacity_factor
        if "cap_warmup_step" in cfg.mor.expert and cfg.mor.expert.cap_warmup_step is not None:
            cap_warmup_step = cfg.mor.expert.cap_warmup_step
        else:
            cap_warmup_step = cfg.num_warmup_steps * cfg.gradient_accumulation_steps

        sharing = cfg.recursive.sharing
        num_recursion = cfg.recursive.num_recursion
        num_hidden_layers = len(self.model.layers)

        # Cycle sharing is for early-exiting mechanism
        if sharing == "cycle":
            base_depth = num_hidden_layers // num_recursion
            self.model.layers = nn.ModuleList([
                MoRLlamaDecoderLayer(
                    self.config,
                    nn.ModuleList([self.model.layers[layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]),
                    cfg, capacity[recur_idx], cap_warmup_step
                ) for recur_idx in range(num_recursion)
            ])
        elif sharing == "middle_cycle":
            base_depth = (num_hidden_layers - 2) // num_recursion
            self.model.layers = nn.ModuleList(
                [self.model.layers[0]] + [
                    MoRLlamaDecoderLayer(
                        self.config,
                        nn.ModuleList([self.model.layers[1 + layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]),
                        cfg, capacity[recur_idx], cap_warmup_step
                    )
                    for recur_idx in range(num_recursion)
                ] + [self.model.layers[-1]]
            )

    def transform_layer_to_mor_token(self, cfg):
        from model.mor_model.token_choice_router import MoRLlamaDecoderLayer

        # warmup_step for balancing
        bal_warmup_step = 0
        if "bal_warmup_step" in cfg.mor.token and cfg.mor.token.bal_warmup_step > 0:
            bal_warmup_step = cfg.mor.token.bal_warmup_step * cfg.gradient_accumulation_steps

        sharing = cfg.recursive.sharing
        num_recursion = cfg.recursive.num_recursion
        num_hidden_layers = len(self.model.layers)

        # Cycle sharing is for early-exiting mechanism
        if sharing == "cycle":
            base_depth = num_hidden_layers // num_recursion
            self.model.layers = MoRLlamaDecoderLayer(
                self.config,
                nn.ModuleList([nn.ModuleList([self.model.layers[layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]) for recur_idx in range(num_recursion)]),
                cfg,
                bal_warmup_step,
            )
        elif sharing == "middle_cycle":
            base_depth = (num_hidden_layers - 2) // num_recursion
            self.model.layers = nn.ModuleList(
                [self.model.layers[0]] + \
                [MoRLlamaDecoderLayer(
                    self.config,
                    nn.ModuleList([nn.ModuleList([self.model.layers[1 + layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]) for recur_idx in range(num_recursion)]),
                    cfg,
                    bal_warmup_step,
                ),] + \
                [self.model.layers[-1]]
            )

    def set_kv_sharing_config(self, cfg):
        if cfg.kv_sharing.sharing in ["cycle", "sequence"]:
            base_depth = self.config.num_hidden_layers // cfg.kv_sharing.num_recursion
        elif cfg.kv_sharing.sharing in ["middle_cycle"]:
            base_depth = (self.config.num_hidden_layers - 2) // cfg.kv_sharing.num_recursion

        if "kv_sharing" in cfg:
            kwargs = {
                "enable": cfg.kv_sharing.enable,
                "base_depth": base_depth,
                "num_recursion": cfg.kv_sharing.num_recursion,
                "sharing": cfg.kv_sharing.sharing,
                "update_cache": cfg.kv_sharing.update_cache if "update_cache" in cfg.kv_sharing else False,
            }
            self.model.config.kv_sharing = kwargs
        else:
            self.model.config.kv_sharing = None

    def route(self, layer_idx):
        pass

    def assemble(self, layer_idx):
        pass

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LlamaMorBaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            # TODO: Recursive Dynamic Cache Check
            if hasattr(self.config, "kv_sharing") and self.config.kv_sharing is not None:
                kv_kwargs = self.config.kv_sharing
                past_key_values = RecursiveDynamicCache(
                    kv_kwargs["base_depth"],
                    kv_kwargs["num_recursion"],
                    kv_kwargs["sharing"],
                    kv_kwargs.get("update_cache", False)
                )
            else:
                past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, attention_mask, position_embeddings, position_ids, past_key_values = self.route(
                decoder_layer
            )

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

            hidden_states = self.assemble()  # assemble routed

        hidden_states = self.norm(hidden_states)
        return LlamaMorBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LlamaMorForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaMorModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
