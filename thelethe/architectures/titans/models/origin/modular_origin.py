from typing import Any, Mapping, Optional, Union, Unpack, Literal, Callable
from collections import defaultdict
import copy

import torch
import torch.nn.functional as F
from torch import nn

from ...modeling_layers import (
    GenericForSequenceClassification,
    GenericForTokenClassification,
)
from ....protogenois.models.ttt_linear.modeling_ttt_linear import (
    TTTLinearCache,
    TTTRMSNorm,
    TTTSwiGluMLP,
    TTTRotaryEmbedding,
    TTTMultiheadLayerNorm,
    TTTMultiheadLinear,
    TTTDynamicLearningGate,
    TTTLinearAdaptationState,
    TTTLinearAdaptation,
    TTTLinearLayer,
    TTTLinearOutput,
    TTTLinearCausalLMOutput,
    TTTLinearModel,
    TTTLinearForCausalLM,
    TTTLinearForImageClassification
)
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer

from ...configuration_utils import PretrainedTitansConfig, TitansVariants, layer_type_validation
from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import PreTrainedTitansModel
from .....ops.scan_ops import associative_scan
from ...utils import logging


logger = logging.get_logger(__name__)


class OriginConfig(PretrainedTitansConfig):
    r"""
    This is the configuration class to store the configuration of a [`OriginModel`]. It is used to instantiate a Titans
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Titans 1.5B.

    Configuration objects inherit from [`PretrainedTitansConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 5504):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        mini_batch_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the mini batch normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        variant (`Union[TitansVariants, Literal["lmm", "mac", "mae", "mag", "mal"]]`, *optional*, defaults to `TitansVariants.LMM`):
            The Titans memory type variant of the model to use.
        memory_depth (`int`, *optional*, defaults to 4):
            The memory depth of the Titans model.
        fixed_memory_size (`int`, *optional*, defaults to None):
            The fixed memory size of the Titans model. If None, the fixed memory size is set to same amount as memory depth.
            To disable fixed memory, set this to 0.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        adapt_base_lr (`float`, *optional*, defaults to 1.0): base learning rate for TTT learner
        chunk_size (`int`, *optional*, defaults to 16): chunk size (mini-batch size) for TTT learner
        scan_checkpoint_group_size (`int`, *optional*, defaults to 0):
            gradient checkpoint group size on seq dimension, 0 means no checkpointing.

    ```python
    >>> from thelethe.titans import OriginModel, OriginConfig

    >>> # Initializing a Titans-Origin 1.5B model config
    >>> configuration = OriginConfig()

    >>> # Initializing a Titans-Origin 1.5B model from config
    >>> model = OriginModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "titans_origin"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int | tuple[int, int] = 5504,
        num_hidden_layers: int | tuple[int, int] = 6,
        num_attention_heads: int | tuple[int, int] = 32,
        hidden_act: str | Callable | tuple[str | Callable, str | Callable] = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        mini_batch_eps: float = 1e-6,
        use_cache: bool = True,
        variant: TitansVariants | Literal["lmm", "mac", "mae", "mag", "mal"] = TitansVariants.LMM,
        memory_depth: int = 4,
        fixed_memory_size: int | None = None,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = True,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        mlp_bias: bool = False,
        adapt_base_lr: float | tuple[float, ...] = 1.0,
        adapt_base_momentum: float | tuple[float, ...] = 0.9,
        adapt_base_weight_decay: float | tuple[float, ...] = 0.01,
        chunk_size: int = 128,
        scan_checkpoint_group_size: int = 0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        attention_layer_types: list | None = None,
        sliding_window: int | None = None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.mini_batch_eps = mini_batch_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.mlp_bias = mlp_bias
        self.adapt_base_lr = adapt_base_lr
        self.adapt_base_momentum = adapt_base_momentum
        self.adapt_base_weight_decay = adapt_base_weight_decay
        self.chunk_size = chunk_size
        self.fixed_memory_size = fixed_memory_size

        self.scan_checkpoint_group_size = scan_checkpoint_group_size

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        if attention_layer_types is None:
            if sliding_window is None:
                attention_layer_types = ["full_attention"] * self.num_hidden_layers
            else:
                pattern = kwargs.get("sliding_window_pattern", 6)
                attention_layer_types = [
                    "sliding_attention" if bool((i + 1) % pattern) else "full_attention"
                    for i in range(self.num_hidden_layers)
                ]
        self.layer_types = attention_layer_types  # for Gemma3 compatibility
        self.attention_layer_types = attention_layer_types
        self.sliding_window = sliding_window

        rope_config_validation(self)
        layer_type_validation(self.layer_types)
        super().__init__(
            variant=variant,
            memory_depth=memory_depth,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class OriginCache(TTTLinearCache):
    """
    Fast-Weight cache designed for models that dynamically updates and stores their weights and gradients
    during a forward pass.

    Unlike standard caches that store key/value states to prevent re-computation, this cache holds the evolving
    learning state of the model. It enables the model to learn and adapt in-context as it processes a sequence.
    The cache maintains two primary types of information for each layer's learnable parameters: the current
    parameter `states` and the `grad` values for the next update.

    Parameters:
        config (`PretrainedConfig`):
            The model configuration, used to infer hyperparameters like the number of layers, hidden size settings.
        batch_size (`int`):
            The number of sequences in the input batch. The cache tensors will be initialized with this
            batch dimension.
        layers (`torch.nn.ModuleList`):
            The list of the model's layers (`Block` modules). This is used to access the initial fast weights
            for initializing the cache states.
        device (`torch.device`):
            The device (e.g., "cuda" or "cpu") on which the cache tensors will be allocated.

    Attributes:
        state_dict (`dict`):
            The core data store for the fast weights. It's a nested dictionary with the structure:
            `{"parameter_name_states/grad": {layer_idx: tensor}}`.
        conv_states_dict (`dict`):
            A dictionary that holds the states for the convolutional layers, if they are enabled in the config.
    """
    layer_list_key = ["self_adapt", "self_rettn"]

    def __init__(self, config: PretrainedTitansConfig, batch_size: int, layers: nn.ModuleList, device: torch.device):
        self.chunk_size = config.chunk_size
        self.memory_depth = config.memory_depth
        self.param_names = [
            f"layers.{i}.weight" for i in range(self.memory_depth)
        ] + [
            f"layers.{i}.bias" for i in range(self.memory_depth)
        ] + [
            f"layers.{i}.past_surprise" for i in range(self.memory_depth)
        ]

        self.token_len = 0
        self.state_dict = defaultdict(dict)
        self.conv_states_dict = defaultdict(dict)
        logger.info(f"Creating cache of size: {batch_size}")

        for layer_idx in range(config.num_hidden_layers):
            for name in self.param_names:
                _, memory_idx, memory_type = name.split(".")
                seq_modeling_layer = None
                for key in self.layer_list_key:
                    try:
                        seq_modeling_layer = getattr(layers[layer_idx], key)
                    except AttributeError:
                        pass
                if seq_modeling_layer is None:
                    raise AttributeError(f"Layer {layer_idx} does not have any of the keys {self.layer_list_key}")
                weight = getattr(seq_modeling_layer.neural_memory.layers[memory_idx], memory_type)

                tiled_weight = torch.tile(weight.unsqueeze(0), (batch_size,) + (1,) * weight.dim()).to(device)
                self.state_dict[name][layer_idx] = tiled_weight


class TitansRMSNorm(TTTRMSNorm):
    pass


class TitansSwiGluMLP(TTTSwiGluMLP):
    pass


class TitansRotaryEmbedding(TTTRotaryEmbedding):
    pass


class TitansMultiheadLayerNorm(TTTMultiheadLayerNorm):
    pass


class TitansMultiheadLinear(TTTMultiheadLinear):
    pass


class TitansMomentumBasedSurpriseGate(TTTDynamicLearningGate):
    def __init__(
        self, num_heads: int, head_dim: int, chunk_size: int,
        adapt_base_lr: float, adapt_base_momentum: float, adapt_base_weight_decay: float
    ):
        super().__init__(num_heads, head_dim, chunk_size, adapt_base_lr)

        self.adapt_base_momentum = adapt_base_momentum
        self.adapt_base_weight_decay = adapt_base_weight_decay

        # [head_dim, 1] -> [num_heads, head_dim, 1]
        target_shape_per_head = (self.head_dim, 1)
        linear_bias_data = nn.Linear(self.head_dim, 1, bias=True).bias.data
        # alpha
        self.alpha = nn.Parameter(torch.stack(
            [torch.normal(0, 0.02, size=target_shape_per_head) for _ in range(self.num_heads)],
            dim=0
        ))
        self.alpha_bias = nn.Parameter(torch.stack(
            [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
            dim=0
        ).unsqueeze(-1))
        # theta
        self.theta = nn.Parameter(torch.stack(
            [torch.normal(0, 0.02, size=target_shape_per_head) for _ in range(self.num_heads)],
            dim=0
        ))
        self.theta_bias = nn.Parameter(torch.stack(
            [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
            dim=0
        ).unsqueeze(-1))
        # eta
        self.eta = nn.Parameter(torch.stack(
            [torch.normal(0, 0.02, size=target_shape_per_head) for _ in range(self.num_heads)],
            dim=0
        ))
        self.eta_bias = nn.Parameter(torch.stack(
            [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
            dim=0
        ).unsqueeze(-1))

    def __repr__(self):
        return f"{self.__class__.__name__}(momentum={self.adapt_base_momentum}, decay={self.adapt_base_weight_decay}, lr={self.adapt_base_lr})"

    def forward(self, x):
        current_mini_batch_size = x.shape[-2]
        token_eta = self.token_idx.view(1, 1, current_mini_batch_size, 1)

        # For Momentary (Current Input) Surprise [B, num_heads, mini_batch_size, 1]
        momentary = torch.einsum("bhkc,hcd->bhkd", x, self.theta) + self.theta_bias.view(1, self.num_heads, 1, 1)
        momentary_eta = self.adapt_base_lr * F.sigmoid(momentary) / self.head_dim

        # For Past Surprise [B, num_heads, mini_batch_size, 1]
        momentum = torch.einsum("bhkc,hcd->bhkd", x, self.eta) + self.eta_bias.view(1, self.num_heads, 1, 1)
        momentum = self.adapt_base_momentum * F.sigmoid(momentum)

        # Forget Gate [B, num_heads, mini_batch_size, 1]
        decay = torch.einsum("bhkc,hcd->bhkd", x, self.alpha) + self.alpha_bias.view(1, self.num_heads, 1, 1)
        decay = self.adapt_base_weight_decay * F.sigmoid(decay)

        return token_eta, momentary_eta, momentum, decay


class OriginAdaptationState(TTTLinearAdaptationState):
    def __init__(
        self, batch_size: int, chunk_size: int, num_heads: int, head_dim: int, past_surprise: list[torch.Tensor] | torch.Tensor | None,
        memory: Union[nn.ModuleList, list[torch.Tensor, torch.Tensor], list[list[torch.Tensor], list[torch.Tensor]]],
        norm: TTTMultiheadLayerNorm, lr_gate: TTTDynamicLearningGate, is_vectorized: Optional[bool] = None
    ):
        super().__init__(batch_size, chunk_size, num_heads, head_dim, memory, norm, lr_gate, is_vectorized)
        self.past_surprise = past_surprise


class OriginAdaptation(TTTLinearAdaptation):
    """Self-Adaptation"""

    @staticmethod
    def struct_details(num_heads: int, head_dim: int, memory_depth: int):
        return [
            dict(num_heads=num_heads, in_features=head_dim, out_features=head_dim)
            for _ in range(memory_depth)
        ]

    def __init__(self, config: OriginConfig, layer_idx: int):
        super(nn.Module, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.width = config.hidden_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        self.chunk_size = config.chunk_size
        self.memory_depth = config.memory_depth

        self.adapt_base_momentum = config.adapt_base_momentum[layer_idx]
        self.adapt_base_weight_decay = config.adapt_base_weight_decay[layer_idx]
        self.adapt_base_lr = config.adapt_base_lr[layer_idx]

        self.past_surprise = None  # always None cause past surprise is loaded after state branch

        self.q_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)

        self.lr_gate = TitansMomentumBasedSurpriseGate(
            self.num_heads, self.head_dim, self.chunk_size,
            self.adapt_base_lr, self.adapt_base_momentum, self.adapt_base_weight_decay
        )
        self.shared_norm = TitansMultiheadLayerNorm(self.num_heads, self.head_dim, self.config.mini_batch_eps)
        self.neural_memory = nn.ModuleList([
            TitansMultiheadLinear(**struct)
            for struct in self.struct_details(self.num_heads, self.head_dim, self.memory_depth)
        ])

        self.post_norm = nn.LayerNorm(self.width, eps=1e-6)

    @staticmethod
    def adapt_step(carry, xs):
        # Projected Inputs
        segment_q, segment_k, segment_v = xs.unbind(dim=2)
        past_surprise = carry.past_surprise

        # Reconstruction Task
        _, reconstructed = carry(segment_k, output_hidden_states=True)
        reconstructed = [segment_k] + reconstructed
        reconstruction_target = segment_v

        # Calculate Gradient
        token_eta, momentary_eta, past_eta, forget_eta = carry.lr_gate(segment_k)  # token_lr, lr, momentum, weight_decay
        lr = token_eta * momentary_eta
        lr_matrix = token_eta @ momentary_eta.transpose(-2, -1)
        deltas, gradients = carry.backward(reconstructed, reconstruction_target, eta=lr)

        # Data Organization
        depth = carry.depth
        momentary_surprise = torch.stack(gradients, dim=0)  # [depth, B, nh, K, f]
        momentary_surprise = momentary_surprise.transpose(0, 3)  # [K, depth, B, nh, f]
        if past_surprise is None:  # initial surprise
            past_surprise = torch.zeros(depth, carry.batch_size, carry.num_heads, carry.head_dim, device=momentary_surprise.device)  # [depth, B, nh, f]
        # [B, nh, K, 1] -> [depth, B, nh, K, 1] -> [K, depth, B, nh, 1]
        past_eta_expanded = past_eta.unsqueeze(0).expand(depth, -1, -1, -1, -1).transpose(0, 3)
        momentary_eta_expanded = momentary_eta.unsqueeze(0).expand(depth, -1, -1, -1, -1).transpose(0, 3)

        # Associate Scan each mini-batch token for accumulative new surprise
        #   new_surprise = past_eta * past_surprise - momentary_eta * momentary_surprise
        #   S_t = η_t * S_{t-1} - θ_t * u_t
        #   Represent as: S_t = η_t * S_{t-1} + (-θ_t * u_t)
        multipliers = past_eta_expanded  # η_t: [K, depth, B, nh, 1]
        additives = -momentary_eta_expanded * momentary_surprise  # -θ_t * u_t: [K, depth, B, nh, f]

        def combine_fn(left, right):
            """ Associative combine function for momentum updates
            Combines: S = right_mult * (left_mult * S_0 + left_add) + right_add
                        = (right_mult * left_mult) * S_0 + (right_mult * left_add + right_add)
            """
            left_mult, left_add = left    # Previous: (η_prev, additive_prev)
            right_mult, right_add = right  # Current: (η_curr, additive_curr)

            # Combined multiplier: η_combined = η_curr * η_prev
            combined_mult = right_mult * left_mult  # [depth, B, nh, 1]

            # Combined additive: additive_combined = η_curr * additive_prev + additive_curr
            # Broadcasting: [depth, B, nh, 1] * [depth, B, nh, f] + [depth, B, nh, f]
            combined_add = right_mult * left_add + right_add  # [depth, B, nh, f]

            return combined_mult, combined_add

        final_mult, final_add = associative_scan(
            combine_fn=combine_fn,
            xs=(multipliers, additives),
            dim=0  # scan for each token K
        ) # [K, depth, B, nh, f]
        # Broadcasting: [K, depth, B, nh, 1] * [1, depth, B, nh, f] + [K, depth, B, nh, f]
        past_surprise_expanded = past_surprise.unsqueeze(0).expand_as(final_add)  # [K, depth, B, nh, f]
        new_surprises = final_mult * past_surprise_expanded + final_add  # [K, depth, B, nh, f]

        # Apply Gradients to fast weight (+ weight decay)
        forget_factor = (1 - forget_eta).mean(dim=(1, 2), keepdim=True)  # [depth, nh, 1, 1]
        # M_t = (1 - α_t) * M_{t-1} + S_t
        gated_deltas = (
            forget_factor.unsqueeze(-1) * carry.weights + (deltas[0] + new_surprises.unsqueeze(-1)),  # [depth, nh, f, f]
            forget_factor * carry.biases + (deltas[1] + new_surprises.unsqueeze(-2))
        ) if carry.is_vectorized else (
            [delta + decay * w + surp for delta, decay, w, surp in zip(deltas[0], forget_factor.unsqueeze(-1), carry.weights, new_surprises.unsqueeze(-1))],
            [delta + decay * w + surp for delta, decay, w, surp in zip(deltas[1], forget_factor, carry.biases, new_surprises.unsqueeze(-2))]
        )

        # Generate Hidden States
        hidden_states = segment_q
        for idx, (val, gradient) in enumerate(zip(reconstructed, gradients)):
            attention_mask = torch.tril(hidden_states @ val.transpose(-2, -1))  # [B,nh,K,K]
            # [B,nh,K,f] @ [B,nh,f,f] + [B,nh,K,f] - ([B,nh,K,K] * [B,nh,K,K]) @ [B,nh,K,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
            update_term = (lr_matrix * attention_mask + torch.tril(lr_matrix)) @ gradient
            hidden_states = carry[idx](hidden_states) - update_term
            if idx < depth - 1:
                hidden_states = carry.activate(hidden_states)
        hidden_states = segment_q + carry.norm(hidden_states)  # residual connection

        # Update Carry
        next_carry = carry.step(gated_deltas)

        return next_carry, hidden_states.unsqueeze(2).expand(-1, -1, 3, -1, -1)  # match to xs shape for scan

    def branch(self, batch_size: int):
        """Branch the neural memory for Test-time learning"""
        return OriginAdaptationState(
            batch_size, self.chunk_size, self.num_heads, self.head_dim,
            self.past_surprise, self.neural_memory, self.shared_norm, self.lr_gate
        )


class OriginRetention(OriginAdaptation):
    """ Self-Retention
    The self-adaptation layer changes its role to memory retention when implemented in the "Memory As ??" pattern.
    """
    pass


class OriginLayer(TTTLinearLayer):
    def __init__(self, config: OriginConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size

        self.self_adapt = OriginAdaptation(config=config, layer_idx=layer_idx)
        self.mlp = TitansSwiGluMLP(config)

        self.seq_norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx


class OriginMemoryLayer(OriginLayer):
    def __init__(self, config: OriginConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size

        self.self_rettn = OriginRetention(config=config, layer_idx=layer_idx)
        self.mlp = TitansSwiGluMLP(config)

        self.seq_norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx


class OriginCoreLayer(Gemma3DecoderLayer):
    pass


class OriginPreTrainedTitansModel(PreTrainedTitansModel):
    config_class = OriginConfig
    base_model_prefix = "titans_origin"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OriginLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class TitansOutput(TTTLinearOutput):
    pass


class TitansCausalLMOutput(TTTLinearCausalLMOutput):
    pass


class OriginModel(OriginPreTrainedTitansModel, TTTLinearModel):
    def validate_config(self):
        config = self.config

        # Config validation
        if isinstance(config.adapt_base_lr, tuple) or isinstance(config.adapt_base_lr, list):
            if len(config.adapt_base_lr) != config.num_hidden_layers:
                raise ValueError("adapt_base_lr must be a single value or a list of length num_hidden_layers")
        else:
            config.adapt_base_lr = [config.adapt_base_lr] * config.num_hidden_layers

        if isinstance(config.adapt_base_momentum, tuple) or isinstance(config.adapt_base_momentum, list):
            if len(config.adapt_base_momentum) != config.num_hidden_layers:
                raise ValueError("adapt_base_momentum must be a single value or a list of length num_hidden_layers")
        else:
            config.adapt_base_momentum = [config.adapt_base_momentum] * config.num_hidden_layers

        if isinstance(config.adapt_base_weight_decay, tuple) or isinstance(config.adapt_base_weight_decay, list):
            if len(config.adapt_base_weight_decay) != config.num_hidden_layers:
                raise ValueError("adapt_base_weight_decay must be a single value or a list of length num_hidden_layers")
        else:
            config.adapt_base_weight_decay = [config.adapt_base_weight_decay] * config.num_hidden_layers

    def __init__(self, config: OriginConfig):
        super().__init__(config)
        self.validate_config()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.chunk_size = config.chunk_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([OriginLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TitansRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class OriginVariantModel(OriginModel):
    supports_gradient_checkpointing = True
    _no_split_modules = ["OriginMemoryLayer", "OriginCoreLayer"]

    def validate_config(self):
        super().validate_config()
        config = self.config

        # Config split
        memory_config, core_config = copy.copy(config), copy.copy(config)
        if isinstance(config.intermediate_size, tuple) or isinstance(config.intermediate_size, list):
            memory_config.intermediate_size, core_config.intermediate_size = config.intermediate_size
        if isinstance(config.num_hidden_layers, tuple) or isinstance(config.num_hidden_layers, list):
            memory_config.num_hidden_layers, core_config.num_hidden_layers = config.num_hidden_layers
        if isinstance(config.num_attention_heads, tuple) or isinstance(config.num_attention_heads, list):
            memory_config.num_attention_heads, core_config.num_attention_heads = config.num_attention_heads
        if isinstance(config.hidden_act, tuple) or isinstance(config.hidden_act, list):
            memory_config.hidden_act, core_config.hidden_act = config.hidden_act
            core_config.hidden_activation = core_config.hidden_act

        return memory_config, core_config

    def __init__(self, config: OriginConfig):
        super().__init__(config)
        memory_config, core_config = self.validate_config()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.chunk_size = config.chunk_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.fixed_memories = nn.Parameter(torch.zeros([config.fixed_memory_size, config.hidden_size]))
        self.memories = nn.ModuleList([OriginMemoryLayer(memory_config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.cores = nn.ModuleList([OriginCoreLayer(core_config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

        self.norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TitansRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def layers(self):
        return self.memories + self.cores


class OriginMACModel(OriginModel):
    base_model_prefix = "titans_origin_mac"


class OriginMAEModel(OriginModel):
    base_model_prefix = "titans_origin_mae"

    @property
    def layers(self):
        for layer in self.memories:
            yield layer
        for layer in self.cores:
            yield layer


class OriginMALModel(OriginModel):
    base_model_prefix = "titans_origin_mal"


class OriginMAGModel(OriginModel):
    base_model_prefix = "titans_origin_mag"


class OriginForCausalLM(OriginPreTrainedTitansModel, TTTLinearForCausalLM):
    model_variants = {
        TitansVariants.LMM.value: OriginModel,
        TitansVariants.MAC.value: OriginMACModel, TitansVariants.MAE.value: OriginMAEModel,
        TitansVariants.MAL.value: OriginMALModel, TitansVariants.MAG.value: OriginMAGModel,
    }

    def __init__(self, config: OriginConfig):
        super().__init__(config)
        self.variant = config.variant if isinstance(config.variant, str) else config.variant.value
        if self.variant not in self.model_variants.keys():
            raise ValueError(f"Cannot recognize variant type `{self.variant}` for `{self.__class__.__name__}`")
        self.model = self.model_variants[self.variant](config)

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class OriginForSequenceClassification(GenericForSequenceClassification, OriginPreTrainedTitansModel):
    pass


class OriginForTokenClassification(GenericForTokenClassification, OriginPreTrainedTitansModel):
    pass


class OriginForImageClassification(TTTLinearForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        self.vit = OriginModel(config)


__all__ = [
    "OriginConfig",
    "OriginCache",
    "OriginAdaptation",
    "OriginLayer",
    "OriginPreTrainedTitansModel",
    "OriginModel",
    "OriginMACModel",
    "OriginMAEModel",
    "OriginMALModel",
    "OriginMAGModel",
    "OriginForCausalLM",
    "OriginForSequenceClassification",
    "OriginForTokenClassification",
    "OriginForImageClassification"
]
