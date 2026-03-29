# coding=utf-8
# Copyright 2025 test-time-training and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This implementation is based on the original TTT-Linear/MLP paper:
# "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
# by Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang,
# Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto,
# and Carlos Guestrin.
# Paper: https://arxiv.org/abs/2407.04620
# Original PyTorch implementation: https://github.com/test-time-training/ttt-lm-pytorch
# Original JAX implementation: https://github.com/test-time-training/ttt-lm-jax
#
# Original code is licensed under MIT License:
# Copyright (c) 2024 test-time-training
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Transformers integration and performance optimizations:
# Copyright (c) 2025 RetentionLabs team
#
# Licensed under Apache License, Version 2.0
from typing import Any, Dict, Optional, Tuple, Union, Unpack, Mapping
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ...utils.scan_ops import scan
from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...modeling_outputs import ModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...modeling_layers import (
    GenericForSequenceClassification,
    GenericForTokenClassification,
)
from ...generation import GenerationMixin
from ..llama.modeling_llama import (
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb
)
from ..vit.modeling_vit import ViTForImageClassification
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging


logger = logging.get_logger(__name__)


def gelu(x):
    return F.gelu(x, approximate="tanh")


# Modified from https://github.com/NVIDIA/Megatron-LM/blob/e33c8f78a35765d5aa37475a144da60e8a2349d1/megatron/core/fusions/fused_bias_gelu.py#L26
def gelu_derivative(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff


class TTTLinearConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TTTModel`]. It is used to instantiate an TTT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the TTTLinear 1.5B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
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
        qkv_conv (`bool`, *optional*, defaults to `False`): Whether the model use conv while qkv projection
        conv_kernel (`int`, *optional*, defaults to 4): kernel size of the conv layer
        scan_checkpoint_group_size (`int`, *optional*, defaults to 0):
            gradient checkpoint group size on seq dimension, 0 means no checkpointing.
            In JAX implementation, we set it 4, which means we group 4 chunks together in 1 gradient checkpointg to save memory.

    ```python
    >>> from transformers import TTTLinearModel, TTTLinearConfig

    >>> # Initializing a TTT ttt_linear 1.5b style configuration
    >>> configuration = TTTLinearConfig()

    >>> # Initializing a model from the ttt_linear 1.5b style configuration
    >>> model = TTTLinearModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ttt_linear"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=5504,
        num_hidden_layers=24,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        mini_batch_eps=1e-6,
        use_cache=False,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        mlp_bias=False,
        adapt_base_lr=1.0,
        chunk_size=16,
        qkv_conv=False,
        conv_kernel=4,
        scan_checkpoint_group_size=0,
        **kwargs,
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
        self.chunk_size = chunk_size

        self.qkv_conv = qkv_conv
        self.conv_kernel = conv_kernel
        self.scan_checkpoint_group_size = scan_checkpoint_group_size

        self.memory_depth = 1  # TTTLinearAdaptation depth

        rope_config_validation(self)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class TTTLinearCache:
    """
    Fast-Weight cache designed for models that dynamically updates and stores their weights and gradients
    during a forward pass.

    Unlike standard caches that store key/value states to prevent re-computation, this cache holds the evolving
    learning state of the model. It enables the model to learn and adapt in-context as it processes a sequence.
    The cache maintains two primary types of information for each layer's learnable parameters: the current
    parameter `states` and the `grad` values for the next update.

    Parameters:
        config (`PretrainedConfig`):
            The model configuration, used to infer hyperparameters like the number of layers, hidden size,
            and convolution settings.
        batch_size (`int`):
            The number of sequences in the input batch. The cache tensors will be initialized with this
            batch dimension.
        layers (`torch.nn.ModuleList`):
            The list of the model's layers (`Block` modules). This is used to access the initial fast weights
            for initializing the cache states.
        device (`torch.device`):
            The device (e.g., "cuda" or "cpu") on which the cache tensors will be allocated.

    Attributes:
        state_params_dict (`dict`):
            The core data store for the fast weights. It's a nested dictionary with the structure:
            `{"parameter_name_states/grad": {layer_idx: tensor}}`.
        conv_states_dict (`dict`):
            A dictionary that holds the states for the convolutional layers, if they are enabled in the config.
    """
    layer_list_key = "self_adapt"

    def __init__(self, config: PretrainedConfig, batch_size: int, layers: nn.ModuleList, device: torch.device):
        self.chunk_size = config.chunk_size
        self.memory_depth = config.memory_depth
        self.param_names = [f"layers.{i}.weight" for i in range(self.memory_depth)] + [f"layers.{i}.bias" for i in range(self.memory_depth)]

        self.token_len = 0
        self.state_dict = defaultdict(dict)
        self.conv_states_dict = defaultdict(dict)
        logger.info(f"Creating cache of size: {batch_size}")

        for layer_idx in range(config.num_hidden_layers):
            for name in self.param_names:
                _, memory_idx, memory_type = name.split(".")
                seq_modeling_layer = getattr(layers[layer_idx], self.layer_list_key)
                weight = getattr(seq_modeling_layer.neural_memory.layers[memory_idx], memory_type)

                tiled_weight = torch.tile(weight.unsqueeze(0), (batch_size,) + (1,) * weight.dim()).to(device)
                self.state_dict[name][layer_idx] = tiled_weight

    def update(self, py_tree, layer_idx):
        for name in self.param_names:
            self.state_dict[name][layer_idx].copy_(py_tree[name])

    def __setitem__(self, layer_idx, py_tree):
        self.update(py_tree, layer_idx)

    def __getitem__(self, layer_idx):
        return {name: self.state_dict[name][layer_idx] for name in self.state_dict}



class TTTRMSNorm(LlamaRMSNorm):
    pass


class TTTSwiGluMLP(LlamaMLP):
    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class TTTRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config, device: Optional[torch.device] = None):
        super().__init__(config, device)


class TTTMultiheadLayerNorm(nn.Module):
    """Multi-head layer normalization which can calculate norm on multiple heads in a one pass"""

    def __init__(self, num_heads: int, dim: Union[int, list[int], torch.Size], eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        _ = nn.LayerNorm(dim)
        ln_weight_data, ln_bias_data = _.weight.data, _.bias.data

        # prepending head dim -> [num_heads, width]
        self.gamma = nn.Parameter(ln_weight_data[None, None, :].expand(num_heads, 1, -1).clone())
        self.beta = nn.Parameter(ln_bias_data[None, None, :].expand(num_heads, 1, -1).clone())
        self.eps = eps

    def __repr__(self):
        return f"{self.__class__.__name__}(({self.num_heads}, {self.dim}), eps={self.eps})"

    def forward(self, x, return_detailed=False):
        # Mean and variance computation
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalization
        std = torch.sqrt(var + self.eps)
        x_hat = (x - mu) / std

        # Scale and shift
        y = self.gamma * x_hat + self.beta

        if return_detailed:
            return y, x_hat, std
        return y


class TTTDynamicLearningGate(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, chunk_size: int, adapt_base_lr: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.adapt_base_lr = adapt_base_lr

        # Configuration of dynamic eta (Sec. 2.7)
        # - token_idx is a scale factor that scale the summation in Eqn. 4
        token_idx = 1.0 / torch.arange(1, self.chunk_size + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)
        # - make the scale factor learnable
        self.alpha = nn.Parameter(torch.zeros((self.chunk_size,)))

        # [head_dim, 1]
        target_shape_per_head = (self.head_dim, 1)
        linear_bias_data = nn.Linear(self.head_dim, 1, bias=True).bias.data
        # prepending head dim -> [num_heads, head_dim, 1]
        self.theta = nn.Parameter(torch.stack(
            [torch.normal(0, 0.02, size=target_shape_per_head) for _ in range(self.num_heads)],
            dim=0
        ))
        # [num_heads, 1]
        self.theta_bias = nn.Parameter(torch.stack(  # init bias to 0 following original JAX impl.
            [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
            dim=0
        ).unsqueeze(-1))

    def __repr__(self):
        return f"{self.__class__.__name__}(heads={self.num_heads}, head_dim={self.head_dim}, lr={self.adapt_base_lr})"

    def forward(self, x):
        current_mini_batch_size = x.shape[-2]

        # [B, num_heads, mini_batch_size, 1]
        learning_rate = torch.einsum("bhkc,hcd->bhkd", x, self.theta) + self.theta_bias.view(1, self.num_heads, 1, 1)
        learning_rate = F.sigmoid(learning_rate)
        learning_rate_eta = self.adapt_base_lr * learning_rate / self.head_dim

        # [K]
        token_idx = self.token_idx[:current_mini_batch_size] + self.alpha[:current_mini_batch_size]
        token_idx = torch.clamp_min(token_idx, 0.0)  # token idx should be greater than 0

        # NOTE: token_eta is a scale factor that applies to each token in the mini-batch
        # [1, 1, K, 1], auto-broadcastable to [B, H, K, 1]
        token_eta = token_idx.view(1, 1, current_mini_batch_size, 1)

        return token_eta, learning_rate_eta


class TTTMultiheadLinearMixin:
    def __init__(self):
        super().__init__()  # To follow MRO
        self.use_bias = None
        self.weight = None
        self.bias = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        output = x @ self.weight
        if self.use_bias:
            return output + self.bias
        return output


class TTTMultiheadLinear(TTTMultiheadLinearMixin, nn.Module):
    """Expressive Multihead Linear layer"""

    def __init__(self, num_heads: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.use_bias = bias
        self.num_heads = num_heads
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.normal(0, 0.02, size=(num_heads, in_features, out_features)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_heads, 1, out_features))
        else:
            self.register_parameter("bias", None)


class TTTLinearAdaptationState:
    """Memory Carry State while Adaptive Learning"""

    @staticmethod
    def rasterize(weight: torch.Tensor, bias: torch.Tensor):
        raw = TTTMultiheadLinearMixin()
        raw.use_bias = not torch.isnan(bias).any()
        raw.weight = weight
        if raw.use_bias:
            raw.bias = bias
        return raw

    def __init__(
        self, batch_size: int, chunk_size: int, num_heads: int, head_dim: int,
        memory: Union[nn.ModuleList, list[torch.Tensor, torch.Tensor], list[list[torch.Tensor], list[torch.Tensor]]],
        norm: TTTMultiheadLayerNorm, lr_gate: TTTDynamicLearningGate, is_vectorized: Optional[bool] = None
    ):
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        if isinstance(memory, nn.ModuleList):  # Initial Branching
            weights, biases = [], []

            for layer in memory:
                weight = torch.tile(layer.weight.unsqueeze(0), dims=(batch_size, 1, 1, 1))
                if layer.use_bias:
                    bias = torch.tile(layer.bias.unsqueeze(0), dims=(batch_size, 1, 1, 1))
                else:
                    bias = torch.tile(torch.tensor(float('nan')), dims=(batch_size, 1, 1, 1))
                weights.append(weight)
                biases.append(bias)

            try:  # try vectorize
                self.weights = torch.stack(weights, dim=0)
                self.biases = torch.stack(biases, dim=0)
                self.is_vectorized = True
            except RuntimeError:  # failed to vectorize (different weight shapes, ...)
                self.weights = weights
                self.biases = biases
                self.is_vectorized = False
        else:  # Inherited from previous state
            if is_vectorized is None:
                raise ValueError("is_vectorized flag must be provided for state updates.")
            self.is_vectorized = is_vectorized
            self.weights: torch.Tensor = memory[0]; self.biases: torch.Tensor = memory[1]

        self.layers = [self.rasterize(*layer) for layer in zip(self.weights, self.biases)]
        self.depth = len(self.layers)
        self.activate = gelu
        self.activate_derivative = gelu_derivative
        self.norm = norm
        self.lr_gate = lr_gate

    def state_dict(self, *args, **kwargs):
        states = {f'layers.{i}.weight': self.layers[i].weight for i in range(self.depth)}
        states.update({f'layers.{i}.bias': self.layers[i].bias for i in range(self.depth)})
        return states

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs):
        for key, val in state_dict.items():
            try:
                _, idx, typ = key.split(".")
                idx = int(idx)
                if typ == "weight":
                    self.layers[idx].weight = val
                elif typ == "bias":
                    self.layers[idx].bias = val
                else:
                    raise ValueError
            except ValueError:
                raise ValueError(f"Invalid memory state dict key {key} found.")

    def step(self, delta: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor, torch.Tensor]):
        if self.is_vectorized:
            new_state = [self.weights - delta[0], self.biases - delta[1]]
        else:
            new_weights = [w - d for w, d in zip(self.weights, delta[0])]
            new_biases = [b - d for b, d in zip(self.biases, delta[1])]
            new_state = [new_weights, new_biases]
        return TTTLinearAdaptationState(
            self.batch_size, self.chunk_size, self.num_heads, self.head_dim,
            new_state, self.norm, self.lr_gate, self.is_vectorized
        )

    def __getitem__(self, idx):
        return self.layers[idx]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, output_hidden_states=False):
        outs = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if output_hidden_states:
                outs.append(x)
            if idx < self.depth-1:
                x = self.activate(x)
        if output_hidden_states:
            return x, outs
        return x

    def backward(
        self, reconstructed: list[torch.Tensor], reconstruction_target: torch.Tensor, eta: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]]:
        """L2 loss & forward creation of fast weights"""
        (y, x_hat, std), gamma = self.norm(reconstructed[-1], return_detailed=True), self.norm.gamma
        D = reconstructed[-1].shape[-1]

        grad_output = y - (reconstruction_target - reconstructed[0])
        grad_x_hat = grad_output * gamma
        grad_x = ((1.0 / D) * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        ) / std)

        backward_grads = [grad_x]
        for fwd, layer in zip(reversed(reconstructed[1:-1]), reversed(self.layers[1:])):
            grad_n = backward_grads[0] @ layer.weight.transpose(-2, -1) * self.activate_derivative(fwd)
            backward_grads.append(grad_n)
        backward_grads = backward_grads[::-1]

        mini_batch_size = reconstructed[0].shape[-2]
        weight_deltas, bias_deltas = [], []
        # NOTE: The length of 'reconstructed' list is may larger (+1) than the others,
        #       but not sliced to avoid redundant operation (zip will stop iterate automatically)
        for fwd, layer, grad in zip(reconstructed, self.layers, backward_grads):
            delta_bias = None
            if self.chunk_size == mini_batch_size:  # Use Dual Form
                # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                delta_weight = (eta * fwd).transpose(-1, -2) @ grad
                if layer.use_bias: delta_bias = torch.sum(eta * grad, dim=-2, keepdim=True)
            else:  # Use Approx. Primal Form (same logic as dual form, but explicitly branch out for annotation purpose)
                delta_weight = (eta * fwd).transpose(-1, -2) @ grad
                if layer.use_bias: delta_bias = torch.sum(eta * grad, dim=-2, keepdim=True)

            weight_deltas.append(delta_weight)
            if delta_bias is None:  # If no bias, create a zero tensor
                weight_shape = delta_weight.shape
                bias_shape = weight_shape[:-2] + (1, weight_shape[-1])  # [B,nh,1,output_dim]
                delta_bias = torch.zeros(bias_shape, dtype=delta_weight.dtype, device=delta_weight.device)
            bias_deltas.append(delta_bias)

        if self.is_vectorized:
            weight_deltas = torch.stack(weight_deltas)
            bias_deltas = torch.stack(bias_deltas)

        return (weight_deltas, bias_deltas), backward_grads


class TTTLinearAdaptation(nn.Module):
    _private_modules = ["neural_memory"]

    @staticmethod
    def struct_details(num_heads: int, head_dim: int):
        return [
            dict(num_heads=num_heads, in_features=head_dim, out_features=head_dim)
        ]

    def __init__(self, config: TTTLinearConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.width = config.hidden_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        self.chunk_size = config.chunk_size
        self.conv_kernel = config.conv_kernel
        self.qkv_conv = config.qkv_conv

        self.q_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)

        if self.qkv_conv:  # depthwise conv
            self.conv_q = nn.Conv1d(
                self.head_dim, self.head_dim, groups=self.head_dim, bias=False,
                kernel_size=config.conv_kernel, padding=config.conv_kernel // 2  # same padding for non-causal conv
            )
            self.conv_k = nn.Conv1d(
                self.head_dim, self.head_dim, groups=self.head_dim, bias=False,
                kernel_size=config.conv_kernel, padding=config.conv_kernel // 2
            )
            self.conv_v = nn.Conv1d(
                self.head_dim, self.head_dim, groups=self.head_dim, bias=False,
                kernel_size=config.conv_kernel, padding=config.conv_kernel // 2
            )

        self.lr_gate = TTTDynamicLearningGate(
            self.num_heads, self.head_dim,
            self.chunk_size, self.config.adapt_base_lr
        )
        self.shared_norm = TTTMultiheadLayerNorm(self.num_heads, self.head_dim, self.config.mini_batch_eps)
        self.neural_memory = nn.ModuleList([
            TTTMultiheadLinear(**struct) for struct in self.struct_details(self.num_heads, self.head_dim)
        ])

        self.post_norm = nn.LayerNorm(self.width, eps=1e-6)

    def __repr__(self):
        lines = [self.__class__.__name__ + "("]

        for name, module in self.named_children():
            if name not in self._private_modules:
                lines.append(f"  ({name}): {repr(module)}")

        lines.append(")")
        return "\n".join(lines)

    @staticmethod
    def adapt_step(carry: TTTLinearAdaptationState, xs: torch.Tensor):
        # Projected Inputs
        # [B,nh,K,f], K=mini_batch_size
        xq_mini_batch, xk_mini_batch, xv_mini_batch = xs.unbind(dim=2)

        # Reconstruction Task
        # [B,nh,K,f] @ [B,nh,f,f] -> [B,nh,K,f]
        _, reconstructed = carry(xk_mini_batch, output_hidden_states=True)
        reconstructed = [xk_mini_batch] + reconstructed
        reconstruction_target = xv_mini_batch
        # token_eta: [1,1,K,1], lr_eta: [B,H,K,1]
        token_eta, lr_eta = carry.lr_gate(xk_mini_batch)
        eta_scalar = token_eta * lr_eta  # [B,h,K,1] * [B,h,K,1] -> [B,h,K,1] for backward
        eta_matrix = token_eta @ lr_eta.transpose(-2, -1)  # [B,h,K,1] @ [B,h,K,1] -> [B,h,K,K] for hidden state update
        deltas, gradients = carry.backward(reconstructed, reconstruction_target, eta=eta_scalar)

        # Generate Hidden States
        depth = len(carry.layers)
        hidden_states = xq_mini_batch
        for idx, (val, gradient) in enumerate(zip(reconstructed, gradients)):
            attention_mask = torch.tril(hidden_states @ val.transpose(-2, -1))  # [B,nh,K,K]
            # [B,nh,K,f] @ [B,nh,f,f] + [B,nh,K,f] - ([B,nh,K,K] * [B,nh,K,K]) @ [B,nh,K,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
            update_term = (eta_matrix * attention_mask + torch.tril(eta_matrix)) @ gradient
            hidden_states = carry[idx](hidden_states) - update_term
            if idx < depth - 1:
                hidden_states = carry.activate(hidden_states)
        hidden_states = xq_mini_batch + carry.norm(hidden_states)  # residual connection

        # Update Carry
        next_carry = carry.step(deltas)

        return next_carry, hidden_states.unsqueeze(2).expand(-1, -1, 3, -1, -1)  # match to xs shape for scan

    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TTTLinearCache] = None,
        mini_batch_size: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[torch.Tensor]:
        """
        Apply TTT (Test-Time Training) linear adaptation to the hidden states.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input hidden states from the previous layer.
            position_embeddings (`torch.Tensor`):
                Precomputed position embeddings (cos, sin) for rotary position encoding.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Position indices for each input token. If not provided, will be auto-generated.
            cache_params (`TTTLinearCache`, *optional*):
                Cache object containing the fast weights and states for efficient inference.
                If provided, enables incremental generation mode.
            mini_batch_size (`int`, *optional*):
                Size of mini-batches for TTT learning. If None, defaults to config.chunk_size.
                Must not exceed config.chunk_size. Used for online adaptation mode.
            **kwargs:
                Additional keyword arguments passed to the function.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`:
                The adapted hidden states after applying TTT learning.
        """
        if mini_batch_size is None:  # Batch Learning Process
            mini_batch_size = self.chunk_size
        elif mini_batch_size > self.chunk_size:
            raise ValueError("Mini-batch size cannot be greater than model chunk size configuration")

        B, L = hidden_states.shape[:2]
        num_heads = self.num_heads
        head_dim = self.head_dim
        num_mini_batch = int(L / mini_batch_size)

        # [B, L, C] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        XQ = self.q_proj(hidden_states).reshape(B, L, num_heads, head_dim).transpose(1, 2)
        XK = self.k_proj(hidden_states).reshape(B, L, num_heads, head_dim).transpose(1, 2)
        XV = self.v_proj(hidden_states).reshape(B, L, num_heads, head_dim).transpose(1, 2)

        # QKV Post Convolution
        if self.qkv_conv:
            XQ = self.conv_q(XQ)  # local pattern
            XK = self.conv_k(XK)  # key representation
            XV = self.conv_v(XV)  # value representation

        # RoPE
        cos, sin = position_embeddings
        XQ, XK = apply_rotary_pos_emb(XQ, XK, cos, sin, position_ids)

        # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        XQ = XQ.reshape(B, num_heads, num_mini_batch, mini_batch_size, head_dim)
        XK = XK.reshape(B, num_heads, num_mini_batch, mini_batch_size, head_dim)
        XV = XV.reshape(B, num_heads, num_mini_batch, mini_batch_size, head_dim)

        # [B, num_heads, 3, num_mini_batch, mini_batch_size, head_dim]
        stacked_qkv = torch.stack([XQ, XK, XV], dim=2)
        xs = stacked_qkv.permute(3, 0, 1, 2, 4, 5)

        init_params = self.branch(batch_size=B)
        if cache_params is not None:
            init_params.load_state_dict(cache_params[self.layer_idx])

        # input: [B, num_heads, num_mini_batch, mini_batch_size, f] -> [num_mini_batch, B, num_heads, mini_batch_size, f]
        # output_hidden_states: [num_mini_batch, B, num_heads, mini_batch_size, head_dim]
        scan_params = dict(combine_fn=self.adapt_step, init=init_params, xs=xs)
        try:
            last_params, output_hidden_states = scan(
                **scan_params,
                checkpoint_group=self.config.scan_checkpoint_group_size if self.training else 0
            )
        except TypeError:  # Using PyTorch official scan (Does not support checkpoint_group / Future use)
            last_params, output_hidden_states = scan(**scan_params)
        output_hidden_states = output_hidden_states[:, :, :, 0, :, :]  # [num_mini_batch, B, num_heads, chunk_size, head_dim]

        # [B, num_heads, L, C]
        if cache_params is not None:
            cache_params[self.layer_idx] = last_params.state_dict()

        # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] -> [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
        output_hidden_states = output_hidden_states.permute(1, 0, 3, 2, 4)
        # [B, L, C]
        output_hidden_states = output_hidden_states.reshape(B, L, self.width)

        output_hidden_states = self.post_norm(output_hidden_states)
        output_hidden_states = self.o_proj(output_hidden_states)

        return output_hidden_states

    def branch(self, batch_size: int):
        """Branch the neural memory for Test-time learning"""
        return TTTLinearAdaptationState(
            batch_size, self.chunk_size, self.num_heads, self.head_dim,
            self.neural_memory, self.shared_norm, self.lr_gate
        )


class TTTLinearLayer(nn.Module):
    def __init__(self, config: TTTLinearConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_adapt = TTTLinearAdaptation(config=config, layer_idx=layer_idx)
        self.mlp = TTTSwiGluMLP(config)

        self.seq_norm = TTTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = TTTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TTTLinearCache] = None,
        mini_batch_size: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.seq_norm(hidden_states)

        if "attention_mask" in kwargs:
            logger.warning_once(f"{self.__class__.__name__} does not use attention mask, but it is provided. It will be ignored.")
            kwargs.pop("attention_mask")  # TTTLinear does not use attention mask

        # TTT Adaptation Layer
        hidden_states = self.self_adapt(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            cache_params=cache_params,
            mini_batch_size=mini_batch_size,
            **kwargs
        )
        hidden_states = residual + hidden_states

        # Feed-Forward-Network
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TTTLinearPreTrainedModel(PreTrainedModel):
    config_class = TTTLinearConfig
    base_model_prefix = "ttt_linear"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TTTLinearLayer"]

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


@dataclass
class TTTLinearOutput(ModelOutput):
    """
    Class for the TTT model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`TTTLinearCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[TTTLinearCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class TTTLinearCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`TTTLinearCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[TTTLinearCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class TTTLinearModel(TTTLinearPreTrainedModel):
    def __init__(self, config: TTTLinearConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.chunk_size = config.chunk_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([TTTLinearLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = TTTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TTTRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TTTLinearCache] = None,
        mini_batch_size: Optional[int] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs]
    ) -> Union[Tuple, TTTLinearOutput]:
        """
        Forward pass through the TTT Linear model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence token in the position embeddings.
            cache_params (`TTTLinearCache`, *optional*):
                Cache object for storing fast weights and states during incremental generation.
                Contains the evolving learning state of the model for efficient inference.
            mini_batch_size (`int`, *optional*):
                Size of mini-batches for TTT adaptation. If None, uses config.chunk_size.
                Controls the granularity of test-time learning updates.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            use_cache (`bool`, *optional*):
                If set to `True`, cache_params is returned and can be used to speed up decoding.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a ModelOutput instead of a plain tuple.

        Returns:
            `Union[Tuple, TTTLinearOutput]`:
                Either a tuple of tensors or TTTLinearOutput containing:
                - last_hidden_state: The sequence of hidden-states at the output of the last layer
                - cache_params: The updated cache containing fast weights and states
                - hidden_states: Hidden-states of all layers (if output_hidden_states=True)
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if "attention_mask" in kwargs:
            logger.warning_once(f"{self.__class__.__name__} does not use attention mask, but it is provided. It will be ignored.")
            kwargs.pop("attention_mask")  # TTTLinear does not use attention mask

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if mini_batch_size is None:  # Batch Learning Process
            mini_batch_size = self.chunk_size
        elif mini_batch_size > self.chunk_size:
            raise ValueError("Mini-batch size cannot be greater than model chunk size configuration")

        rope_start_pos = 0
        if cache_params is None and use_cache:
            cache_params = TTTLinearCache(self.config, inputs_embeds.size(0), self.layers, self.device)
            rope_start_pos = cache_params.token_len

        if position_ids is None:
            position_ids = torch.arange(rope_start_pos, inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)

        hidden_states = inputs_embeds

        pad_length = 0
        if mini_batch_size != self.chunk_size:  # Online Adaptation Mode (do not pad)
            if hidden_states.shape[1] != mini_batch_size:
                raise ValueError("Input token count does not match to the online mini-batch size.")
        elif hidden_states.shape[1] % mini_batch_size > 0:  # Padding for batch process
            pad_length = mini_batch_size - (hidden_states.shape[1] % mini_batch_size)
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length))
            position_ids = F.pad(position_ids, (0, pad_length))

        position_embeddings = self.rotary_emb(input_ids, position_ids)

        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    position_embeddings,
                    position_ids,
                    cache_params,
                    mini_batch_size,
                    **kwargs
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    position_embeddings,
                    position_ids=position_ids,
                    cache_params=cache_params,
                    mini_batch_size=mini_batch_size,
                    **kwargs
                )

            if output_hidden_states:
                if pad_length > 0:
                    all_hidden_states = all_hidden_states + (hidden_states[:, : -pad_length])
                else:
                    all_hidden_states = all_hidden_states + (hidden_states,)

        if pad_length > 0:
            hidden_states = hidden_states[:, : -pad_length]
        hidden_states = self.norm(hidden_states)

        if use_cache:
            cache_params.token_len += hidden_states.shape[1]

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return TTTLinearOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            cache_params=cache_params if use_cache else None
        )


class TTTLinearForCausalLM(TTTLinearPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: TTTLinearConfig):
        super().__init__(config)
        self.model = TTTLinearModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs['cache_params'] = outputs.get('cache_params', None)
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TTTLinearCache] = None,
        mini_batch_size: Optional[int] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        if use_cache is None:
            use_cache = True  # TTTLinear always use cache on generation mode if the user explicitly set it to False

        if use_cache and cache_params is not None:  # Generation with cache
            past_length = cache_params.token_len if hasattr(cache_params, 'token_len') else 0
            input_len = input_ids.shape[-1]
            new_token_len = input_len - past_length
            if new_token_len > self.config.chunk_size:  # falling back to normal mode
                pass
            else:  # auto-regressive generation mode
                if mini_batch_size is None:
                    mini_batch_size = new_token_len
                if input_ids is not None:
                    input_ids = input_ids[:, -new_token_len].unsqueeze(-1)
                elif inputs_embeds is not None:
                    inputs_embeds = inputs_embeds[:, -new_token_len].unsqueeze(-1)
                else:
                    raise ValueError("input_ids or inputs_embeds must be specified")
                if position_ids is not None:
                    position_ids = position_ids[:, -new_token_len].unsqueeze(-1)
        else:  # Prefill or non-caching Generation
            cache_params = None
            mini_batch_size = None

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}

        model_inputs.update({
            'position_ids': position_ids,
            'cache_params': cache_params,
            'mini_batch_size': mini_batch_size,
            'use_cache': use_cache
        })
        return model_inputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TTTLinearCache] = None,
        mini_batch_size: Optional[int] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, TTTLinearCausalLMOutput]:
        """
        Forward pass through the TTT Linear model for causal language modeling.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence token in the position embeddings.
            cache_params (`TTTLinearCache`, *optional*):
                Cache object containing fast weights and gradient states for efficient incremental generation.
                Enables the model to maintain its learned adaptations across generation steps.
            mini_batch_size (`int`, *optional*):
                Mini-batch size for TTT learning updates. If None, defaults to config.chunk_size.
                Smaller values enable more frequent adaptation but may be less stable.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, cache_params is returned and can be used to speed up decoding.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a ModelOutput instead of a plain tuple.
            **kwargs:
                Additional keyword arguments passed to the underlying model.

        Returns:
            `Union[Tuple, TTTLinearCausalLMOutput]`:
                Either a tuple or TTTLinearCausalLMOutput containing:
                - loss: Language modeling loss (if labels provided)
                - logits: Prediction scores for each vocabulary token
                - cache_params: Updated cache with fast weights and states
                - hidden_states: Hidden states of all layers (if requested)
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: TTTLinearOutput = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_params=cache_params,
            mini_batch_size=mini_batch_size,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return TTTLinearCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states,
        )


class TTTLinearForSequenceClassification(GenericForSequenceClassification, TTTLinearPreTrainedModel):
    pass


class TTTLinearForTokenClassification(GenericForTokenClassification, TTTLinearPreTrainedModel):
    pass


class TTTLinearForImageClassification(ViTForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        self.vit = TTTLinearModel(config)  # TODO: Add patch config and patch embedding

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        return super().forward(
            pixel_values, head_mask, labels,
            output_attentions, output_hidden_states,
            interpolate_pos_encoding, return_dict
        )


__all__ = [
    "TTTLinearConfig",
    "TTTLinearCache",
    "TTTLinearAdaptation",
    "TTTLinearLayer",
    "TTTLinearPreTrainedModel",
    "TTTLinearModel",
    "TTTLinearForCausalLM",
    "TTTLinearForSequenceClassification",
    "TTTLinearForTokenClassification",
    "TTTLinearForImageClassification"
]
