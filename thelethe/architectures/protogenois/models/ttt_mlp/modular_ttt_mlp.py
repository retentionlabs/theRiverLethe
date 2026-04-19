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
from torch import nn

from .....transformers.modeling_layers import (
    GenericForSequenceClassification,
    GenericForTokenClassification,
)
from ..ttt_linear.configuration_ttt_linear import TTTLinearConfig
from ..ttt_linear.modeling_ttt_linear import (
    TTTLinearCache,
    TTTRMSNorm,
    TTTSwiGluMLP,
    TTTRotaryEmbedding,
    TTTMultiheadLayerNorm,
    TTTDynamicLearningGate,
    TTTMultiheadLinearMixin,
    TTTMultiheadLinear,
    TTTLinearAdaptationState,
    TTTLinearAdaptation,
    TTTLinearLayer,
    TTTLinearPreTrainedModel,
    TTTLinearOutput,
    TTTLinearCausalLMOutput,
    TTTLinearModel,
    TTTLinearForCausalLM,
    TTTLinearForImageClassification
)

from .....utils import logging


logger = logging.get_logger(__name__)


class TTTMLPConfig(TTTLinearConfig):
    model_type = "ttt_mlp"

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
        use_cache=True,
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
        scan_checkpoint_group_size=0,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            mini_batch_eps=mini_batch_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            mlp_bias=mlp_bias,
            adapt_base_lr=adapt_base_lr,
            chunk_size=chunk_size,
            scan_checkpoint_group_size=scan_checkpoint_group_size,
            **kwargs,
        )

        self.memory_depth = 2  # TTTMLPAdaptation depth


class TTTMLPCache(TTTLinearCache):
    pass


class TTTMLPAdaptation(TTTLinearAdaptation):

    @staticmethod
    def struct_details(num_heads: int, head_dim: int):
        return [
            dict(num_heads=num_heads, in_features=head_dim, out_features=4*head_dim),
            dict(num_heads=num_heads, in_features=4*head_dim, out_features=head_dim)
        ]


class TTTMLPLayer(TTTLinearLayer):
    def __init__(self, config: TTTMLPConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size

        self.self_adapt = TTTMLPAdaptation(config=config, layer_idx=layer_idx)
        self.mlp = TTTSwiGluMLP(config)

        self.seq_norm = TTTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = TTTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx


class TTTMLPPreTrainedModel(TTTLinearPreTrainedModel):
    config_class = TTTMLPConfig
    base_model_prefix = "ttt_mlp"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TTTMLPLayer"]


class TTTMLPOutput(TTTLinearOutput):
    pass


class TTTMLPCausalLMOutput(TTTLinearCausalLMOutput):
    pass


class TTTMLPModel(TTTLinearModel):
    def __init__(self, config: TTTMLPConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([TTTMLPLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = TTTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TTTRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class TTTMLPForCausalLM(TTTLinearForCausalLM):
    def __init__(self, config: TTTMLPConfig):
        super().__init__(config)
        self.model = TTTMLPModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class TTTMLPForSequenceClassification(GenericForSequenceClassification, TTTMLPPreTrainedModel):
    pass


class TTTMLPForTokenClassification(GenericForTokenClassification, TTTMLPPreTrainedModel):
    pass


class TTTMLPForImageClassification(TTTLinearForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        self.vit = TTTMLPModel(config)


__all__ = [
    "TTTMLPConfig",
    "TTTMLPCache",
    "TTTMLPAdaptation",
    "TTTMLPLayer",
    "TTTMLPPreTrainedModel",
    "TTTMLPModel",
    "TTTMLPForCausalLM",
    "TTTMLPForSequenceClassification",
    "TTTMLPForTokenClassification",
    "TTTMLPForImageClassification"
]
