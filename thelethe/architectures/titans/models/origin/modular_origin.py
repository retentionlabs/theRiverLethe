import torch
from torch import nn
import torch.nn.functional as F

from transformers.modeling_layers import (
    GenericForSequenceClassification,
    GenericForTokenClassification,
)
from transformers.models.ttt_linear.configuration_ttt_linear import TTTLinearConfig
from transformers.models.ttt_linear.modeling_ttt_linear import (
    TTTRMSNorm,
    TTTSwiGluMLP,
    TTTRotaryEmbedding,
    TTTCausalConv1d,
    TTTMultiHeadLayerNorm,
    TTTDynamicLearningGate,
    TTTAdaptiveLinear,
    TTTLinearMemory,
    TTTLinearCache,
    TTTLinearAdaptation,
    TTTLinearLayer,
    TTTLinearPreTrainedModel,
    TTTLinearOutput,
    TTTLinearCausalLMOutput,
    TTTLinearModel,
    TTTLinearForCausalLM,
    TTTLinearForImageClassification
)
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.import_utils import is_causal_conv1d_available
if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

from ...configs import PretrainedTitansConfig
from ...modelings import PreTrainedTitansModel
from .....utils.scan_ops import associative_scan
from .....utils import logging


logger = logging.get_logger(__name__)


class OriginConfig(PretrainedTitansConfig, TTTLinearConfig):
    """
    This is the configuration class to store the configuration of a `Origin` model.
    It is used to initialize the `Origin` model with the specified configuration parameters.
    """

    model_type = "origin"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # TODO: Memory architecture description
        self.variant = kwargs.pop("variant", "lmm")
        # TODO: Memory depth
        self.memory_depth = kwargs.pop("memory_depth", 4)


class TitansRMSNorm(TTTRMSNorm):
    pass


class TitansSwiGluMLP(TTTSwiGluMLP):
    pass


class TitansRotaryEmbedding(TTTRotaryEmbedding):
    pass


class TitansCausalConv1d(TTTCausalConv1d):
    pass


class TitansMultiHeadLayerNorm(TTTMultiHeadLayerNorm):
    pass


class TitansMomentumBasedSurpriseGate(TTTDynamicLearningGate):
    def __init__(self, num_heads: int, head_dim: int, chunk_size: int, adapt_base_lr: float, momentum: float, weight_decay: float):
        super().__init__(num_heads, head_dim, chunk_size, adapt_base_lr)
        del self.token_idx

        self.momentum = momentum
        self.weight_decay = weight_decay

        # [head_dim, 1] -> [num_heads, head_dim, 1]
        target_shape_per_head = (self.head_dim, 1)
        linear_bias_data = nn.Linear(self.head_dim, 1, bias=True).bias.data
        # alpha
        self.alpha = nn.Parameter(torch.stack(
            [torch.normal(0, 0.02, size=target_shape_per_head) for _ in range(self.num_heads)],
            dim=0
        ))
        self.alpha_bias = nn.Parameter(torch.stack(  # init bias to 0 following original JAX impl.
            [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
            dim=0
        ).unsqueeze(-1))
        # theta
        self.theta = nn.Parameter(torch.stack(
            [torch.normal(0, 0.02, size=target_shape_per_head) for _ in range(self.num_heads)],
            dim=0
        ))
        self.theta_bias = nn.Parameter(torch.stack(  # init bias to 0 following original JAX impl.
            [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
            dim=0
        ).unsqueeze(-1))
        # eta
        self.eta = nn.Parameter(torch.stack(
            [torch.normal(0, 0.02, size=target_shape_per_head) for _ in range(self.num_heads)],
            dim=0
        ))
        self.eta_bias = nn.Parameter(torch.stack(  # init bias to 0 following original JAX impl.
            [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
            dim=0
        ).unsqueeze(-1))

    def __repr__(self):
        return f"{self.__class__.__name__}(momentum={self.momentum}, decay={self.weight_decay}, lr={self.adapt_base_lr})"

    def forward(self, x):
        current_mini_batch_size = x.shape[-2]

        # Momentary Surprise (Current Input) [B, num_heads, mini_batch_size, 1]
        moment_surprise = torch.einsum("bhkc,hcd->bhkd", x, self.theta) + self.theta_bias.view(1, self.num_heads, 1, 1)
        moment_surprise = F.sigmoid(moment_surprise)
        moment_surprise_eta = self.adapt_base_lr * moment_surprise / self.head_dim

        # Past Surprise [B, num_heads, mini_batch_size, 1]
        past_surprise = torch.einsum("bhkc,hcd->bhkd", x, self.theta) + self.theta_bias.view(1, self.num_heads, 1, 1)
        past_surprise = F.sigmoid(moment_surprise)
        past_surprise_eta = self.adapt_base_lr * moment_surprise / self.head_dim

        # Forget Gate [B, num_heads, mini_batch_size, 1]
        forgetting = torch.einsum("bhkc,hcd->bhkd", x, self.theta) + self.theta_bias.view(1, self.num_heads, 1, 1)
        forgetting = F.sigmoid(moment_surprise)
        forgetting_eta = self.adapt_base_lr * moment_surprise / self.head_dim

        return moment_surprise_eta, past_surprise_eta, forgetting_eta


class OriginMemory(TTTLinearMemory):
    depth = 0  # for compatibility with TTTLinearMemory

    @property
    def struct_detail(self):
        return [TTTAdaptiveLinear(self.num_heads, self.head_dim, self.head_dim) for _ in range(self.depth)]

    def __init__(
            self, parent_module: "OriginAdaptation", lr_gate: TitansMomentumBasedSurpriseGate, norm: TitansMultiHeadLayerNorm
    ):
        self.config = parent_module.config
        self.depth = self.config.memory_depth
        super().__init__(parent_module, lr_gate, norm)


class TitansCache(TTTLinearCache):
    pass


class OriginAdaptation(TTTLinearAdaptation):
    memory_class = OriginMemory

    def __init__(self, config: TTTLinearConfig, layer_idx: int, momentum, weight_decay, adapt_base_lr):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.width = config.hidden_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        self.chunk_size = config.chunk_size
        self.conv_kernel = config.conv_kernel

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.adapt_base_lr = adapt_base_lr

        self.q_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)

        self.lr_gate = TitansMomentumBasedSurpriseGate(
            self.num_heads, self.head_dim,
            self.chunk_size, self.adapt_base_lr,
            self.momentum, self.weight_decay
        )
        self.shared_norm = TitansMultiHeadLayerNorm(self.num_heads, self.head_dim, self.config.mini_batch_eps)
        self.neural_memory = self.memory_class(self, self.lr_gate, self.shared_norm)

        self.post_norm = nn.LayerNorm(self.width, eps=1e-6)

    @staticmethod
    def step(carry, xs):  # TODO: write titans method below
        # Projected Inputs
        # [B,nh,K,f], K=mini_batch_size
        XQ_mini_batch, XK_mini_batch, XV_mini_batch = xs.unbind(dim=2)

        # Reconstruction Task
        # [B,nh,K,f] @ [B,nh,f,f] -> [B,nh,K,f]
        _, reconstructed = carry(XK_mini_batch, output_hidden_states=True)
        reconstructed.insert(0, XK_mini_batch)
        reconstruction_target = XV_mini_batch
        # token_eta: [1,1,K,1], lr_eta: [B,H,K,1]
        token_eta, lr_eta = carry.lr_gate(XK_mini_batch)
        eta_scalar = token_eta * lr_eta  # [B,h,K,1] * [B,h,K,1] -> [B,h,K,1] for backward
        eta_matrix = token_eta @ lr_eta.transpose(-2, -1)  # [B,h,K,1] @ [B,h,K,1] -> [B,h,K,K] for hidden state update
        gradients = carry.backward(reconstructed, reconstruction_target, eta=eta_scalar)

        # Generate Hidden States
        hidden_states = XQ_mini_batch
        for idx, (val, gradient) in enumerate(zip(reconstructed, gradients)):
            attention_mask = torch.tril(hidden_states @ val.transpose(-2, -1))  # [B,nh,K,K]
            # [B,nh,K,f] @ [B,nh,f,f] + [B,nh,K,f] - ([B,nh,K,K] * [B,nh,K,K]) @ [B,nh,K,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
            update_term = (eta_matrix * attention_mask + torch.tril(eta_matrix)) @ gradient
            hidden_states = carry[idx](hidden_states) - update_term
            if idx < carry.depth - 1:
                hidden_states = carry.activate(hidden_states)
        hidden_states = XQ_mini_batch + carry.norm(hidden_states)  # residual connection

        # Apply Gradients to fast weight
        carry.step()

        return carry, hidden_states.unsqueeze(2).expand(-1, -1, 3, -1, -1)  # match to xs shape for scan


class OriginLayer(TTTLinearLayer):
    def __init__(self, config: OriginConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.pre_conv = config.pre_conv

        self.self_adapt = OriginAdaptation(config=config, layer_idx=layer_idx)

        self.mlp = TitansSwiGluMLP(config)
        if self.pre_conv:
            self.conv = TitansCausalConv1d(config, layer_idx)

        self.seq_norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx


class OriginPreTrainedModel(PreTrainedTitansModel, TTTLinearPreTrainedModel):
    config_class = OriginConfig
    base_model_prefix = "origin"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OriginLayer"]


class TitansOutput(TTTLinearOutput):
    pass


class TitansCausalLMOutput(TTTLinearCausalLMOutput):
    pass


class OriginPreTrainedTitansModel(PreTrainedTitansModel):
    config_class = OriginConfig


class OriginModel(OriginPreTrainedTitansModel, TTTLinearModel):
    def __init__(self, config: OriginConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        momentums = [0 for _ in range(config.num_hidden_layers)]
        decays = [0 for _ in range(config.num_hidden_layers)]

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            OriginLayer(config, idx, momentum, decay)
            for idx, (momentum, decay) in enumerate(zip(momentums, decays))
        ])
        self.norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TitansRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class OriginForCausalLM(OriginPreTrainedTitansModel, TTTLinearForCausalLM):
    def __init__(self, config: OriginConfig):
        super().__init__(config)
        self.model = OriginModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class OriginForSequenceClassification(GenericForSequenceClassification, TTTLinearPreTrainedModel):
    pass


class OriginForTokenClassification(GenericForTokenClassification, TTTLinearPreTrainedModel):
    pass


class OriginForImageClassification(TTTLinearForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        self.vit = OriginModel(config)


__all__ = [
    "OriginConfig",
    "OriginAdaptation",
    "OriginLayer",
    "OriginPreTrainedModel",
    "OriginModel",
    "OriginForCausalLM",
    "OriginForSequenceClassification",
    "OriginForTokenClassification",
    "OriginForImageClassification"
]
