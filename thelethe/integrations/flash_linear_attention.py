from functools import lru_cache
import torch

from ..utils import (
    is_flash_linear_attention_available,
    logging,
)
from ..utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING, is_tracing


logger = logging.get_logger(__name__)


@lru_cache
def is_fla_available() -> bool:
    return is_flash_linear_attention_available()


SUPPORTED_KERNELS = [
    # Momentary Surprise
    "linear_d1_momentary",  # memory depth 1

    # Momentary Surprise + Momentum Surprise
    "linear_d2_momentary_momentum"  # memory depth 1
]


@lru_cache
def _lazy_imports(implementation: str | None) -> tuple[callable, callable, callable]:
    match implementation:
        case "linear_d1_momentary":  # ttt_linear
            from fla.ops.ttt.fused_chunk import (
                fused_chunk_ttt_linear_fwd, fused_chunk_ttt_linear_bwd, fused_chunk_ttt_linear
            )
            forward, backward, autograd = fused_chunk_ttt_linear_fwd, fused_chunk_ttt_linear_bwd, fused_chunk_ttt_linear
        case "linear_d2_momentary_momentum":  # titans_origin
            from fla.ops.titans.naive import chunk_titans_linear
            forward, backward, autograd = chunk_titans_linear, None, None  # TODO: Add Titans backward
        case _:
            raise NotImplementedError(f"Requested kernel implementation `{implementation}` is not supported yet.")

    return forward, backward, autograd


@lru_cache
def is_fla_supported_kernel(
    memory_shape: str, memory_depth: int, loss_type: list[str] | tuple[str]
) -> tuple[bool, str]:
    key = f"{memory_shape}_d{memory_depth}_" + "_".join(loss_type)
    return key in SUPPORTED_KERNELS, key


def flash_linear_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
):
    config = module.config
    implementation = is_fla_supported_kernel(
        config.memory_shape, config.memory_depth, config.memory_loss_type
    )[1]
    forward, backward, autograd = _lazy_imports(implementation)
    if backward is None:  # Autograd mode
        pass
    else:
        pass