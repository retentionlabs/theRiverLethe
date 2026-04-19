from transformers.utils.import_utils import *
from transformers.utils.import_utils import _is_package_available


@lru_cache
def is_flash_linear_attention_available() -> bool:
    is_available = _is_package_available("fla")[0]
    return is_available and is_torch_cuda_available()
