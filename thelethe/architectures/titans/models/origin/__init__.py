from .modeling_origin import OriginPreTrainedTitansModel, OriginModel, OriginForCausalLM
from .configuration_origin import OriginConfig

from .modeling_utils import PreTrainedTitansModel, AutoTitansModelForCausalLM
from .configuration_utils import PretrainedTitansConfig, AutoTitansConfig


AutoTitansConfig.register(OriginConfig.model_type, OriginConfig)
AutoTitansModelForCausalLM.register(OriginConfig, OriginForCausalLM)
