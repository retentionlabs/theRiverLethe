from .models.origin import *
from .models.atlas import *

from .modelings import PreTrainedTitansModel, AutoTitansModelForCausalLM
from .configs import PretrainedTitansConfig, AutoTitansConfig


AutoTitansConfig.register(OriginConfig.model_type, OriginConfig)
AutoTitansModelForCausalLM.register(OriginConfig, OriginForCausalLM)

AutoTitansConfig.register(AtlasConfig.model_type, AtlasConfig)
AutoTitansModelForCausalLM.register(AtlasConfig, AtlasForCausalLM)
