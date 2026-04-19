from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM

from transformers.modeling_utils import *


class PreTrainedTitansModel(PreTrainedModel):
    pass


class AutoTitansModel(AutoModel):
    # TODO: Add per-autoclass specific model registry
    pass


class AutoMacModel(AutoTitansModel):
    pass


class AutoMaeModel(AutoTitansModel):
    pass


class AutoMagModel(AutoTitansModel):
    pass


class AutoMalModel(AutoTitansModel):
    pass


class AutoTitansModelForCausalLM(AutoModelForCausalLM):
    pass


class AutoMacModelForCausalLM(AutoTitansModelForCausalLM):
    pass



class AutoMaeModelForCausalLM(AutoTitansModelForCausalLM):
    pass


class AutoMagModelForCausalLM(AutoTitansModelForCausalLM):
    pass


class AutoMalModelForCausalLM(AutoTitansModelForCausalLM):
    pass


class AutoTitansModelForConditionalGeneration(AutoModel):
    pass


class AutoMacModelForConditionalGeneration(AutoTitansModelForConditionalGeneration):
    pass


class AutoMaeModelForConditionalGeneration(AutoTitansModelForConditionalGeneration):
    pass


class AutoMagModelForConditionalGeneration(AutoTitansModelForConditionalGeneration):
    pass


class AutoMalModelForConditionalGeneration(AutoTitansModelForConditionalGeneration):
    pass
