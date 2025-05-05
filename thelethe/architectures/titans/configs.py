from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from ...utils import logging


class PretrainedTitansConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a `PretrainedTitansModel`.
    It is used to instantiate a PretrainedTitans model according to the specified arguments, defining the model architecture.
    The configuration class is used to create a model instance from scratch using the `from_pretrained` method.
    """
    pass


class AutoTitansConfig(AutoConfig):
    pass
