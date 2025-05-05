from ...modelings import PreTrainedTitansModel
from .configuration_origin import OriginConfig


class OriginPreTrainedTitansModel(PreTrainedTitansModel):
    config_class = OriginConfig


class OriginModel(OriginPreTrainedTitansModel):
    """
    This is the base class for all Origin models.
    It inherits from `OriginPreTrainedTitansModel` and provides the basic functionality for loading and saving model weights.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.init_weights()  # Initialize weights using the method from PreTrainedTitansModel


class OriginForCausalLM(OriginPreTrainedTitansModel):
    """
    This is the base class for all Origin models used for causal language modeling.
    It inherits from `OriginPreTrainedTitansModel` and provides the basic functionality for loading and saving model weights.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.init_weights()
