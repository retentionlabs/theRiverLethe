from ...modelings import PreTrainedTitansModel
from .configuration_atlas import AtlasConfig


class AtlasPreTrainedTitansModel(PreTrainedTitansModel):
    config_class = AtlasConfig


class AtlasModel(AtlasPreTrainedTitansModel):
    """
    This is the base class for all Atlas models.
    It inherits from `AtlasPreTrainedTitansModel` and provides the basic functionality for loading and saving model weights.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.init_weights()  # Initialize weights using the method from PreTrainedTitansModel


class AtlasForCausalLM(AtlasPreTrainedTitansModel):
    """
    This is the base class for all Atlas models used for causal language modeling.
    It inherits from `AtlasPreTrainedTitansModel` and provides the basic functionality for loading and saving model weights.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.init_weights()
