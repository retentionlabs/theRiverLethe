from ...modeling_utils import PreTrainedTitansModel
from .configuration_cronos import CronosConfig


class CronosPreTrainedTitansModel(PreTrainedTitansModel):
    config_class = CronosConfig


class CronosModel(CronosPreTrainedTitansModel):
    """
    This is the base class for all Cronos models.
    It inherits from `CronosPreTrainedTitansModel` and provides the basic functionality for loading and saving model weights.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.init_weights()  # Initialize weights using the method from PreTrainedTitansModel


class CronosForCausalLM(CronosPreTrainedTitansModel):
    """
    This is the base class for all Cronos models used for causal language modeling.
    It inherits from `CronosPreTrainedTitansModel` and provides the basic functionality for loading and saving model weights.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.init_weights()
