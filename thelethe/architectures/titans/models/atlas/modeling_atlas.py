from transformers.modeling_utils import PreTrainedModel


class AtlasPreTrainedModel(PreTrainedModel):
    pass


class AtlasModel(AtlasPreTrainedModel):
    """
    This is the base class for all Atlas models.
    It inherits from `PreTrainedModel` and provides the basic functionality for loading and saving model weights.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.init_weights()  # Initialize weights using the method from PreTrainedModel


class AtlasForCausalLM(AtlasPreTrainedModel):
    """
    This is the base class for all Atlas models used for causal language modeling.
    It inherits from `AtlasPreTrainedModel` and provides the basic functionality for loading and saving model weights.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.init_weights()
