from transformers.modeling_utils import PreTrainedModel


class OriginPreTrainedModel(PreTrainedModel):
    pass


class OriginModel(OriginPreTrainedModel):
    """
    This is the base class for all Origin models.
    It inherits from `PreTrainedModel` and provides the basic functionality for loading and saving model weights.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.init_weights()  # Initialize weights using the method from PreTrainedModel


class OriginForCausalLM(OriginPreTrainedModel):
    """
    This is the base class for all Origin models used for causal language modeling.
    It inherits from `OriginPreTrainedModel` and provides the basic functionality for loading and saving model weights.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.init_weights()
