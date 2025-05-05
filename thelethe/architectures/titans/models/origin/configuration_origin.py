from ....configuration import PretrainedTitansConfig


logger = logging.get_logger(__name__)


class OriginConfig(PretrainedTitansConfig):
    """
    This is the configuration class to store the configuration of a `Origin` model.
    It is used to initialize the `Origin` model with the specified configuration parameters.
    """

    model_type = "origin"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

