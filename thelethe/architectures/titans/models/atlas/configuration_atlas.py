from ....configuration import PretrainedTitansConfig


logger = logging.get_logger(__name__)


class AtlasConfig(PretrainedTitansConfig):
    """
    This is the configuration class to store the configuration of a `Atlas` model.
    It is used to initialize the `Atlas` model with the specified configuration parameters.
    """

    model_type = "atlas"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

