from ...configs import PretrainedTitansConfig
from .....utils import logging


logger = logging.get_logger(__name__)


class CronosConfig(PretrainedTitansConfig):
    """
    This is the configuration class to store the configuration of a `Cronos` model.
    It is used to initialize the `Cronos` model with the specified configuration parameters.
    """

    model_type = "cronos"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

