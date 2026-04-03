from collections import namedtuple
from dataclasses import dataclass, field

from peft.config import PeftConfig
from peft.utils import PeftType

from model.relaxation.prompt.util import llama_compute_query_states


@dataclass
class AdaptionPromptConfig(PeftConfig):
    """Stores the configuration of an [`AdaptionPromptModel`]."""

    target_modules: str = field(
        default=None, metadata={"help": "Name of the attention submodules to insert adaption prompts into."}
    )
    adapter_len: int = field(default=None, metadata={"help": "Number of adapter tokens to insert"})
    adapter_layers: int = field(default=None, metadata={"help": "Number of adapter layers (from the top)"})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.ADAPTION_PROMPT

    @property
    def is_adaption_prompt(self) -> bool:
        """Return True if this is an adaption prompt config."""
        return True


ModelTypeConfig = namedtuple(
    "ModelTypeConfig", ["compute_query_states", "target_modules", "k_proj_layer", "v_proj_layer", "o_proj_layer"]
)


TRANSFORMERS_MODEL_CONFIG = {
    "llama": ModelTypeConfig(
        compute_query_states=llama_compute_query_states,
        target_modules="self_attn",
        k_proj_layer="k_proj",
        v_proj_layer="v_proj",
        o_proj_layer="o_proj",
    ),
    "mistral": ModelTypeConfig(  # same as llama,
        compute_query_states=llama_compute_query_states,
        target_modules="self_attn",
        k_proj_layer="k_proj",
        v_proj_layer="v_proj",
        o_proj_layer="o_proj",
    ),
}


def prepare_config(
    peft_config: AdaptionPromptConfig,
    model,
) -> AdaptionPromptConfig:
    """Prepare the config based on the llama model type."""
    if model.config.model_type not in TRANSFORMERS_MODEL_CONFIG:
        raise ValueError("Unsupported model type for adaption prompt: '{model.config.model_type}'.")

    model_config = TRANSFORMERS_MODEL_CONFIG[model.config.model_type]

    if peft_config.target_modules is None:
        peft_config.target_modules = model_config.target_modules

    return peft_config
