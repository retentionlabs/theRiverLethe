from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.models.auto.configuration_auto import AutoConfig
from ...utils import logging

from typing import Literal
from enum import Enum


class TitansVariants(Enum):
    LMM = "lmm"  # Linear Recurrent Model (Long-term Memory Module As Sequence Model)
    MAC = "mac"  # Memory As Context
    MAE = "mae"  # Memory As Embedding
    MAG = "mag"  # Memory As Gate
    MAL = "mal"  # Memory As Layer


class PretrainedTitansConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a `PretrainedTitansModel`.
    It is used to instantiate a PretrainedTitans model according to the specified arguments, defining the model architecture.
    The configuration class is used to create a model instance from scratch using the `from_pretrained` method.
    """
    model_variants = TitansVariants

    def __init__(
        self,
        variant: TitansVariants | Literal["lmm", "mac", "mae", "mag", "mal"] = TitansVariants.LMM,
        memory_depth: int = 4,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        **kwargs
    ):
        if isinstance(variant, str):
            self.variant = self.model_variants(variant)
        else:
            self.variant = variant

        self.memory_depth = memory_depth

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class AutoTitansConfig(AutoConfig):
    pass
