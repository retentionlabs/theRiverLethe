import random

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("recursive_lm", "recursive_transformer")
class RecursiveTransformer(HFLM):
    def __init__(
        self,
        pretrained,
        tokenizer=None,
        **kwargs,
    ) -> None:
                
        assert not isinstance(pretrained, str), "pretrained must be RecursiveTransformer object, not str"
        
        super().__init__(
            pretrained=pretrained,
            tokenizer=tokenizer,
            # set appropriate defaults for tokenizer, max length, etc
            **kwargs,
        )
