import os

from transformers import AutoTokenizer

if "wandb_mode" not in os.environ:
    local_files_only = True
else:
    local_files_only = os.environ["WANDB_MODE"] == "offline"

TOKENIZERS = {
    "smollm": AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=local_files_only),
    "smollm2": AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", local_files_only=local_files_only),
}


def load_tokenizer_from_config(cfg):
    tokenizer = TOKENIZERS[cfg.tokenizer]
    if tokenizer.pad_token is None:
        if cfg.tokenizer in ["smollm", "smollm2"]:
            # '<|endoftext|>'
            tokenizer.pad_token_id = 0
        else:
            raise ValueError(f"Tokenizer {cfg.tokenizer} does not have a pad token, please specify one in the config")
    return tokenizer