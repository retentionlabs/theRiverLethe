import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# save directory for checkpoints, logs, and other results
# make sure you set symbolic link: ln -s {disk_path}/{fpath} ./{fpath}
SAVE_DIR = os.path.join(PROJECT_ROOT, "results")

# cache directory for huggingface datasets, models, tokenizers
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_cache")

# directory for source data
DATA_DIR = os.path.join(PROJECT_ROOT, "hf_datasets")

# directory for model checkpoints from huggingface (for distillation and other purposes)
MODEL_DIR = os.path.join(PROJECT_ROOT, "hf_models")