#!/bin/bash

# Run all Python scripts in parallel
python3 - <<EOF
import os
from paths import SAVE_DIR, HF_CACHE_DIR, DATA_DIR; os.environ["HF_HOME"] = HF_CACHE_DIR
from datasets import load_dataset

dataset_splits = load_dataset("HuggingFaceTB/smollm-corpus", 'fineweb-edu-dedup')
print("fineweb-edu-dedup completed successfully")
EOF

echo "Downloading fineweb-edu-dedup completed!"