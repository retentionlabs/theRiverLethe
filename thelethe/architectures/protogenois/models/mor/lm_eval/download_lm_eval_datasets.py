import os
from paths import SAVE_DIR, PROJECT_ROOT, HF_CACHE_DIR
os.environ["HF_HOME"] = HF_CACHE_DIR

import lm_eval
from lm_eval.tasks import TaskManager


# List of tasks to download
TASKS_TO_DOWNLOAD = [
    'arc_challenge',
    'arc_easy',
    'hellaswag',
    'lambada_openai',
    'mmlu',
    'mmlu_continuation',
    'openbookqa',
    'piqa',
    'winogrande',
]


def download_datasets():
    task_manager = TaskManager()
    tasks = task_manager.load_task_or_group(TASKS_TO_DOWNLOAD)


if __name__ == "__main__":
    print("Starting dataset download process...")
    download_datasets()
    print("Dataset download process completed!")