import string
import random


characters = string.ascii_letters + string.digits
wandb_run_id = "".join(random.choices(characters, k=8))
print(f"wandb_run_id: {wandb_run_id}")