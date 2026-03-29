import os
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
from paths import SAVE_DIR, PROJECT_ROOT, HF_CACHE_DIR; os.environ["HF_HOME"] = HF_CACHE_DIR

import re
import string
import random
import warnings
import argparse
from omegaconf import DictConfig, OmegaConf, open_dict

DEFAULT_MODEL_NAME_OR_PATH = {
    "smollm-135m": "HuggingFaceTB/SmolLM-135M",
    "smollm-360m": "HuggingFaceTB/SmolLM-360M",
    "smollm-800m": "HuggingFaceTB/SmolLM-1.7B",
    "smollm-1.7b": "HuggingFaceTB/SmolLM-1.7B",
    "smollm": "HuggingFaceTB/SmolLM-360M",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "gemma-2b": "google/gemma-2b",
    "gemma3-1b": "google/gemma-3-1b-pt",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia": "EleutherAI/pythia-160m",
    "olmo-1b": "allenai/OLMo-1B-hf",
}


def generate_configs(args: argparse.Namespace):
    """
    Loads an OmegaConf file from example config files of pretrain and eval_fewshot,
    overrides it with command-line arguments (args),
    and saves it to save_dir.
    """
    
    # Default path to the base config file (constant)
    BASE_PRETRAIN_CONFIG_PATH = os.path.join(PROJECT_ROOT, f"conf/pretrain/{args.base_cfg}.yaml")
    BASE_EVAL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "conf/eval_fewshot/example.yaml")

    # ======== Pretrain config file ========
    conf = OmegaConf.load(BASE_PRETRAIN_CONFIG_PATH)

    # update name
    conf.name = args.name
    split_name = args.name.split("_")

    # update wandb_run_id
    characters = string.ascii_letters + string.digits
    wandb_run_id = "".join(random.choices(characters, k=8))
    conf.wandb_run_id = wandb_run_id
    conf.wandb_mode = args.wandb_mode

    # update batch_size
    if args.batch_size is not None:
        conf.per_device_train_batch_size = args.batch_size
        
    # update model and tokenizer
    candidates = list(DEFAULT_MODEL_NAME_OR_PATH.keys())
    found = False
    for cand in candidates:
        if cand in split_name:
            model_name = cand.split("-")[0]
            conf.model = model_name
            conf.tokenizer = model_name
            if args.model_name_or_path is None:
                args.model_name_or_path = DEFAULT_MODEL_NAME_OR_PATH[cand]
            found = True; break
    if not found:
        raise ValueError(f"Model name must contain one of {candidates}")
    
    # update add_bos_token
    if conf.model in ["tinyllama", "gemma", "gemma3"]:
        conf.add_bos_token = True
    
    # update model_name_or_path
    assert args.model_name_or_path is not None, "Model name or path must be provided."
    conf.model_name_or_path = args.model_name_or_path    
    
    # update use_pretrained_weights
    if "pretrain" in split_name:
        conf.use_pretrained_weights = False
    elif "uptrain" in split_name:
        conf.use_pretrained_weights = True
    else:
        raise ValueError("Name must contain either 'pretrain' or 'uptrain'")

    # update dataset
    if args.dataset is not None:
        conf.dataset = args.dataset

    # update dataloader_num_workers
    if args.dataloader_num_workers is not None:
        conf.dataloader_num_workers = args.dataloader_num_workers

    # update learning rate
    found = False
    for strs in split_name:
        if "lr" in strs:
            match = re.search(r"lr(\d+\.?\d*e[-+]?\d+)", strs)
            if match:
                conf.learning_rate = float(match.group(1))
            else:
                raise ValueError("learning rate must be included in the name.")
            found = True; break
    if not found:
        raise ValueError("Learning rate must be included in the name.")

    # update save_interval
    if args.save_interval is not None:
        conf.save_interval = args.save_interval

    # update training steps
    for strs in split_name:
        if '-' in strs: continue
        match = re.search(r"(\d+)b", strs)
        if match:
            steps = int(int(match.group(1)) * 1_000_000_000 / \
                (conf.total_batch_size * conf.max_length))
            conf.num_train_steps = conf.stop_steps = steps
            found = True; break
    if not found:
        raise ValueError("Number of total token numbers must be included in the name.")
    
    # update recursive enable and number
    found = False
    for strs in split_name:
        if "rec" in strs:
            match = re.search(r"rec(\d+)", strs)
            if match:
                conf.recursive.num_recursion = int(match.group(1))
            else:
                raise ValueError("rec{digit} must be included in the name.")
            found = True; break
    if not found:
        conf.recursive.enable = False
    
    if conf.recursive.enable:
        # update base_depth when num_recursion = 1
        if conf.recursive.num_recursion == 1:
            found = False
            for strs in split_name:
                if "depth" in strs:
                    match = re.search(r"depth(\d+)", strs)
                    if match:
                        conf.recursive.base_depth = int(match.group(1))
                    else:
                        raise ValueError("depth{digit} must be included in the name.")
                    found = True; break
            if not found:
                raise ValueError("Base depth must be included in the name.")            
        
        # update recursive sharing
        candidates = ["middle_sequence", "middle_cycle", "cycle", "sequence"]
        found = False
        if conf.recursive.num_recursion == 1: # overwrite "cycle" for no recursion
            found = True
            conf.recursive.sharing = "cycle"
        for cand in candidates:
            if cand in conf.name:
                conf.recursive.sharing = cand
                found = True; break
        if not found:
            raise ValueError(f"Model name must contain one of {candidates}")
        
        # update recursive initialization
        candidates = ["stepwise", "average", "random", "lower", "upper"]
        found = False
        for cand in candidates:
            if cand in split_name:
                conf.recursive.initialization = cand
                found = True; break
        if not found:
            raise ValueError(f"Model name must contain one of {candidates}")
            
        # update ln_share
        if not args.ln_share:
            conf.recursive.ln_share = False
        
        # update relaxation
        candidates = ["lora", "dora", "prompt", "prefix", "recenc"]
        found = False
        for cand in candidates:
            if cand in split_name:
                conf.relaxation.enable = True
                if cand in ["lora", "dora"]:
                    conf.relaxation.method = cand
                elif cand == "prompt":
                    conf.relaxation.method = "adaption_prompt"
                elif cand == "recenc":
                    conf.relaxation.method = "recursion_encoding"
                found = True; break
        if not found:
            conf.relaxation.enable = False
        
        if conf.relaxation.enable:
            # update skip_first_loop
            if not args.skip_first_loop:
                conf.relaxation.skip_first_loop = False
            
            # update lora and dora config
            if conf.relaxation.method in ["lora", "dora"]:
                found = False
                for strs in split_name:
                    match = re.search(r"(?<!\w)r(\d+)", strs) # Make sure "r" is not preceded by a anything as we have "lr" in the name
                    if match:
                        conf.relaxation.lora.r = int(match.group(1))
                        found = True; break
                if not found:
                    conf.relaxation.enable = False
                    warnings.warn("r{digit} must be included in the name for LORA or DORA.")
                    
            for strs in split_name:
                if "svd" in strs:
                    conf.relaxation.lora.svd_init = True
                    break
                    
            if args.alpha_pattern is not None:
                conf.relaxation.lora.alpha_pattern = args.alpha_pattern
                conf.relaxation.lora.alpha = conf.relaxation.lora.r * args.alpha_pattern

            warnings.warn("Please check the target modules for correctness!!!")

            # update adaption_prompt config
            if conf.relaxation.method == "adaption_prompt":
                found = False
                for strs in split_name:
                    match = re.search(r"l(\d+)", strs)
                    if match:
                        conf.relaxation.prompt.len = int(match.group(1))
                        found = True; break
                if not found:
                    conf.relaxation.enable = False
                    warnings.warn("l{digit} must be included in the name for Adaption Prompt or Prefix Tuning.")
            
        # update mor model
        found = False
        for strs in split_name:
            if "mor" in strs:
                conf.mor.enable = True
                found = True; break
        if not found:
            conf.mor.enable = False
        
        if conf.mor.enable:
            # update mor router types
            candidates = ["expert", "token"]
            found = False
            for cand in candidates:
                if cand in split_name:
                    conf.mor.type = cand
                    found = True; break
            if not found:
                raise ValueError(f"MoR router type must contain one of {candidates}")

            warnings.warn("Please check the MoR config for correctness!!!")
    
    # update kv cache sharing
    found = False
    for strs in split_name:
        if "kv-share" in strs:
            conf.kv_sharing.enable = True
            found = True; break
    if not found:
        conf.kv_sharing.enable = False

    if conf.kv_sharing.enable:
        warnings.warn("Please check the kv_sharing config for correctness!!!")

    # update expert alpha
    if conf.mor.type == "expert" and "alpha" in conf.name:
        match = re.search(r"alpha_(\d+\.\d+)", conf.name)
        if match:
            conf.mor.expert.alpha = float(match.group(1))
        else:
            conf.mor.expert.alpha = 1.0
            warnings.warn("'alpha' is included in the name, but no alpha is provided. Alpha set to '1.0' by default.")
    
    # update router type
    candidates = ["linear", "mlp", "wmlp"]
    found = False
    for cand in candidates:
        if cand in split_name:
            conf.mor.router_type = {"linear": "linear", "mlp": "mlp", "wmlp": "wide_mlp"}[cand]
            found = True; break
    if not found:
        conf.mor.router_type = "mlp"
        warnings.warn("Router type set to 'mlp' by default.")

    # update sampling coefficient for aux_loss
    if "aux_loss" in conf.name:
        match = re.search(r"aux_loss_(\d+\.\d+)", conf.name)
        if match:
            conf.mor.expert.coeff = float(match.group(1))
        else:
            conf.mor.expert.coeff = 0.01
            warnings.warn("'aux_loss' is included in the name, but no sampling coefficient is provided. Sampling coefficient set to '0.01' by default.")
    
    # update zloss
    if "zloss" in conf.name:
        conf.mor.z_loss = True
        match = re.search(r"zloss_(\d+\.\d+)", conf.name)
        if match:
            conf.mor.z_coeff = float(match.group(1))
        else:
            conf.mor.z_coeff = 0.001
            warnings.warn("'zloss' is included in the name, but no z-loss coefficient is provided. Z-loss coefficient set to '0.001' by default.")  

    # save the configuration
    save_dir = os.path.join(PROJECT_ROOT, "conf/pretrain", f"{args.name}.yaml")
    OmegaConf.save(config=conf, f=save_dir)
    print(f"Configuration saved to: {save_dir}")
    
    # ======== Eval_fewshot config file ========
    conf = OmegaConf.load(BASE_EVAL_CONFIG_PATH)
    
    conf.wandb_mode = args.wandb_mode

    # update name
    conf.name = args.name
    conf.model_name_or_path = f"pretrain/{args.name}"

    # save the configuration
    save_dir = os.path.join(PROJECT_ROOT, "conf/eval_fewshot", f"{args.name}.yaml")
    OmegaConf.save(config=conf, f=save_dir)
    print(f"Configuration saved to: {save_dir}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OmegaConf configuration for pretrain and eval_fewshot.")

    # Add arguments (examples) - add as many as needed
    parser.add_argument("--name", type=str, default=None, required=True,
                        help="Name of the configuration file.")
    parser.add_argument("--base_cfg", type=str, default="example",
                        help="Base config file for experiments.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Pretraining dataset.")
    parser.add_argument("--dataloader_num_workers", type=int, default=None,
                        help="Increase this when you want to use TokenizedDataset.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per device train batch size.")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Model checkpoint in HuggingFace.")
    parser.add_argument("--ln_share", type=bool, default=True,
                        help="Decide whether to share LN parameters.")
    parser.add_argument("--save_interval", type=float, default=0.25)
    parser.add_argument("--skip_first_loop", type=bool, default=False,
                        help="Whether to skip the first loop in relaxation.")
    parser.add_argument("--alpha_pattern", type=float, default=2.0)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline"],
                        help="The online mode is used when you want to log locally, typically when no internet connection is available.")
    args: argparse.Namespace = parser.parse_args()

    # Call the config creation and saving function
    generate_configs(args=args)