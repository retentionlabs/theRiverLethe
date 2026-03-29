import os
import yaml
import copy
import math
import warnings
from pathlib import Path
from paths import SAVE_DIR, DATA_DIR, PROJECT_ROOT, HF_CACHE_DIR; os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
from typing import Callable, List, Optional, Tuple, Union, Dict # Added Dict
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
import argparse # Added for command-line arguments
import json # Added for JSON output

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import transformers
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer, AutoConfig
from transformers.utils import ModelOutput
from transformers.processing_utils import Unpack

from lm_dataset.language_modeling_dataset import LanguageModelingDataset
from lm_dataset.data_preprocessing import AddLabels, RemoveIndex
from model.util import load_model_from_config
from model.sharing_strategy import SHARING_STRATEGY
from model.relaxation.util import relax_weight_sharing
from util.tokenizer import load_tokenizer_from_config
from util.config import preprocess_config
from util.misc import get_torch_dtype, get_latest_checkpoint_path

# LM_DATASETS definition (same as in the original code)
LM_DATASETS = {
    "slimpajama": {"path": f"{DATA_DIR}/slimpajama", "split": "train"},
    "slimpajama_chunk1": {"path": "json", "data_files": f"{DATA_DIR}/slimpajama_chunk1/*.jsonl", "split": "train"},
    "slimpajama_test": {"path": "json", "data_files": f"{DATA_DIR}/slimpajama_test/*.jsonl", "split": "train"},
    "pg19": {"path": f"emozilla/pg19-test", "split": "test", "trust_remote_code": "True"},
    "cosmopedia": {"path": f"{DATA_DIR}/cosmopedia-v2", "split": "train"},
    "fineweb_edu": {"path": f"{DATA_DIR}/fineweb-edu-dedup", "split": "train"},
    "fineweb_test": {"path": f"{DATA_DIR}/fineweb-test", "split": "train"},
    "python_edu": {"path": f"{DATA_DIR}/python-edu", "split": "train"},
    "open_web_math": {"path": f"{DATA_DIR}/open-web-math", "split": "train"},
    "math_code_pile": {"path": f"{DATA_DIR}/math-code-pile", "split": "train"},
    "starcoderdata": {"path": f"{DATA_DIR}/starcoderdata", "split": "train"},  # "data_dir": "python",
    "finemath": {"path": f"{DATA_DIR}/finemath", "split": "train"},  # "name": "finemath-4plus",
}


def load_dataset_from_config(cfg, tokenizer):
    dataset_name_str = cfg.dataset # cfg.dataset might be a string or already a list
    if isinstance(dataset_name_str, str):
        dataset_names = [ds.strip() for ds in dataset_name_str.split(',')]
    else: # Assuming it's already a list-like OmegaConf object
        dataset_names = [str(ds).strip() for ds in dataset_name_str]

    if len(dataset_names) > 1:
        assert all(ds in LM_DATASETS for ds in dataset_names), "Only LM datasets can be combined"
        assert "weights" in cfg, "When combining datasets, weights must be provided"
        weights_str = cfg.weights
        if isinstance(weights_str, str):
            weights = [float(w) for w in weights_str.split(',')]
        else: # Assuming it's already a list-like OmegaConf object of numbers
            weights = [float(w) for w in weights_str]
        assert len(dataset_names) == len(weights), "Number of weights must match number of datasets"

    if all(ds in LM_DATASETS for ds in dataset_names):
        dataset_type = "lm"
        loaded_datasets = []
        for ds_name in dataset_names:
            _dataset = load_dataset(**LM_DATASETS[ds_name], streaming=True)
            if ds_name == "starcoderdata":
                _dataset = _dataset.rename_column("content", "text") # Ensure renaming happens before appending
            loaded_datasets.append(_dataset)

        if len(loaded_datasets) == 1:
            final_dataset = loaded_datasets[0]
        else:
            # Ensure weights are correctly parsed as floats for interleave_datasets
            probabilities = [float(w) for w in cfg.weights.split(',')] if isinstance(cfg.weights, str) else [float(w) for w in cfg.weights]
            final_dataset = interleave_datasets(loaded_datasets, probabilities=probabilities, seed=42)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    transforms = [
        AddLabels(),
        RemoveIndex(),
    ]

    if dataset_type == "lm":
        return LanguageModelingDataset(final_dataset, tokenizer,
                                    max_length=cfg.max_length,
                                    transforms=transforms,
                                    global_shuffling=cfg.get("global_shuffling", False),
                                    local_shuffling=cfg.get("local_shuffling", False),
                                    add_bos_token=cfg.get("add_bos_token", False),)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def evaluate_model(exp_name: str, global_sample_number: int) -> Optional[float]:
    """
    Evaluates a model specified by exp_name on the fineweb_test dataset.
    Returns the average loss if successful, None otherwise.
    """
    print(f"Evaluating experiment: {exp_name}")
    cfg_path = os.path.join(PROJECT_ROOT, "conf/pretrain", f"{exp_name}.yaml")
    if not os.path.exists(cfg_path):
        print(f"Configuration file not found for {exp_name} at {cfg_path}. Skipping.")
        return None

    cfg = OmegaConf.load(cfg_path)
    cfg = preprocess_config(cfg) # This should handle setting default max_length if not present

    cfg.resume_from_checkpoint = False
    
    model = load_model_from_config(cfg)
    
    lora_init_dict = None # Initialize lora_init_dict

    if cfg.recursive.get("enable"):
        # Assuming cfg.model is a DictConfig, access name with .name or adjust as needed
        model_strategy_key = cfg.model.name if hasattr(cfg.model, "name") else cfg.model
        model, lora_init_dict = SHARING_STRATEGY[model_strategy_key](cfg, model)

    if "kv_sharing" in cfg and cfg.kv_sharing.get("enable"):
        model.set_kv_sharing_config(cfg)

    if cfg.get("relaxation") and cfg.relaxation.get("enable"):
        model = relax_weight_sharing(cfg, model, lora_init_dict=lora_init_dict)

        if cfg.get("resume_from_checkpoint"): # Check if resume_from_checkpoint is True
            latest_checkpoint_path_obj = get_latest_checkpoint_path(cfg, resume_step=cfg.resume_step if ("resume_step" in cfg and cfg.resume_step is not None) else None)
            if latest_checkpoint_path_obj:
                latest_checkpoint = str(latest_checkpoint_path_obj)
                pytorch_model_path_relax = os.path.join(latest_checkpoint, "pytorch_model.bin")
                safetensors_model_path_relax = os.path.join(latest_checkpoint, "model.safetensors")

                loaded_relaxation_checkpoint = False
                if os.path.exists(pytorch_model_path_relax):
                    try:
                        state_dict = torch.load(pytorch_model_path_relax, map_location='cpu')
                        model.get_base_model().load_state_dict(state_dict)
                        loaded_relaxation_checkpoint = True
                        print(f"Loaded relaxation checkpoint from {pytorch_model_path_relax}")
                    except Exception as e:
                        print(f"Error loading relaxation pytorch_model.bin: {e}")
                elif os.path.exists(safetensors_model_path_relax):
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(safetensors_model_path_relax, device='cpu')
                        model.get_base_model().load_state_dict(state_dict)
                        loaded_relaxation_checkpoint = True
                        print(f"Loaded relaxation checkpoint from {safetensors_model_path_relax}")
                    except Exception as e:
                        print(f"Error loading relaxation model.safetensors: {e}")

                if not loaded_relaxation_checkpoint:
                     print(f"Specified resume_from_checkpoint for relaxation, but checkpoint not found at {latest_checkpoint}. Model will use initial weights for relaxation stage if not overridden by main exp checkpoint.")
            else:
                print(f"Specified resume_from_checkpoint for relaxation, but get_latest_checkpoint_path returned None.")

    if "mor" in cfg and cfg.mor.get("enable"):
        if cfg.mor.type == "expert":
            model.transform_layer_to_mor_expert(cfg)
        elif cfg.mor.type == "token":
            model.transform_layer_to_mor_token(cfg)
        else:
            raise ValueError(f"Unknown MoR type {cfg.mor.type}.")

    # Define checkpoint paths for the main experiment
    checkpoint_dir = os.path.join(SAVE_DIR, "pretrain", exp_name)
    pytorch_model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    safetensors_model_path = os.path.join(checkpoint_dir, "model.safetensors")

    main_checkpoint_loaded = False
    if os.path.exists(pytorch_model_path):
        try:
            state_dict = torch.load(pytorch_model_path, map_location='cpu') # Load to CPU first to save GPU memory
            model.load_state_dict(state_dict)
            main_checkpoint_loaded = True
            print(f"Successfully loaded main experiment checkpoint from {pytorch_model_path}")
        except Exception as e:
            print(f"Error loading main experiment pytorch_model.bin for {exp_name}: {e}")
            # If checkpoint loading fails, we might not want to proceed with this model.
            # Depending on desired behavior, could return None here.
    elif os.path.exists(safetensors_model_path):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_model_path, device='cpu') # Load to CPU first
            model.load_state_dict(state_dict)
            main_checkpoint_loaded = True
            print(f"Successfully loaded main experiment checkpoint from {safetensors_model_path}")
        except Exception as e:
            print(f"Error loading main experiment model.safetensors for {exp_name}: {e}")
            # Similarly, could return None here.

    if not main_checkpoint_loaded:
        warnings.warn(f"No main checkpoint found or loaded for {exp_name} at {checkpoint_dir}. Evaluation might use base/relaxation weights or be inaccurate. Skipping further evaluation for this experiment.")
        # Clean up before returning if model was partially initialized
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    # Load dataset
    cfg_for_dataset = deepcopy(cfg)
    cfg_for_dataset.dataset = "fineweb_test"

    tokenizer = load_tokenizer_from_config(cfg_for_dataset)
    dataset = load_dataset_from_config(cfg_for_dataset, tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device, dtype=get_torch_dtype(cfg))
    model.eval()

    total_loss = 0
    evaluated_samples = 0
    avg_loss = None # Initialize avg_loss

    print(f"Starting evaluation for {exp_name} on fineweb_test using device: {device}...")
    try:
        for i, sample in enumerate(dataset):
            if i >= global_sample_number:
                break
            try:
                input_ids = sample["input_ids"].unsqueeze(0).to(device)
                attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
                labels = sample["labels"].unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=True,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                total_loss += output.loss.item()
                evaluated_samples += 1
            except Exception as e:
                print(f"Error processing sample {i} for {exp_name}: {e}")
                continue

        if evaluated_samples > 0:
            avg_loss = round(total_loss / evaluated_samples, 4)
            print(f"Experiment: {exp_name}, fineweb_test loss ({evaluated_samples} samples): {avg_loss}")
        else:
            print(f"Experiment: {exp_name}, No samples were evaluated successfully.")
            avg_loss = None # Explicitly set to None if no samples evaluated

    except Exception as e:
        print(f"An error occurred during evaluation for {exp_name}: {e}")
        avg_loss = None # Set to None on major evaluation error
        
    finally:
        del model
        del dataset
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple language model experiments.")
    parser.add_argument(
        "--exp_names",
        type=str,
        nargs='+',
        required=True,
        help="List of experiment names to evaluate (e.g., exp_A exp_B)."
    )
    parser.add_argument(
        "--sample_number",
        type=int,
        default=500,
        help="Number of samples to evaluate for each experiment."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Path to save the JSON file with evaluation results."
    )

    args = parser.parse_args()
    experiment_names = args.exp_names
    global_sample_number = args.sample_number
    output_file_path = args.output_file

    if not experiment_names:
        print("Please provide at least one experiment name using --exp_names.")
        return

    print(f"Running evaluation for experiments: {', '.join(experiment_names)}")
    print(f"Number of samples per experiment: {global_sample_number}")

    all_results: Dict[str, Optional[float]] = {} # Initialize dictionary to store all results

    for exp_name in experiment_names:
        loss = evaluate_model(exp_name, global_sample_number)
        all_results[exp_name] = loss # loss can be float or None
        print("-" * 70)

    # Save the results to a JSON file
    try:
        with open(output_file_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"Evaluation results saved to {output_file_path}")
    except IOError as e:
        print(f"Error saving results to JSON file {output_file_path}: {e}")

    print("\nSummary of results:")
    for exp, res_loss in all_results.items():
        if res_loss is not None:
            print(f"  {exp}: {res_loss}")
        else:
            print(f"  {exp}: Evaluation failed or no samples processed.")


if __name__ == "__main__":
    main()