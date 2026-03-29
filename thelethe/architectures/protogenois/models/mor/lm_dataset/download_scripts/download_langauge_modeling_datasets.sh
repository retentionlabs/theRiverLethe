#!/bin/bash

# Array of dataset names
datasets=("fineweb-edu-dedup")

# Directory where datasets will be stored
hf_datasets_dir="hf_datasets"

# Directory containing the download scripts
download_scripts_dir="lm_dataset/download_scripts"

# Iterate through each dataset
for dataset in "${datasets[@]}"; do
  # Dataset path
  dataset_path="$hf_datasets_dir/$dataset"

  # Check if dataset exists
  if [ ! -d "$dataset_path" ]; then
    echo "Dataset '$dataset' not found. Downloading..."

    # Download script path
    download_script="$download_scripts_dir/download_$dataset.sh"

    # Check if download script exists
    if [ -f "$download_script" ]; then
      # Execute download script
      bash "$download_script"
    else
      echo "Error: Download script for '$dataset' not found at '$download_script'"
    fi
  else
    echo "Dataset '$dataset' already exists. Skipping download."
  fi

  # Determine Hugging Face cache directory and final destination based on dataset
  case "$dataset" in
    "cosmopedia-v2" | "fineweb-edu-dedup" | "python-edu")
      cached_dataset_path="hf_cache/datasets/HuggingFaceTB___smollm-corpus/$dataset"
      final_dataset_path="$hf_datasets_dir/$dataset" 
      ;;
    "math-code-pile")
      cached_dataset_path="hf_cache/datasets/MathGenie___math_code-pile/default"
      final_dataset_path="$hf_datasets_dir/$dataset"
      ;;
    "open-web-math")
      cached_dataset_path="hf_cache/datasets/open-web-math___open-web-math/default"
      final_dataset_path="$hf_datasets_dir/$dataset"
      ;;
    "slimpajama")
      cached_dataset_path="hf_cache/datasets/cerebras___slim_pajama-627_b/default"
      final_dataset_path="$hf_datasets_dir/$dataset"
      ;;
    "starcoderdata")
      cached_dataset_path="hf_cache/datasets/bigcode___starcoderdata/default-fad560847e57bb78"
      final_dataset_path="$hf_datasets_dir/$dataset"
      ;;
    "finemath")
      cached_dataset_path="hf_cache/datasets/HuggingFaceTB___finemath/finemath-4plus"
      final_dataset_path="$hf_datasets_dir/$dataset"
      ;;
    *)
      cached_dataset_path=""
      final_dataset_path="$hf_datasets_dir/$dataset"
      ;;
  esac

  # Check if dataset exists in cache
  if [ -d "$cached_dataset_path" ]; then
    echo "Found '$dataset' in cache."

    # Move cached dataset to the final destination if it doesn't already exist
    if [ ! -d "$final_dataset_path" ]; then
      echo "Moving '$dataset' from cache to '$final_dataset_path'..."
      mv "$cached_dataset_path" "$final_dataset_path"
      echo "Moved '$dataset' to '$final_dataset_path'."
    else
      echo "Dataset '$dataset' already exists in '$final_dataset_path'. Skipping move."
    fi
  fi

done

echo "Dataset download process completed."