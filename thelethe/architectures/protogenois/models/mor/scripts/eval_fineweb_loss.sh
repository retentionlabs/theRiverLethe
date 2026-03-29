#!/bin/bash

# Check if at least a GPU ID and one experiment name are provided
if [ "$#" -lt 2 ]; then # Need at least 2 arguments: GPU_ID and exp_name1
    echo "Usage: $0 <gpu_id> <exp_name1> [exp_name2 ...]"
    echo "Please provide a GPU ID and at least one experiment name."
    echo "Example: $0 0 experiment_alpha experiment_beta"
    exit 1
fi

# The first argument is the GPU ID
GPU_ID=$1
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Setting CUDA_VISIBLE_DEVICES to: $GPU_ID"

# Remove the first argument (GPU_ID) from the list of arguments,
# so $@ now contains only the experiment names.
shift

# Check if any arguments are provided to the bash script
if [ $# -eq 0 ]; then
    echo "Usage: $0 <exp_name1> [exp_name2 ...]"
    echo "Please provide at least one experiment name."
    exit 1
fi

echo "Starting evaluation for experiments: $@"

# Pass all script arguments ($@) to the --exp_names option of the Python script
python evaluate_fineweb_test.py --exp_names "$@" # The quotes around "$@" are important for handling spaces within individual arguments if that were ever a case, though not typical for exp_names. For this nargs='+', it ensures each argument is passed as a separate item.

echo "All configurations processed."