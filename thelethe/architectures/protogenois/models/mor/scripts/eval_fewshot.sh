#!/bin/bash

launcher_type="accelerate"

if [[ "$1" == "deepspeed" ]] || [[ "$1" == "accelerate" ]]; then
  launcher_type="$1"
  shift
fi

if [[ "$1" == "online" ]] || [[ "$1" == "offline" ]]; then
  user_specified_run_mode="$1"
  shift
fi

if [ -n "$user_specified_run_mode" ]; then
  export WANDB_MODE="$user_specified_run_mode"
else
  export WANDB_MODE="online" # Default value if not specified
fi
echo "INFO: WANDB_MODE is set to '$WANDB_MODE'"

# Use the first argument as GPU numbers.
gpu_numbers="$1"

# Use all arguments after the first one as config-names.
shift # Remove the first argument

# Check if GPU numbers are provided.
if [ -z "$gpu_numbers" ]; then
  echo "Usage: $0 <GPU numbers> <config name 1> <config name 2> ..."
  exit 1
fi
# num_processes=$(echo "$gpu_numbers" | tr ',' '\n' | grep -c .)
first_gpu=$(echo "$gpu_numbers" | tr ',' '\n' | head -n 1)

# if [ "$num_processes" -eq 1 ]; then
#   config="acc_configs/single_gpu_config.yaml"
# elif [ "$num_processes" -gt 1 ]; then
#   config="acc_configs/default_config.yaml"
# fi

# Check if at least one config-name argument is provided.
if [ $# -eq 0 ]; then
  echo "Usage: $0 <GPU numbers> <config name 1> <config name 2> ..."
  exit 1
fi

# Loop through each config-name argument and execute the command.
for config_name in "$@"; do
  # Generate a 5-digit random port.
  random_port=$((10000 + RANDOM % 90000))

  # Execute the command.
  echo "Running with config: $config_name, GPUs: $gpu_numbers, Port: $random_port"
  
  if [ "$launcher_type" == "deepspeed" ]; then
    echo "Launch with DeepSpeed..."
    HYDRA_FULL_ERROR=1 deepspeed --include "localhost:$first_gpu" --no_local_rank --master_port "$random_port" eval_fewshot.py --config-name "$config_name"
  elif [ "$launcher_type" == "accelerate" ]; then
    echo "Launch with accelerate..."
    HYDRA_FULL_ERROR=1 accelerate launch --config_file acc_configs/single_gpu_config.yaml --gpu_ids $first_gpu --num_processes 1 \
    --main_process_port "$random_port" eval_fewshot.py --config-name "$config_name"
  fi
  # Optional: Add a delay after each config execution.
  # sleep 5 # Wait for 5 seconds.
done

echo "All configurations processed."