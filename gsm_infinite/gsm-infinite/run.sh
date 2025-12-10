#!/bin/bash

# Source the configuration file
source config.sh

# Function to generate a comma-separated string of numbers with a given stride
generate_sequence() {
  local start=$1
  local end=$2
  local stride=$3
  numbers=$(seq "$start" "$stride" "$end")
  result=$(echo "$numbers" | paste -s -d, -)
  echo "$result"
}

for length in "${lengths[@]}"; do
    for suffix in "${dataset_suffixes[@]}"; do
        dataset_name="${dataset_base}_${suffix}"
        save_dataset="$suffix"

        config_key="${length}_${suffix}"
        if [[ -z "${ops_config[$config_key]}" ]]; then
            echo "Skipping ${dataset_name} because no ops configuration found for $config_key."
            continue  # Skip to the next iteration
        else
            ops_start=$(echo "${ops_config[$config_key]}" | jq -r '.start')
            ops_end=$(echo "${ops_config[$config_key]}" | jq -r '.end')
            ops_stride=$(echo "${ops_config[$config_key]}" | jq -r '.stride')
            ops=$(generate_sequence "$ops_start" "$ops_end" "$ops_stride")
        fi

        echo "Running with length: $length, dataset: $dataset_name, save-dataset: $save_dataset"

        if [[ "$run_sampling" == true && ! "$run_symbolic_evaluation" == true && ! "$run_realistic_evaluation" == true ]]; then
            # Set API keys for sampling
            export OPENAI_BASE_URL=$SAMPLER_OPENAI_BASE_URL
            export OPENAI_API_KEY=$SAMPLER_OPENAI_API_KEY

            # Set temperature and limit based on suffix
            if [[ "$suffix" == "symbolic" ]]; then
                temperature=$temperature_symbolic
                limit=$limit_symbolic
                python3 pred/pred.py \
                    --dataset-name "$dataset_name" \
                    --model-name "$model_name" \
                    --save-dataset "$save_dataset" \
                    --save-name "$save_name" \
                    --backend-type "$backend_type" \
                    --num-samples "$num_samples" \
                    --temperature "$temperature" \
                    --max-tokens "$max_tokens" \
                    --length "$length" \
                    --op-range "$ops" \
                    --batch-size "$batch_size" \
                    --limit "$limit"
            else
                temperature=$temperature_realistic
                limit=$limit_realistic
                # filter_arg=$(echo "--filter-config \"$filter_config\"") # Corrected line
                python3 pred/pred.py \
                    --dataset-name "$dataset_name" \
                    --model-name "$model_name" \
                    --save-dataset "$save_dataset" \
                    --save-name "$save_name" \
                    --backend-type "$backend_type" \
                    --num-samples "$num_samples" \
                    --temperature "$temperature" \
                    --max-tokens "$max_tokens" \
                    --length "$length" \
                    --op-range "$ops" \
                    --batch-size "$batch_size" \
                    --limit "$limit" \
                    --filter-config "$filter_config"
            fi

        fi

        if [[ "$run_evaluation" == true || "$run_symbolic_evaluation" == true ]] && [[ "$suffix" == "symbolic" ]]; then
            # Set API keys for evaluation
            export OPENAI_BASE_URL=$EVAL_OPENAI_BASE_URL
            export OPENAI_API_KEY=$EVAL_OPENAI_API_KEY

            python3 pred/eval_symbolic.py \
                --save-name "$save_name" \
                --num-samples "$num_samples" \
                --length "$length"
        fi


        if [[ "$run_evaluation" == true || "$run_realistic_evaluation" == true ]] && [[ "$suffix" != "symbolic" ]]; then
            python3 pred/eval_realistic.py \
                --save-dataset "$save_dataset" \
                --save-name "$save_name" \
                --num-samples "$num_samples" \
                --length "$length" \
                --filter-config "$filter_config" # Add filter argument only for medium/hard
        fi
    done
done