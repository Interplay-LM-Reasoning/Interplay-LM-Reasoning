#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi

# This scripts corresponds to the extrapolation RL training in Section 3.
# We run GRPO post-training for composition tasks with different difficulty distributions:

# For pre-training: we use the data mixture of 20% easy, 30% medium, and 50% hard tasks.
bash scripts/composition/op-difficulty-10B/script_pt/run_pretrain_id2-10_0.2easy_0.3medium_0.5hard.sh

# For GRPO post-training with different variants of the composition tasks.
bash scripts/composition/op-difficulty-10B/script_rl/run_rl_id2-10_0.2easy_0.3medium_0.5hard.sh


