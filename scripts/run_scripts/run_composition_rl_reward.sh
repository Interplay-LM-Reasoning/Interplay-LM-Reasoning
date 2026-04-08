#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi

# This scripts corresponds to the extrapolation RL training with reward shaping in Section 6.

# For pre-training: we use the data mixture of 20% easy, 30% medium, and 50% hard tasks.
bash scripts/composition/op-difficulty-10B/script_pt/run_pretrain_id2-10_0.2easy_0.3medium_0.5hard.sh

# For GRPO post-training with reward shaping.

# 1. fully answer score in the outcome rewards with strict proces verification.
bash scripts/composition/op-difficulty-10B/script_rl/run_rl_id2-10_0.2easy_0.3medium_0.5hard/run_r11-14_process_strict.sh

# 2. 50% process score + 50% answer score in the outcome rewards.
bash scripts/composition/op-difficulty-10B/script_rl/run_rl_id2-10_0.2easy_0.3medium_0.5hard/run_r11-14_process0.5.sh

# 3. 80% process score + 20% answer score in the outcome rewards.
basg scripts/composition/op-difficulty-10B/script_rl/run_rl_id2-10_0.2easy_0.3medium_0.5hard/run_r11-14_process.sh