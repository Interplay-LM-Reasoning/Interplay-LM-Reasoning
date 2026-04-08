#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi

# This scripts corresponds to the mid / post-training data mixing ratio mentioned in Section 5.
# We explore on the extrapolation setting:
# For pre-training: (the same as Section 3) we use the data mixture of 20% easy, 30% medium, and 50% hard tasks.
bash scripts/composition/op-difficulty-10B/script_pt/run_pretrain_id2-10_0.2easy_0.3medium_0.5hard.sh

# In Section 5, we demonstrate the results of totally 1B tokens (50 steps for RL) for mid- / post-training with different mixing ratios.
# For mid-training, we first continue the pre-training with:
bash scripts/composition/op-difficulty-10B/script_cpt_rl/id2-10_0.2easy_0.3medium_0.5hard_cpt11-14/run_rl_op11-14_uniform_plus.sh
# Then, if you want to run with 50 step budget, you may use the script:
bash scripts/composition/op-difficulty-10B/script_cpt_rl/id2-10_0.2easy_0.3medium_0.5hard_cpt11-14/run_cpt_rl_op11-14_unfirom_50step_budget.sh

# Alternatively, you may use the script with different budgets in the directory:
# scripts/composition/op-difficulty-10B/script_cpt_rl/id2-10_0.2easy_0.3medium_0.5hard_cpt11-14