#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi

# receive cuda_visible_devices from command line
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
CHECKPOINTS_ROOT="saves/composition-10B/op_level/id2-10_0.2easy_0.3medium_0.5hard/rl/id2-10_0.2easy_0.3medium_0.5hard--op7-10_uniform_process_strict"
VERL_CONFIG="scripts/composition/op-difficulty-10B/rl-200steps/op7-10_uniform_process_strict.yaml"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT}" \
VERL_CONFIG="${VERL_CONFIG}" \
scripts/composition/run_rl_eval.sh

CHECKPOINTS_ROOT="saves/composition-10B/op_level/id2-10_0.2easy_0.3medium_0.5hard/rl/id2-10_0.2easy_0.3medium_0.5hard--op9-12_uniform_process_strict"
VERL_CONFIG="scripts/composition/op-difficulty-10B/rl-200steps/op9-12_uniform_process_strict.yaml"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT}" \
VERL_CONFIG="${VERL_CONFIG}" \
scripts/composition/run_rl_eval.sh
