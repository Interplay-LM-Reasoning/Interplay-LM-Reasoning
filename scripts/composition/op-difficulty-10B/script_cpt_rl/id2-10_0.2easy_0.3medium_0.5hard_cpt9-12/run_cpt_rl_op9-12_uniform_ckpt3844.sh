#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

BASE_MODEL="id2-10_0.2easy_0.3medium_0.5hard"
PT_CONFIG="scripts/composition/op-difficulty-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5/${BASE_MODEL}.yaml"
CPT_CONFIG_ROOT="scripts/composition/op-difficulty-10B/cpt-rl-200steps"
CONFIG_NAME="cpt-rl-op9-12_uniform-ckpt3844.yaml"
CPT_CHECKPOINT_ROOT="saves/composition-10B/op_level/${BASE_MODEL}/cpt0.2-uniform_0.8-9-12"
CPT_CHECKPOINT_PATH="${CPT_CHECKPOINT_ROOT}/checkpoint-3844"

EVAL_DATA_ROOT="data/composition/test" \
EVAL_DATA_DIR="${EVAL_DATA_ROOT}" \
VERL_EXTRA_ARGS="actor_rollout_ref.model.path=${CPT_CHECKPOINT_PATH} data.preset_path=data/PRESET.json" \
LLAMA_CONFIG="${PT_CONFIG}" \
VERL_CONFIG="${CPT_CONFIG_ROOT}/${CONFIG_NAME}" \
./scripts/meta_run.sh \
  --skip-pretrain
