#!/bin/bash

set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi

BASE_MODEL=id2-10_0.2easy_0.3medium_0.5hard
CHECKPOINTS_ROOT="saves/composition-10B/op_level/${BASE_MODEL}/cpt0.2-uniform_0.8-11-14"

EVAL_DATA_ROOT="data/composition/test"
EVAL_DATA_DIR="${EVAL_DATA_ROOT}"
EVAL_CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT}"
EVAL_CHECKPOINTS_PATTERN="checkpoint-*"
EVAL_SUMMARY_FILENAME="summary_all_checkpoints"
LLAMA_CONFIG="scripts/composition/op-difficulty-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5/${BASE_MODEL}_cpt11-14.yaml"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT}" \
EVAL_DATA_DIR="${EVAL_DATA_DIR}" \
EVAL_CHECKPOINTS_ROOT="${EVAL_CHECKPOINTS_ROOT}" \
EVAL_CHECKPOINTS_PATTERN="${EVAL_CHECKPOINTS_PATTERN}" \
EVAL_SUMMARY_FILENAME="${EVAL_SUMMARY_FILENAME}" \
LLAMA_CONFIG="${LLAMA_CONFIG}" \
./scripts/meta_run.sh \
  --do-eval \
  --skip-pretrain \
  "$@"
