#!/bin/bash
set -euo pipefail


if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi

BASE_MODEL=id2-10_0.495easy_0.495medium_0.01hard

LLAMA_CONFIG=scripts/composition/op-difficulty-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5/${BASE_MODEL}.yaml
VERL_CONFIG=scripts/composition/op-difficulty-10B/rl-200steps/op9-12_uniform.yaml
LLAMA_CONFIG="${LLAMA_CONFIG}" \
VERL_CONFIG="${VERL_CONFIG}" \
./scripts/meta_run.sh \
 --skip-pretrain
