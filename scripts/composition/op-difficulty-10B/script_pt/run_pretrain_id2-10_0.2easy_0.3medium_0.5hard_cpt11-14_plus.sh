#!/bin/bash

set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi

CONFIG_ROOT="scripts/composition/op-difficulty-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5"

CONFIG_NAME=id2-10_0.2easy_0.3medium_0.5hard_cpt11-14_plus.yaml
CONFIG_NAME="${CONFIG_NAME}" \
    LLAMA_CONFIG="${CONFIG_ROOT}/${CONFIG_NAME}" \
        ./scripts/meta_run.sh --skip-rl "$@"
