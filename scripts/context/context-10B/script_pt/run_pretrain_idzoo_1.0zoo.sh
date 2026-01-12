#!/bin/bash

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi

LLAMA_WANDB_PROJECT=context-10B-context-PT \
EVAL_DATA_ROOT=data/context/test \
LLAMA_CONFIG=scripts/context/context-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5/idzoo_1.0zoo.yaml \
    ./scripts/meta_run.sh --skip-rl
