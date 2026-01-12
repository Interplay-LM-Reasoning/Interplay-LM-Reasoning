#!/bin/bash

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi
EVAL_DATA_ROOT=data/composition/test \
LLAMA_CONFIG=scripts/composition/op-difficulty-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5/id2-10_0.4easy_0.4medium_0.2hard.yaml \
    ./scripts/meta_run.sh --skip-rl --do-eval
