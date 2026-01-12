#!/bin/bash

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi

BASE_MODEL=id2-10_0.475easy_0.475medium_0.05hard

LLAMA_CONFIG=scripts/composition/op-difficulty-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5/${BASE_MODEL}.yaml \
VERL_CONFIG=scripts/composition/op-difficulty-10B/rl-200steps/op7-10_uniform.yaml \
./scripts/meta_run.sh \
 --skip-pretrain
