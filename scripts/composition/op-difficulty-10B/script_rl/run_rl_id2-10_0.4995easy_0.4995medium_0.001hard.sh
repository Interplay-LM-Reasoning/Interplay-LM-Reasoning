#!/bin/bash

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi

BASE_MODEL=id2-10_0.4995easy_0.4995medium_0.001hard


LLAMA_CONFIG=scripts/composition/op-difficulty-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5/${BASE_MODEL}.yaml \
VERL_CONFIG=scripts/composition/op-difficulty-10B/rl-200steps/op7-10_uniform.yaml \
./scripts/meta_run.sh \
 --skip-pretrain

LLAMA_CONFIG=scripts/composition/op-difficulty-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5/${BASE_MODEL}.yaml \
VERL_CONFIG=scripts/composition/op-difficulty-10B/rl-200steps/op9-12_uniform.yaml \
./scripts/meta_run.sh \
 --skip-pretrain
 
LLAMA_CONFIG=scripts/composition/op-difficulty-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5/${BASE_MODEL}.yaml \
VERL_CONFIG=scripts/composition/op-difficulty-10B/rl-200steps/op11-14_uniform.yaml \
./scripts/meta_run.sh \
 --skip-pretrain

 LLAMA_CONFIG=scripts/composition/op-difficulty-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5/${BASE_MODEL}.yaml \
VERL_CONFIG=scripts/composition/op-difficulty-10B/rl-200steps/op17-20_uniform.yaml \
./scripts/meta_run.sh \
 --skip-pretrain