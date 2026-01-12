#!/bin/bash

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi

LLAMA_CONFIG=scripts/context/context-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5/idzoo_0.999zoo_0.001teacher.yaml \
VERL_CONFIG=scripts/context/context-10B/rl-200steps/contextzoo_0.5zoo_0.5teacher.yaml \
./scripts/meta_run.sh \
 --skip-pretrain
