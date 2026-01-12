

#!/bin/bash

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi
EVAL_DATA_ROOT=data/context/test \
LLAMA_CONFIG=scripts/mixed/mix-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5/0.999zoo_op2-20+0.001teacher_1.0op2.yaml \
VERL_CONFIG=scripts/mixed/mix-10B/rl-200steps/contextzoo_0.5zoo_0.5teacher_process_strict.yaml \
./scripts/meta_run.sh \
 --skip-pretrain
