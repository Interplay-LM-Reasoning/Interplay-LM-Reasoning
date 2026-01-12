#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi

BASE_MODEL=id2-10_0.2easy_0.3medium_0.5hard
EVAL_DATA_ROOT=data/composition/test \
CHECKPOINT_PATH=saves/composition-10B/op_level/id2-10_0.2easy_0.3medium_0.5hard/rl/id2-10_0.2easy_0.3medium_0.5hard--op11-14_uniform_process_strict/global_step_120/actor/huggingface \
LLAMA_CONFIG=scripts/composition/op-difficulty-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5/${BASE_MODEL}.yaml \
VERL_EXTRA_ARGS="actor_rollout_ref.model.path=${CHECKPOINT_PATH} data.preset_path=data/PRESET.json" \
EVAL_OUTPUT_TEMPLATE="results/composition/op-difficulty-10B/rl-200steps/rl/op11-14_uniform_process_strict/{run_name}/global_step_120" \
VERL_CONFIG=scripts/composition/op-difficulty-10B/rl-200steps/op11-14_uniform_process_strict.yaml \
./scripts/meta_run.sh \
 --skip-pretrain --do-eval
