#!/bin/bash
set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
  exit 1
fi

MODEL_SLUG="idzoo_0.999zoo_0.001teacher"
LLAMA_CONFIG="scripts/context/context-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5/${MODEL_SLUG}.yaml"
PRETRAIN_CHECKPOINT="checkpoint-17078"
RL_VARIANTS=(
  "contextzoo_0.1zoo_0.9teacher"
  "contextzoo_0.5zoo_0.5teacher"
  "contextzoo_0.9zoo_0.1teacher"
  "contextzoo_1.0teacher"
)
RL_STEP="${RL_STEP:-200}"
RESULTS_ROOT="results/context-10B/context_level/${MODEL_SLUG}/eval"
EVAL_DATA_ROOT="data/context/test"

run_pretrain_eval() {
  env \
    LLAMA_CONFIG="${LLAMA_CONFIG}" \
    VERL_CONFIG= \
    EVAL_DATA_ROOT="${EVAL_DATA_ROOT}" \
    EVAL_CHECKPOINTS_PATTERN="${PRETRAIN_CHECKPOINT}" \
    EVAL_OUTPUT_TEMPLATE="${RESULTS_ROOT}/pt/${PRETRAIN_CHECKPOINT}" \
    ./scripts/meta_run.sh --do-eval
}

run_rl_eval() {
  local variant="$1"
  local verl_config="scripts/context/context-10B/rl-200steps/${variant}.yaml"
  local ck_root="saves/context-10B/context_level/${MODEL_SLUG}/rl/${MODEL_SLUG}--${variant}/global_step_${RL_STEP}/actor/huggingface"
  local output_dir="${RESULTS_ROOT}/rl/${variant}/global_step_${RL_STEP}"

  if [[ ! -f "${verl_config}" ]]; then
    echo "Skipping ${variant}; missing config ${verl_config}" >&2
    return
  fi
  if [[ ! -d "${ck_root}" ]]; then
    echo "Skipping ${variant}; missing checkpoint ${ck_root}" >&2
    return
  fi

  env \
    LLAMA_CONFIG="${LLAMA_CONFIG}" \
    VERL_CONFIG="${verl_config}" \
    EVAL_DATA_ROOT="${EVAL_DATA_ROOT}" \
    EVAL_CHECKPOINTS_ROOT="${ck_root}" \
    EVAL_OUTPUT_TEMPLATE="${output_dir}" \
    ./scripts/meta_run.sh --do-eval
}

mkdir -p "${RESULTS_ROOT}"
run_pretrain_eval
for variant in "${RL_VARIANTS[@]}"; do
  run_rl_eval "${variant}"
done
