#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    "run_cpt_rl_op9-12_uniform_ckpt961.sh"
    "run_cpt_rl_op9-12_uniform_ckpt1922.sh"
    "run_cpt_rl_op9-12_uniform_ckpt2883.sh"
    "run_cpt_rl_op9-12_uniform_ckpt3844.sh"
    "run_cpt_rl_op9-12_uniform_ckpt4804.sh"
)

for script in "${SCRIPTS[@]}"; do
    bash "${SCRIPT_DIR}/${script}"
done
