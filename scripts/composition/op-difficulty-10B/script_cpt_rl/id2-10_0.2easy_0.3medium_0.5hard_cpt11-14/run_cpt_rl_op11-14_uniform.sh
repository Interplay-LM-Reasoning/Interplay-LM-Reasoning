#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    "run_cpt_rl_op11-14_uniform_ckpt971.sh"
    "run_cpt_rl_op11-14_uniform_ckpt1942.sh"
    "run_cpt_rl_op11-14_uniform_ckpt2913.sh"
    "run_cpt_rl_op11-14_uniform_ckpt3884.sh"
    "run_cpt_rl_op11-14_uniform_ckpt4851.sh"
)

for script in "${SCRIPTS[@]}"; do
    bash "${SCRIPT_DIR}/${script}"
done
