#!/usr/bin/env bash
# E4 GPU generation job (PACE).
#
# Single-job mode (the guide sizes all of E4 to one 3-hour job on 1x L40S):
#   sbatch scripts/slurm/e4_generate.sbatch
#
# Array mode — one (context, checkpoint) pair per task, ~15 min each, useful
# if the smoke test projects > 2.5 h on your GPU:
#   sbatch --array=0-9 scripts/slurm/e4_generate.sbatch
#
#SBATCH --job-name=e4-generate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
# On PACE Phoenix add your account/QOS, e.g.:
##SBATCH --account=gts-<pi-username>
##SBATCH --qos=embers
# If L40S is unavailable, swap the gres line for:  --gres=gpu:A100:1

# NOTE: run 'mkdir -p logs' BEFORE sbatch — SLURM cannot create the output dir itself.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# Activate the environment and FAIL HARD if it isn't right — a silent fallback
# to system python would waste the whole GPU allocation.
ENV_NAME="${E4_CONDA_ENV:-e4}"
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
fi
source config/e4_env.sh
python src/preflight.py --repo "$INTERPLAY_REPO" --models-dir "$MODELS_DIR" --gpu

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  # Map task id -> (context, label): 0-4 = zoo x labels, 5-9 = teachers x labels
  idx="$SLURM_ARRAY_TASK_ID"
  ctx="${E4_CONTEXTS[$(( idx / ${#E4_LABELS[@]} ))]}"
  label="${E4_LABELS[$(( idx % ${#E4_LABELS[@]} ))]}"
  echo "[sbatch] array task $idx -> ctx=$ctx label=$label"
  bash scripts/03_generate.sh "$ctx" "$label"
else
  bash scripts/03_generate.sh
fi
