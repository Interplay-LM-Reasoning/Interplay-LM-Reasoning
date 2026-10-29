#!/usr/bin/env bash
# E4 CPU job: strict graph scoring + aggregation + analysis + figures.
# Submit after all generation jobs finish:
#   sbatch scripts/slurm/e4_score.sbatch
#
#SBATCH --job-name=e4-score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
##SBATCH --account=gts-<pi-username>

# NOTE: run 'mkdir -p logs' BEFORE sbatch — SLURM cannot create the output dir itself.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# Activate the environment and FAIL HARD if it isn't right.
ENV_NAME="${E4_CONDA_ENV:-e4}"
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
fi
source config/e4_env.sh
python src/preflight.py --repo "$INTERPLAY_REPO" --models-dir "$MODELS_DIR"

bash scripts/04_score_strict.sh
bash scripts/05_analyze.sh
