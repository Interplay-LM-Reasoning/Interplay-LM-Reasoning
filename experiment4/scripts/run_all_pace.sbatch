#!/usr/bin/env bash
# ============================================================================
# E4 — EVERYTHING IN ONE JOB (PACE, 1x H100 or H200)
#
# Runs end to end: env bootstrap -> clone authors' repo -> HF downloads ->
# eval folders -> smoke test -> full generation (640k samples) -> strict
# scoring -> analysis + figures. Every phase is idempotent, so if the job
# dies, just resubmit and it resumes where it stopped.
#
# Submit from the experiment4/ directory:
#   sbatch scripts/run_all_pace.sbatch                      # H100 (header default)
#   sbatch --gres=gpu:H200:1 scripts/run_all_pace.sbatch    # H200 instead
#
# Phoenix (instead of ICE) additionally needs your account/QOS:
#   sbatch --account=gts-<pi-username> --qos=embers scripts/run_all_pace.sbatch
#
# Optional env toggles (export before sbatch, they propagate):
#   SCRATCH=/path/to/scratch     where models/data/caches live (default ~/scratch)
#   E4_SKIP_SMOKE=1              skip the ~10-min smoke phase
#   E4_SMOKE_ONLY=1              stop after the smoke test (validate the env on a
#                                cheap/any GPU before booking an H100/H200 slot:
#                                E4_SMOKE_ONLY=1 sbatch --gres=gpu:1 -t1:00:00 scripts/run_all_pace.sbatch)
#   E4_CONDA_ENV=e4              conda env name to create/use
#   HF_TOKEN=hf_...              optional, raises HF rate limits for downloads
# ============================================================================
#SBATCH --job-name=e4-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# NOTE: run 'mkdir -p logs' BEFORE sbatch — SLURM cannot create the output dir itself.

set -euo pipefail

# --- locate experiment4 root -------------------------------------------------
if [ -n "${E4_ROOT:-}" ] && [ -f "${E4_ROOT}/config/e4_env.sh" ]; then
  cd "$E4_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/config/e4_env.sh" ]; then
  cd "$SLURM_SUBMIT_DIR"
elif [ -f "$(dirname "${BASH_SOURCE[0]}")/../config/e4_env.sh" ]; then
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
else
  echo "ERROR: submit this job from the experiment4/ directory (config/e4_env.sh not found)" >&2
  exit 1
fi
mkdir -p logs

phase() { echo; echo "================ [$(date '+%F %T')] $* ================"; }

phase "PHASE 0 — environment"
echo "[info] host=$(hostname)  cwd=$(pwd)"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "[warn] nvidia-smi unavailable (CPU-only phases will still run)"

# Conda env (ICE/Phoenix module stack), with a venv fallback
ENV_NAME="${E4_CONDA_ENV:-e4}"
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${SCRATCH:-$HOME/scratch}/conda_pkgs}"
  # grep without -q: consume all input so SIGPIPE can't trip pipefail
  if ! conda env list | awk '{print $1}' | grep -x "$ENV_NAME" >/dev/null; then
    echo "[env] creating conda env '$ENV_NAME' (python 3.12)"
    conda create -y -n "$ENV_NAME" python=3.12
  fi
  conda activate "$ENV_NAME"
else
  echo "[env] conda not found; using a venv on scratch"
  VENV="${SCRATCH:-$HOME/scratch}/interplay/venv-e4"
  [ -d "$VENV" ] || python3 -m venv "$VENV"
  source "$VENV/bin/activate"
fi
echo "[env] python: $(command -v python) ($(python -V))"

source config/e4_env.sh
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"           # keep HF cache off home quota
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH/pip_cache}"  # ditto for pip
mkdir -p "$SCRATCH/interplay" "$HF_HOME" "$PIP_CACHE_DIR"
echo "[env] SCRATCH=$SCRATCH"
echo "[env] INTERPLAY_REPO=$INTERPLAY_REPO"

# Python deps — gated on a hash of requirements.txt, NOT on "imports work":
# a wrong-but-importable version (e.g. transformers 5.x) must trigger reinstall.
REQ_HASH="$(cksum requirements.txt | awk '{print $1}')"
PIP_MARKER="$SCRATCH/interplay/.e4_pip_ok"
if [ "$(cat "$PIP_MARKER" 2>/dev/null || true)" != "$REQ_HASH" ]; then
  phase "PHASE 0b — pip install (pinned matrix; runs when requirements.txt changes)"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  echo "$REQ_HASH" > "$PIP_MARKER"
fi

# Authors' code repo
if [ ! -d "$INTERPLAY_REPO/.git" ]; then
  phase "PHASE 0c — cloning authors' repo"
  git clone --depth 1 https://github.com/Interplay-LM-Reasoning/Interplay-LM-Reasoning.git "$INTERPLAY_REPO"
fi

phase "PHASE 0d — preflight (fail fast BEFORE spending GPU time)"
if ! python src/preflight.py --repo "$INTERPLAY_REPO" --models-dir "$MODELS_DIR" --gpu; then
  echo "[preflight] forcing reinstall of the pinned matrix and retrying once..."
  python -m pip install --force-reinstall -r requirements.txt
  python src/preflight.py --repo "$INTERPLAY_REPO" --models-dir "$MODELS_DIR" --gpu
  echo "$REQ_HASH" > "$PIP_MARKER"   # only reached if the retry preflight passed
fi

phase "PHASE 1 — downloads (checkpoints + eval data)"
all_present=1
for label in "${E4_LABELS[@]}"; do
  for f in model.safetensors config.json tokenizer.json; do
    [ -f "$MODELS_DIR/$(e4_subfolder "$label")/$f" ] || all_present=0
  done
done
for ctx in "${E4_CONTEXTS[@]}"; do
  tmpl="$(e4_template "$ctx")"
  for op in "${E4_OPS[@]}"; do
    [ -s "$DATA_DIR/$tmpl/val/op${op}_200.jsonl" ] || all_present=0
  done
done
if [ "$all_present" -eq 1 ]; then
  echo "[download] everything already on scratch — skipping"
else
  bash scripts/01_download.sh
fi

phase "PHASE 2 — eval folders"
bash scripts/02_make_eval_folders.sh

if [ "${E4_SKIP_SMOKE:-0}" != "1" ] && [ ! -f results/.smoke_passed ]; then
  phase "PHASE 3 — smoke test (~10 min; set E4_SKIP_SMOKE=1 to skip)"
  bash scripts/smoke_test.sh
  mkdir -p results && touch results/.smoke_passed
else
  phase "PHASE 3 — smoke test skipped (already passed or E4_SKIP_SMOKE=1)"
fi

if [ "${E4_SMOKE_ONLY:-0}" = "1" ]; then
  phase "DONE (E4_SMOKE_ONLY=1 — environment validated, stopping before full generation)"
  exit 0
fi

phase "PHASE 4 — generation: 5 checkpoints x 2 contexts x 64 samples (resumable)"
bash scripts/03_generate.sh

phase "PHASE 5 — strict (process-verified) scoring"
bash scripts/04_score_strict.sh

phase "PHASE 6 — analysis + figures"
bash scripts/05_analyze.sh

phase "DONE"
echo "[done] deliverables in: $ANALYSIS_DIR"
ls -la "$ANALYSIS_DIR"
