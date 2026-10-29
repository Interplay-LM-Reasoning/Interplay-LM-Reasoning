#!/usr/bin/env bash
# Set up the Python environment and clone the authors' code repo.
# Run once on a login node (needs internet). Safe to re-run.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../config/e4_env.sh"

echo "[setup] scratch:        $SCRATCH"
echo "[setup] interplay repo: $INTERPLAY_REPO"
mkdir -p "$SCRATCH/interplay"

# 1) Clone the authors' repo (evaluation scripts live here; the full verl /
#    LLaMA-Factory installs are for training and are NOT needed for E4).
if [ ! -d "$INTERPLAY_REPO/.git" ]; then
  git clone --depth 1 https://github.com/Interplay-LM-Reasoning/Interplay-LM-Reasoning.git "$INTERPLAY_REPO"
else
  echo "[setup] repo already cloned"
fi

# 2) Python deps. Evaluation only needs vllm + transformers (+ analysis libs).
#    On PACE, load the module stack first, e.g.:
#      module load anaconda3   (then: conda create -n e4 python=3.12 -y; conda activate e4)
python -m pip install --upgrade pip
python -m pip install -r "$E4_ROOT/requirements.txt"

python - <<'EOF'
import torch, transformers
print(f"[setup] torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
print(f"[setup] transformers {transformers.__version__}")
try:
    import vllm
    print(f"[setup] vllm {vllm.__version__}")
except ImportError:
    print("[setup] WARNING: vllm not importable — generation step (03) needs it on the GPU node")
EOF
echo "[setup] done"
