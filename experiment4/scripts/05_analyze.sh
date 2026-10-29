#!/usr/bin/env bash
# Final step: compute E4 metrics (strict/answer-only pass@k + bootstrap CIs,
# reward-hacking rate, error breakdown) and render all figures.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../config/e4_env.sh"

python "$E4_ROOT/src/analyze_e4.py" \
  --gen-dir "$GEN_DIR" --graph-dir "$GRAPH_DIR" --out "$ANALYSIS_DIR" "$@"
python "$E4_ROOT/src/plots_e4.py" --analysis-dir "$ANALYSIS_DIR"

echo "[analyze] outputs:"
ls -1 "$ANALYSIS_DIR"
