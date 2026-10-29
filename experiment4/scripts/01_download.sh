#!/usr/bin/env bash
# Download the 5 E4 checkpoints and the two context val/ splits from Hugging Face.
# Run on a login node; everything lands on $SCRATCH. Safe to re-run (resumes).
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../config/e4_env.sh"

# `hf download` is the current CLI; fall back to the legacy name if absent.
HF_CLI="hf"
command -v hf >/dev/null 2>&1 || HF_CLI="huggingface-cli"
echo "[download] using CLI: $HF_CLI"

# --- Checkpoints (~1.25 GB total: base ~413 MB fp32, 4 RL ~210 MB bf16 each) ---
for label in "${E4_LABELS[@]}"; do
  sub="$(e4_subfolder "$label")"
  echo "[download] checkpoint '$label'  <-  $HF_MODEL_REPO/$sub"
  "$HF_CLI" download "$HF_MODEL_REPO" \
    --include "$sub/*" \
    --local-dir "$MODELS_DIR"
done

# --- Eval data: only the op files E4 uses, for both contexts ------------------
for ctx in "${E4_CONTEXTS[@]}"; do
  tmpl="$(e4_template "$ctx")"
  for op in "${E4_OPS[@]}"; do
    echo "[download] data $tmpl/val/op${op}_200.jsonl"
    "$HF_CLI" download "$HF_DATA_REPO" --repo-type dataset \
      --include "$tmpl/val/op${op}_200.jsonl" \
      --local-dir "$DATA_DIR"
  done
done

echo "[download] verifying..."
missing=0
for label in "${E4_LABELS[@]}"; do
  sub="$(e4_subfolder "$label")"
  for f in config.json model.safetensors tokenizer.json; do
    [ -f "$MODELS_DIR/$sub/$f" ] || { echo "  MISSING $MODELS_DIR/$sub/$f"; missing=1; }
  done
done
for ctx in "${E4_CONTEXTS[@]}"; do
  tmpl="$(e4_template "$ctx")"
  for op in "${E4_OPS[@]}"; do
    f="$DATA_DIR/$tmpl/val/op${op}_200.jsonl"
    [ -s "$f" ] || { echo "  MISSING $f"; missing=1; }
  done
done
[ "$missing" -eq 0 ] && echo "[download] all files present" || { echo "[download] FAILED verification"; exit 1; }
