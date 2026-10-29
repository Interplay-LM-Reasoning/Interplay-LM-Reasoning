#!/usr/bin/env bash
# Build the two eval folders (one per context) by symlinking the op files the
# experiment uses. Each folder becomes the evaluator's --data-root and the
# input to gold-graph building. Both contexts use identical file names, which
# is exactly why they need separate folders.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../config/e4_env.sh"

for ctx in "${E4_CONTEXTS[@]}"; do
  tmpl="$(e4_template "$ctx")"
  dest="$EVAL_DIR/e4_${ctx}"
  mkdir -p "$dest"
  for op in "${E4_OPS[@]}"; do
    src="$DATA_DIR/$tmpl/val/op${op}_200.jsonl"
    [ -s "$src" ] || { echo "missing $src — run 01_download.sh first" >&2; exit 1; }
    ln -sf "$src" "$dest/op${op}_200.jsonl"
  done
  echo "[eval-folder] $dest -> $(ls "$dest" | tr '\n' ' ')"
done
