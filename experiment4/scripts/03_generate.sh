#!/usr/bin/env bash
# GPU step: sample solutions with the authors' vLLM evaluator.
#
# Usage:
#   bash scripts/03_generate.sh                 # all 5 checkpoints x both contexts
#   bash scripts/03_generate.sh zoo             # one context, all checkpoints
#   bash scripts/03_generate.sh zoo strict      # one (context, checkpoint) pair
#
# Output per (context, label):
#   $GEN_DIR/<ctx>/<label>/<ckpt-folder-name>_ood_generations.jsonl  (all samples)
#   $GEN_DIR/<ctx>/<label>/<ckpt-folder-name>_metrics.json           (answer-only pass stats)
# With --id-op-threshold 10, every E4 op (12-20) is OOD, so only *_ood_* files
# appear. The evaluator skips a (ctx, label) whose outputs already exist, so
# re-running after a timeout resumes where it left off.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../config/e4_env.sh"

contexts=("${E4_CONTEXTS[@]}")
labels=("${E4_LABELS[@]}")
[ $# -ge 1 ] && contexts=("$1")
[ $# -ge 2 ] && labels=("$2")

[ -f "$INTERPLAY_REPO/scripts/eval_checkpoints.py" ] || {
  echo "authors' repo not found at $INTERPLAY_REPO — run 00_setup_env.sh" >&2; exit 1; }

failed=()
for ctx in "${contexts[@]}"; do
  data_root="$EVAL_DIR/e4_${ctx}"
  [ -d "$data_root" ] || { echo "missing $data_root — run 02_make_eval_folders.sh" >&2; exit 1; }
  expected=$(( $(cat "$data_root"/*.jsonl | wc -l) * SAMPLE_K ))
  for label in "${labels[@]}"; do
    ckpt="$MODELS_DIR/$(e4_subfolder "$label")"
    out="$GEN_DIR/$ctx/$label"
    mkdir -p "$out"
    # Guard against truncated outputs from a wall-time kill mid-write: the
    # evaluator writes metrics BEFORE generations, so a kill during the
    # generations write leaves a complete-looking pair it would then skip.
    for g in "$out"/*_ood_generations.jsonl; do
      [ -e "$g" ] || continue
      actual=$(wc -l < "$g")
      if [ "$actual" -ne "$expected" ]; then
        echo "[generate] $g has $actual/$expected records (truncated) — deleting pair for clean rerun"
        rm -f "$g" "${g%_ood_generations.jsonl}_metrics.json"
      fi
    done
    echo "=============================================================="
    echo "[generate] ctx=$ctx label=$label"
    echo "           ckpt=$ckpt"
    echo "           out=$out  (k=$SAMPLE_K, max_new_tokens=$MAX_NEW_TOKENS)"
    echo "=============================================================="
    # --checkpoints-pattern keeps its default; find_checkpoints falls back to
    # treating the root itself as the checkpoint because it holds config.json.
    # A single failing pair must not forfeit the GPU time of the others.
    python "$INTERPLAY_REPO/scripts/eval_checkpoints.py" \
      --checkpoints-root "$ckpt" \
      --data-root "$data_root" \
      --id-op-threshold "$ID_OP_THRESHOLD" \
      --gen-backend vllm --skip-loss \
      --sample-k "$SAMPLE_K" \
      --temperature "$TEMPERATURE" --top-p "$TOP_P" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --output-dir "$out" || failed+=("$ctx/$label")
  done
done
if [ "${#failed[@]}" -gt 0 ]; then
  echo "[generate] FAILED pairs (rerun this script to retry just these): ${failed[*]}" >&2
  exit 1
fi
echo "[generate] all requested runs complete"
