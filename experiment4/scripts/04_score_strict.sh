#!/usr/bin/env bash
# CPU step: strict (process-verified) scoring of the generated solutions.
#   1. Build gold dependency graphs from each eval folder (once per context).
#   2. Score every *_generations.jsonl against the gold graphs
#      -> per-sample *.graph_results.jsonl under $GRAPH_DIR/<ctx>/<label>/.
#   3. Aggregate strict pass@k -> results.json in each label directory.
#
# Usage: bash scripts/04_score_strict.sh [ctx]      (default: both contexts)
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../config/e4_env.sh"

contexts=("${E4_CONTEXTS[@]}")
[ $# -ge 1 ] && contexts=("$1")

for ctx in "${contexts[@]}"; do
  eval_folder="$EVAL_DIR/e4_${ctx}"
  gold="$GOLD_DIR/e4_${ctx}"

  # 1) Gold graphs (idempotent: rebuilt only if missing)
  if [ ! -d "$gold" ] || [ -z "$(ls -A "$gold" 2>/dev/null)" ]; then
    echo "[gold] building gold graphs for $ctx"
    python "$INTERPLAY_REPO/scripts/build_gold_dependency_graphs.py" "$eval_folder" "$gold"
  else
    echo "[gold] $gold already built"
  fi

  # 2) Score each generations file explicitly (avoids picking up vLLM .cache
  #    files; *_ood_* only — the *_id_* files are empty with threshold 10).
  #    The scorer parallelizes across FILES, so a single-file invocation is
  #    single-threaded; run the per-label invocations concurrently instead.
  pids=()
  pids_desc=()
  for label in "${E4_LABELS[@]}"; do
    gen_dir="$GEN_DIR/$ctx/$label"
    [ -d "$gen_dir" ] || { echo "[score] skip $ctx/$label (no generations yet)"; continue; }
    found=0
    for gen_file in "$gen_dir"/*_ood_generations.jsonl; do
      [ -e "$gen_file" ] || continue
      found=1
      # The scorer permanently skips an existing graph_results file, even a
      # truncated one from a killed job. Its completion report is written only
      # after a file finishes, so results-without-report = partial: delete.
      base="$(basename "$gen_file")"
      out_dir="$GRAPH_DIR/$ctx/$label"
      graph_file="$out_dir/${base}.graph_results.jsonl"
      report_file="$out_dir/${base}_score_report.jsonl"
      if [ -f "$graph_file" ] && [ ! -f "$report_file" ]; then
        echo "[score] $graph_file has no completion report (partial) — deleting for clean rescore"
        rm -f "$graph_file"
      elif [ -f "$graph_file" ] && [ "$gen_file" -nt "$graph_file" ]; then
        echo "[score] $base is newer than its graph results (regenerated) — deleting stale scores"
        rm -f "$graph_file" "$report_file"
      fi
      mkdir -p "$out_dir"
      echo "[score] $ctx/$label  <- $base  (log: $out_dir/score_log.txt)"
      python "$INTERPLAY_REPO/scripts/score_dependency_graphs.py" \
        "$gold" "$gen_file" \
        --results-dir "$out_dir" --workers 1 \
        > "$out_dir/score_log.txt" 2>&1 &
      pids+=($!)
      pids_desc+=("$ctx/$label/$base")
    done
    [ "$found" -eq 1 ] || echo "[score] skip $ctx/$label (no *_ood_generations.jsonl in $gen_dir)"
  done
  score_failed=0
  if [ "${#pids[@]}" -gt 0 ]; then
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || { echo "[score] FAILED: ${pids_desc[$i]} (see its score_log.txt)" >&2; score_failed=1; }
    done
  fi
  [ "$score_failed" -eq 0 ] || exit 1

  # 3) Aggregate authors' strict pass@k (writes results.json per label dir)
  if [ -d "$GRAPH_DIR/$ctx" ]; then
    echo "[aggregate] $ctx"
    python "$INTERPLAY_REPO/scripts/aggregate_graph_pass_results.py" \
      "$GRAPH_DIR/$ctx" --overwrite \
      --workers "${SLURM_CPUS_PER_TASK:-8}" \
      --pass-at 1 --pass-at 2 --pass-at 4 --pass-at 8 \
      --pass-at 16 --pass-at 32 --pass-at 64
  fi
done
echo "[score] done"
