#!/usr/bin/env bash
# ~15-minute smoke test before booking any 3-hour jobs (per the experiment guide):
# one checkpoint (base), one context (zoo), 2 op files, --sample-k 4,
# all four pipeline steps end to end. Verifies:
#   [1] generation writes *_metrics.json and *_generations.jsonl
#   [2] strict scorer pairs generations to gold problems via __idx (missing_gold == 0)
#   [3] prints samples/sec and projects the full-experiment wall time
# Run on a GPU node AFTER 00_setup_env.sh, 01_download.sh, 02_make_eval_folders.sh.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../config/e4_env.sh"

SMOKE="$E4_ROOT/results/smoke"
rm -rf "$SMOKE"
mkdir -p "$SMOKE/eval" "$SMOKE/gen" "$SMOKE/graph"

# Eval folder with just 2 op files (an easy and a hard one)
for op in 12 20; do
  src="$DATA_DIR/crazy_zootopia/val/op${op}_200.jsonl"
  [ -s "$src" ] || { echo "missing $src — run 01_download.sh first" >&2; exit 1; }
  ln -sf "$src" "$SMOKE/eval/op${op}_200.jsonl"
done

ckpt="$MODELS_DIR/$(e4_subfolder base)"
n_samples=$((2 * 200 * 4))

echo "[smoke] generating: base x {op12, op20} x k=4  ($n_samples samples)"
start=$(date +%s)
python "$INTERPLAY_REPO/scripts/eval_checkpoints.py" \
  --checkpoints-root "$ckpt" \
  --data-root "$SMOKE/eval" \
  --id-op-threshold "$ID_OP_THRESHOLD" \
  --gen-backend vllm --skip-loss \
  --sample-k 4 \
  --temperature "$TEMPERATURE" --top-p "$TOP_P" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output-dir "$SMOKE/gen"
elapsed=$(( $(date +%s) - start ))

echo "[smoke] [1] checking generation outputs..."
ls "$SMOKE/gen"/base_metrics.json "$SMOKE/gen"/base_ood_generations.jsonl
n_lines=$(wc -l < "$SMOKE/gen/base_ood_generations.jsonl")
echo "        $n_lines generation records (expect $n_samples)"

echo "[smoke] building gold graphs + strict scoring..."
python "$INTERPLAY_REPO/scripts/build_gold_dependency_graphs.py" "$SMOKE/eval" "$SMOKE/gold"
python "$INTERPLAY_REPO/scripts/score_dependency_graphs.py" \
  "$SMOKE/gold" "$SMOKE/gen/base_ood_generations.jsonl" \
  --results-dir "$SMOKE/graph" --workers 4 | tee "$SMOKE/score_stdout.json" >/dev/null
python "$INTERPLAY_REPO/scripts/aggregate_graph_pass_results.py" "$SMOKE/graph" --overwrite \
  --pass-at 1 --pass-at 2 --pass-at 4

echo "[smoke] [2] checking scorer pairing (missing_gold must be 0)..."
python - "$SMOKE" "$elapsed" "$n_samples" <<'EOF'
import json, sys
from pathlib import Path
smoke, elapsed, n_samples = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

report = [json.loads(l) for l in open(next(smoke.glob("graph/*_score_report.jsonl")))][-1]
agg = report["aggregate"]
assert agg["total"] == n_samples, f"scored {agg['total']} != {n_samples} samples"
assert agg["missing_gold"] == 0, f"missing_gold={agg['missing_gold']} — pairing broken!"
print(f"        scored={agg['total']} missing_gold=0 parse_fail={agg['parse_fail']} "
      f"perfect={agg['perfect']}")

results = json.load(open(smoke / "graph" / "results.json"))
fname = next(iter(results["files"]))
per_op = results["files"][fname]["total"]["per_op_pass_at_k"]
print(f"        strict pass@k per op: {json.dumps(per_op)}")

rate = n_samples / max(1, elapsed)
full = 5 * 10 * 200 * 64          # 5 ckpts x 10 op files x 200 problems x k=64
hours = full / rate / 3600
print(f"[smoke] [3] {rate:.1f} samples/s -> full E4 ({full:,} samples) ~ {hours:.2f} h")
print("        NOTE: add ~1-2 min model-load overhead per checkpoint x context (10 loads).")
if hours > 2.5:
    print("        WARNING: projected > 2.5 h. Split generation into per-context or "
          "per-checkpoint jobs (03_generate.sh takes ctx/label args) rather than lowering k.")
EOF
echo "[smoke] PASSED"
