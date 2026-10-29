# E4 — Process-Aware Rewards and Reward Hacking

Evaluation-only replication of **§6 (Figs. 7–8)** of *Zhang, Neubig & Yue (2025), "On the
Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"*
([arXiv 2512.07783](https://arxiv.org/abs/2512.07783)), run in the **context setting**
using the authors' released checkpoints and code. No training is performed.

**Question.** Compared with an outcome-only reward, do mixed or strict process rewards
raise strict pass@k and lower the share of right answers reached through wrong steps?

## Setup under test

All five checkpoints share one base model (context A = zoo at op 2–20, context B =
teachers at 0.1%). The four RL runs used the same data (90% A / 10% B) and differ only
in the reward (paper Eq. 3: `R = α·R_out + (1−α)·R_pv`, plus the strict gate):

| Label | HF subfolder (`Interplay-LM-Reasoning/context_pretrain`, under `idzoo_0.999zoo_0.001teacher/`) | Reward |
|---|---|---|
| `base` | `base` | none (pre-training only) |
| `outcome` | `rl/contextzoo_0.9zoo_0.1teacher` | presumed answer-only (no config in repo) |
| `mix_0.5_0.5` | `rl/contextzoo_0.9zoo_0.1teacher_process0.5` | 0.5 answer + 0.5 step score |
| `mix_0.2_0.8` | `rl/contextzoo_0.9zoo_0.1teacher_process` | 0.2 answer + 0.8 step score |
| `strict` | `rl/contextzoo_0.9zoo_0.1teacher_process_strict` | answer reward only if every step is right |

**Eval data.** `Interplay-LM-Reasoning/context` dataset, `crazy_zootopia/val/` and
`teachers_in_school/val/`, ops 12, 14, 16, 18, 20 (`opN_200.jsonl`, 200 problems each).
Both contexts use identical file names, so each gets its own eval folder.

**Budget.** 5 checkpoints × 10 op files × 200 problems × 64 samples = **640k samples**,
sized for one 3-hour job on 1× L40S/A100 on PACE.

**Sampling.** temperature 0.7, top-p 1.0, max 1,024 new tokens, pass@1…64
(the paper's settings; pass@128 does not fit the 3-hour budget).

## Repo layout

```
experiment4/
├── config/e4_env.sh          # single source of truth for paths + settings
├── requirements.txt
├── scripts/
│   ├── 00_setup_env.sh       # pip install + clone authors' repo        (login node)
│   ├── 01_download.sh        # 5 checkpoints + 10 val files from HF     (login node)
│   ├── 02_make_eval_folders.sh  # symlink op files -> eval/e4_zoo, eval/e4_teachers
│   ├── smoke_test.sh         # 15-min end-to-end check                  (GPU node)
│   ├── 03_generate.sh        # vLLM sampling, 640k solutions            (GPU node)
│   ├── 04_score_strict.sh    # gold graphs + strict scoring + pass@k    (CPU)
│   ├── 05_analyze.sh         # metrics, bootstrap CIs, figures          (CPU)
│   └── slurm/                # PACE sbatch wrappers (single job or array)
├── src/
│   ├── e4_registry.py        # checkpoint registry (sync w/ e4_env.sh)
│   ├── analyze_e4.py         # pass@k + CIs, hacking rate, error breakdown
│   └── plots_e4.py           # all figures
├── results/                  # generations/, graph_results/ (created at runtime)
└── analysis/                 # CSVs, summary.json, PNGs (created by step 05)
```

## How to run (PACE)

**One-shot (recommended):** a single job that does everything on 1× H100/H200 —
see `../PACE_COMMANDS.md` for the full command reference.

```bash
mkdir -p logs
sbatch scripts/run_all_pace.sbatch                    # H100
sbatch --gres=gpu:H200:1 scripts/run_all_pace.sbatch  # H200
```

**Step-by-step alternative:**

```bash
cd experiment4

# 0. Once, on a login node (create/activate a Python 3.12 env first):
bash scripts/00_setup_env.sh

# 1. Downloads go to $SCRATCH (defaults to ~/scratch; override before running):
export SCRATCH=/storage/scratch1/<x>/<user>     # keep off home!
bash scripts/01_download.sh
bash scripts/02_make_eval_folders.sh

# 2. Smoke test on a GPU node BEFORE booking 3-hour jobs (~15 min):
#    (interactive GPU shell, e.g. salloc --gres=gpu:L40S:1 -t1:00:00 ...)
bash scripts/smoke_test.sh

# 3. Generation (GPU, ~2-2.5 h in one job, resumable):
sbatch scripts/slurm/e4_generate.sbatch           # single 3-h job
#   or: sbatch --array=0-9 scripts/slurm/e4_generate.sbatch   # 10 short tasks

# 4. Strict scoring + analysis + figures (CPU):
sbatch scripts/slurm/e4_score.sbatch
#   or locally: bash scripts/04_score_strict.sh && bash scripts/05_analyze.sh
```

Everything is idempotent: generation skips checkpoints whose outputs exist, the
scorer skips already-scored files, gold graphs are built once.

Running elsewhere (any CUDA box): same commands, skip `sbatch` and call
`bash scripts/03_generate.sh` directly.

## Pipeline internals (what talks to what)

1. **Generate** — `scripts/eval_checkpoints.py` (authors') with
   `--gen-backend vllm --skip-loss --sample-k 64 --max-new-tokens 1024
   --id-op-threshold 10`. Each HF checkpoint folder is passed directly as
   `--checkpoints-root` (the script falls back to treating a dir containing
   `config.json` as the checkpoint). Since ops 12–20 all exceed threshold 10,
   outputs are `<ckpt>_ood_generations.jsonl` (one record per sample: `op`,
   `__idx`, `gen_solution_answer`, answer-only `exact_match`, `resp_tokens`)
   plus `<ckpt>_metrics.json` (answer-only pass stats).
2. **Strict scoring** — `build_gold_dependency_graphs.py` turns each eval folder
   into gold DAGs; `score_dependency_graphs.py` parses every generated solution
   into a predicted DAG and compares (per sample: `perfect`,
   `dependency_mismatch_count`, `missing_nodes`, `value_mismatch_count`,
   `answer_mismatch`); `aggregate_graph_pass_results.py` writes the authors'
   strict pass@k (`results.json` per run dir). Pairing is by
   `(op, __idx mod 200)` — the smoke test asserts `missing_gold == 0`.
3. **Analysis** (ours, `src/analyze_e4.py`) — recomputes pass@k from the
   per-sample files so that prompts whose samples all failed to parse count as
   failures (the authors' aggregator drops them), adds **bootstrap 95% CIs over
   problems** (the guide's "call a difference real" requirement), and computes:
   - **Strict and answer-only pass@1/pass@64** per reward, context, and op.
   - **Reward-hacking rate** = among samples with the right answer, the share
     whose steps are wrong. Primary ("joined") definition counts answer-correct
     samples the graph parser could not parse as hacked; a graph-only variant is
     also reported. This is exactly the answer-only vs strict comparison on the
     same generations that the guide asks for.
   - **Error breakdown** (Fig. 8 analogue): shares of samples that are correct /
     have dependency mismatches / missing steps / unparseable, per difficulty
     band (edge = op 12–14, hard = op 16–20).

## What to report (deliverables)

From `analysis/`:

- `per_op_passk.csv`, `passk_curves.csv` → strict pass@1 and pass@64 for each
  reward, per context and per op count (figures `fig_passk_per_op.png`,
  `fig_passk_curves.png`, `fig7_bars_<ctx>.png`).
- `hacking_rate.csv` → reward-hacking rate (fig `fig_hacking_rate.png`).
- `error_breakdown.csv` → Fig. 8-style error shares
  (`fig_error_breakdown_<ctx>.png`).
- `summary.json` → headline table + the paper's predictions to check against.

**The paper predicts:** (1) any process signal raises strict pass@1 by ~4–5
points over outcome-only on the hardest problems; (2) 0.2/0.8 and strict are
best, with strict having the lowest hacking rate; (3) process rewards shift
errors away from wrong dependencies and missing steps.

## Caveats and open questions

- The paper ran §6 on extrapolation (RL on op 11–14); the released
  extrapolation checkpoints lack the reward variants, so **this version tests
  the same question in the context setting** — comparable in direction, not in
  exact numbers.
- `rl/contextzoo_0.9zoo_0.1teacher` has no training config in the code repo;
  treating it as **answer-only ("1.0 Outcome") is an assumption** — check its
  `config.json`/HF README after download and note the answer in the report.
- Strict pass@k uses the scorer's `sample_rank`, which skips unparseable
  samples (authors' convention); our analysis keeps that convention for
  ranks but restores the full prompt denominator.
- Single-run checkpoints: always quote the bootstrap CIs before claiming a
  difference is real.
