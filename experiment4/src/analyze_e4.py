#!/usr/bin/env python3
"""E4 analysis: strict vs answer-only pass@k, reward-hacking rate, error breakdown.

Inputs (produced by scripts/03_generate.sh and scripts/04_score_strict.sh):
  <gen-dir>/<ctx>/<label>/*_generations.jsonl     per-sample answer-only results
  <graph-dir>/<ctx>/<label>/*.graph_results.jsonl per-sample strict (process) results

Joining convention: a generation record carries the global row index `__idx`
(0..N-1 across the context's eval rows); the graph scorer stores the same index
normalized per op (`__idx % 200`). We therefore key every prompt by
(op, __idx % 200) on both sides. Samples the graph scorer could not parse get
no graph entry at all, so strict metrics treat them as failures and the
reward-hacking join counts them as "right answer through wrong/unparseable
steps" when their answer was correct.

Outputs under --out:
  per_op_passk.csv     strict + answer-only pass@k per (ctx, label, op) w/ bootstrap CIs
  passk_curves.csv     strict pass@k, k=1..64, per (ctx, label, scope)
  hacking_rate.csv     reward-hacking rate per (ctx, label, scope) w/ bootstrap CIs
  error_breakdown.csv  per-sample error-category shares per (ctx, label, band)
  summary.json         headline numbers + checks against the paper's predictions
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e4_registry import BANDS, BY_LABEL, KS, LABELS, N_PROBLEMS_PER_OP, OPS  # noqa: E402

CONTEXT_KEYS = ["zoo", "teachers"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path):
    bad = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad += 1  # e.g. half-written final line from a killed job
    if bad:
        print(f"[analyze] WARNING {path.name}: skipped {bad} unparseable line(s) "
              f"(truncated write?) — consider re-running that pipeline stage")


def load_generations(gen_dir: Path) -> dict:
    """Per-prompt answer-only sample lists.

    Returns {(op:int, idx:int): {"em": [0/1 per sample in order], "resp": [...]}}.
    """
    files = sorted(gen_dir.glob("*_generations.jsonl"))
    files = [f for f in files if ".cache" not in f.parts]
    if not files:
        raise FileNotFoundError(f"no *_generations.jsonl under {gen_dir}")
    prompts: dict = defaultdict(lambda: {"em": [], "resp": []})
    for f in files:
        for rec in iter_jsonl(f):
            op = int(rec["op"])
            idx = int(rec["__idx"]) % N_PROBLEMS_PER_OP
            p = prompts[(op, idx)]
            p["em"].append(int(rec.get("exact_match", 0)))
            p["resp"].append(int(rec.get("resp_tokens", 0)))
    return dict(prompts)


def load_graph_results(graph_dir: Path) -> dict:
    """Per-prompt strict sample entries.

    Returns {(op:int, idx:int): list of entry dicts sorted by sample_rank}.
    """
    files = sorted(graph_dir.glob("*.graph_results.jsonl"))
    if not files:
        raise FileNotFoundError(f"no *.graph_results.jsonl under {graph_dir}")
    prompts: dict = defaultdict(list)
    for f in files:
        for rec in iter_jsonl(f):
            op = int(rec["op"])
            idx = rec.get("index")
            if idx is None:
                idx = rec.get("prompt_key_value")
            idx = int(idx) % N_PROBLEMS_PER_OP
            prompts[(op, idx)].append(rec)
    for entries in prompts.values():
        entries.sort(key=lambda r: r.get("sample_rank", 0))
    return dict(prompts)


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def min_success_rank(successes: list[int]) -> float:
    """0-based rank of the first success; inf if never."""
    for i, s in enumerate(successes):
        if s:
            return float(i)
    return math.inf


def pass_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks < k)) if len(ranks) else float("nan")


def bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int, rng: np.random.Generator,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap over prompts. `values` is per-prompt; stat_fn maps a
    resampled per-prompt array to a scalar."""
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        stats[b] = stat_fn(sample)
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def ratio_of_sums_ci(pairs: np.ndarray, n_boot: int, rng: np.random.Generator,
                     alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap CI for sum(num)/sum(den) over prompts; pairs is (n, 2)."""
    n = len(pairs)
    if n == 0:
        return (float("nan"), float("nan"))
    stats = np.full(n_boot, np.nan)
    for b in range(n_boot):
        s = pairs[rng.integers(0, n, n)]
        den = s[:, 1].sum()
        if den > 0:
            stats[b] = s[:, 0].sum() / den
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Per-(ctx, label) computation
# ---------------------------------------------------------------------------

def analyze_run(gens: dict, graphs: dict) -> dict:
    """Combine both per-sample views into per-prompt records."""
    records = []
    for (op, idx), g in sorted(gens.items()):
        entries = graphs.get((op, idx), [])
        strict_rank = math.inf
        n_perfect = 0
        n_graph_ans_ok = 0
        err = {"dependency_mismatch": 0, "missing_nodes": 0,
               "value_mismatch": 0, "answer_mismatch": 0}
        for e in entries:
            if e.get("perfect"):
                n_perfect += 1
                if e.get("sample_rank", math.inf) < strict_rank:
                    strict_rank = float(e["sample_rank"])
            if not e.get("answer_mismatch", False):
                n_graph_ans_ok += 1
            if e.get("dependency_mismatch_count", 0) > 0:
                err["dependency_mismatch"] += 1
            if e.get("missing_nodes", 0) > 0:
                err["missing_nodes"] += 1
            if e.get("value_mismatch_count", 0) > 0:
                err["value_mismatch"] += 1
            if e.get("answer_mismatch", False):
                err["answer_mismatch"] += 1
        n_samples = len(g["em"])
        n_ans_correct = int(sum(g["em"]))
        records.append({
            "op": op, "idx": idx,
            "n_samples": n_samples,
            "n_scored": len(entries),
            "ans_rank": min_success_rank(g["em"]),
            "strict_rank": strict_rank,
            "n_ans_correct": n_ans_correct,
            "n_perfect": n_perfect,
            "n_graph_ans_ok": n_graph_ans_ok,
            # joined hacking numerator: right answer but not strictly correct
            "n_hacked": max(0, n_ans_correct - n_perfect),
            "resp_tokens_sum": int(sum(g["resp"])),
            **{f"n_{k}": v for k, v in err.items()},
        })
    return {"prompts": pd.DataFrame.from_records(records)}


def scopes_for(df: pd.DataFrame):
    """Yield (scope_name, sub-dataframe) for per-op, per-band and overall."""
    for op in sorted(df["op"].unique()):
        yield f"op{op}", df[df["op"] == op]
    for band, ops in BANDS.items():
        sub = df[df["op"].isin(ops)]
        if len(sub):
            yield band, sub
    yield "overall", df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--gen-dir", type=Path, default=root / "results" / "generations")
    ap.add_argument("--graph-dir", type=Path, default=root / "results" / "graph_results")
    ap.add_argument("--out", type=Path, default=root / "analysis")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    per_op_rows, curve_rows, hack_rows, err_rows = [], [], [], []
    summary: dict = {"runs": {}}

    for ctx in CONTEXT_KEYS:
        for label in LABELS:
            gen_dir = args.gen_dir / ctx / label
            graph_dir = args.graph_dir / ctx / label
            if not gen_dir.is_dir():
                print(f"[analyze] skip {ctx}/{label}: no generations at {gen_dir}")
                continue
            try:
                gens = load_generations(gen_dir)
            except FileNotFoundError as exc:
                print(f"[analyze] skip {ctx}/{label}: {exc}")
                continue
            if not gens:
                print(f"[analyze] skip {ctx}/{label}: generations file(s) empty")
                continue
            graphs = {}
            if graph_dir.is_dir():
                try:
                    graphs = load_graph_results(graph_dir)
                except FileNotFoundError:
                    pass
            if not graphs:
                print(f"[analyze] WARNING {ctx}/{label}: no graph results — "
                      f"strict metrics will be all-zero. Run 04_score_strict.sh.")
            df = analyze_run(gens, graphs)["prompts"]
            if df.empty:
                print(f"[analyze] skip {ctx}/{label}: no usable records")
                continue
            n_prompts = len(df)
            n_samples = int(df["n_samples"].sum())
            # Cap pass@k at the samples actually drawn (a k=4 smoke run must not
            # report columns labeled pass@64).
            max_samples = int(df["n_samples"].max())
            ks_eff = [k for k in KS if k <= max_samples] or [max_samples]
            k_hi = ks_eff[-1]
            print(f"[analyze] {ctx}/{label}: {n_prompts} prompts, {n_samples} samples "
                  f"(k={max_samples}), {int(df['n_scored'].sum())} scored by graph parser")

            run_summary = {"n_prompts": n_prompts, "n_samples": n_samples,
                           "n_samples_per_prompt": max_samples,
                           "avg_resp_tokens": df["resp_tokens_sum"].sum() / max(1, n_samples)}

            for scope, sub in scopes_for(df):
                strict_ranks = sub["strict_rank"].to_numpy(dtype=float)
                ans_ranks = sub["ans_rank"].to_numpy(dtype=float)

                # pass@k curves (strict + answer-only)
                for k in ks_eff:
                    curve_rows.append({
                        "ctx": ctx, "label": label, "scope": scope, "k": k,
                        "strict": pass_at_k(strict_ranks, k),
                        "answer_only": pass_at_k(ans_ranks, k),
                        "n_prompts": len(sub),
                    })

                # headline pass@1 / pass@64 with bootstrap CIs (per-op scopes only
                # feed per_op_passk.csv; bands/overall land in summary via curves)
                if scope.startswith("op"):
                    row = {"ctx": ctx, "label": label, "op": int(scope[2:]),
                           "n_prompts": len(sub), "k_hi": k_hi}
                    for k in dict.fromkeys((1, k_hi)):
                        for name, ranks in (("strict", strict_ranks),
                                            ("answer_only", ans_ranks)):
                            val = pass_at_k(ranks, k)
                            lo, hi = bootstrap_ci(
                                ranks, lambda r, k=k: pass_at_k(r, k),
                                args.bootstrap, rng)
                            row[f"{name}_pass@{k}"] = val
                            row[f"{name}_pass@{k}_lo"] = lo
                            row[f"{name}_pass@{k}_hi"] = hi
                            row[f"gap_pass@{k}"] = (row.get(f"answer_only_pass@{k}", np.nan)
                                                    - row.get(f"strict_pass@{k}", np.nan))
                    per_op_rows.append(row)

                # reward-hacking rate: of samples with the right answer, the share
                # whose steps are wrong (joined = includes unparseable solutions;
                # graph_only = restricted to samples the graph parser could read)
                pairs_joined = sub[["n_hacked", "n_ans_correct"]].to_numpy(dtype=float)
                num_j, den_j = pairs_joined[:, 0].sum(), pairs_joined[:, 1].sum()
                g_hacked = (sub["n_graph_ans_ok"] - sub["n_perfect"]).clip(lower=0)
                pairs_graph = np.stack([g_hacked.to_numpy(dtype=float),
                                        sub["n_graph_ans_ok"].to_numpy(dtype=float)], axis=1)
                num_g, den_g = pairs_graph[:, 0].sum(), pairs_graph[:, 1].sum()
                lo_j, hi_j = ratio_of_sums_ci(pairs_joined, args.bootstrap, rng)
                hack_rows.append({
                    "ctx": ctx, "label": label, "scope": scope,
                    "hacking_rate": num_j / den_j if den_j else float("nan"),
                    "hacking_rate_lo": lo_j, "hacking_rate_hi": hi_j,
                    "hacking_rate_graph_only": num_g / den_g if den_g else float("nan"),
                    "n_answer_correct": int(den_j),
                })

            # error breakdown per band + overall (shares of ALL samples)
            for band, sub in [(b, df[df["op"].isin(ops)]) for b, ops in BANDS.items()] \
                             + [("overall", df)]:
                if not len(sub):
                    continue
                total = int(sub["n_samples"].sum())
                scored = int(sub["n_scored"].sum())
                cats = {
                    "correct": int(sub["n_perfect"].sum()),
                    "dependency_mismatch": int(sub["n_dependency_mismatch"].sum()),
                    "missing_nodes": int(sub["n_missing_nodes"].sum()),
                    "value_mismatch": int(sub["n_value_mismatch"].sum()),
                    "answer_mismatch": int(sub["n_answer_mismatch"].sum()),
                    "unparseable": total - scored,
                }
                for cat, count in cats.items():
                    err_rows.append({
                        "ctx": ctx, "label": label, "band": band, "category": cat,
                        "count": count, "share": count / total if total else float("nan"),
                        "n_samples": total,
                    })

            summary["runs"][f"{ctx}/{label}"] = run_summary

    if not curve_rows:
        sys.exit("[analyze] nothing to analyze — run generation and scoring first")

    per_op = pd.DataFrame(per_op_rows)
    curves = pd.DataFrame(curve_rows)
    hacking = pd.DataFrame(hack_rows)
    errors = pd.DataFrame(err_rows)

    per_op.to_csv(args.out / "per_op_passk.csv", index=False)
    curves.to_csv(args.out / "passk_curves.csv", index=False)
    hacking.to_csv(args.out / "hacking_rate.csv", index=False)
    errors.to_csv(args.out / "error_breakdown.csv", index=False)

    # ----- headline summary & predictions check ----------------------------
    def grab(ctx, label, scope, col):
        m = curves[(curves.ctx == ctx) & (curves.label == label)
                   & (curves.scope == scope) & (curves.k == 1)]
        return float(m[col].iloc[0]) if len(m) else float("nan")

    headline = {}
    for ctx in CONTEXT_KEYS:
        for label in LABELS:
            if f"{ctx}/{label}" not in summary["runs"]:
                continue
            run_curves = curves[(curves.ctx == ctx) & (curves.label == label)]
            kmax = int(run_curves.k.max())
            mk = run_curves[(run_curves.k == kmax) & (run_curves.scope == "overall")]
            hv = hacking[(hacking.ctx == ctx) & (hacking.label == label)
                         & (hacking.scope == "overall")]
            headline[f"{ctx}/{label}"] = {
                "display": BY_LABEL[label].display,
                "kmax": kmax,
                "strict_pass@1_overall": grab(ctx, label, "overall", "strict"),
                "strict_pass@kmax_overall": (float(mk["strict"].iloc[0])
                                             if len(mk) else float("nan")),
                "strict_pass@1_hard": grab(ctx, label, "hard (op 16-20)", "strict"),
                "hacking_rate_overall": (float(hv["hacking_rate"].iloc[0])
                                         if len(hv) else float("nan")),
            }
    summary["headline"] = headline
    summary["paper_predictions"] = [
        "Any process signal raises strict pass@1 by ~4-5 points over outcome-only "
        "on the hardest problems.",
        "0.2/0.8 and strict are best; strict has the lowest hacking rate.",
        "Process rewards shift errors away from wrong dependencies and missing steps.",
    ]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ----- console table ----------------------------------------------------
    print("\n=== E4 headline (overall strict pass@1 / pass@kmax / hacking rate) ===")
    for key, h in headline.items():
        print(f"  {key:22s} {h['display']:32s} "
              f"p@1={100*h['strict_pass@1_overall']:5.1f}%  "
              f"p@{h['kmax']}={100*h['strict_pass@kmax_overall']:5.1f}%  "
              f"hack={100*h['hacking_rate_overall']:5.1f}%")
    print(f"\n[analyze] wrote per_op_passk.csv, passk_curves.csv, hacking_rate.csv, "
          f"error_breakdown.csv, summary.json -> {args.out}")


if __name__ == "__main__":
    main()
