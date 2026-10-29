"""Paired bootstrap validity check for E4 claims (independent reimplementation)."""
import json
import math
from pathlib import Path

import numpy as np

RES = Path("/Users/jayitsaha/Documents/COURSES/SEM1/DLT/FINAL_PROJ/experiment4/results")
LABELS = ["base", "outcome", "mix_0.5_0.5", "mix_0.2_0.8", "strict"]
CTXS = ["zoo", "teachers"]
N_PER_OP = 200
B = 5000
rng = np.random.default_rng(12345)


def iter_jsonl(p):
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_run(ctx, label):
    """Per-problem stats keyed (op:int, idx:int)."""
    gdir = RES / "graph_results" / ctx / label
    gen_dir = RES / "generations" / ctx / label
    graph_file = next(gdir.glob("*_ood_generations.jsonl.graph_results.jsonl"))
    gen_file = next(gen_dir.glob("*_ood_generations.jsonl"))

    prob = {}
    for r in iter_jsonl(graph_file):
        key = (int(r["op"]), int(r["index"]) % N_PER_OP)
        d = prob.setdefault(key, {"strict_rank": math.inf, "n_perfect": 0,
                                  "n_ans_ok_graph": 0, "n_ans_correct": 0,
                                  "ans_rank": math.inf, "n_samples": 0})
        if r["perfect"]:
            d["n_perfect"] += 1
            if r["sample_rank"] < d["strict_rank"]:
                d["strict_rank"] = float(r["sample_rank"])
        if not r["answer_mismatch"]:
            d["n_ans_ok_graph"] += 1
    counters = {}
    for r in iter_jsonl(gen_file):
        key = (int(r["op"]), int(r["__idx"]) % N_PER_OP)
        d = prob[key]
        rank = counters.get(key, 0)
        counters[key] = rank + 1
        d["n_samples"] += 1
        if int(r.get("exact_match", 0)):
            d["n_ans_correct"] += 1
            if rank < d["ans_rank"]:
                d["ans_rank"] = float(rank)
    return prob


print("loading runs...")
runs = {(ctx, lab): load_run(ctx, lab) for ctx in CTXS for lab in LABELS}
for (ctx, lab), p in runs.items():
    assert len(p) == 1000, (ctx, lab, len(p))
    assert all(v["n_samples"] == 64 for v in p.values()), (ctx, lab)
print("loaded; all runs 1000 problems x 64 samples")

# aligned problem key list per ctx (same order across labels)
keys = {ctx: sorted(runs[(ctx, "base")].keys()) for ctx in CTXS}
for ctx in CTXS:
    for lab in LABELS:
        assert sorted(runs[(ctx, lab)].keys()) == keys[ctx]


def arrays(ctx, lab, ops):
    ks = [k for k in keys[ctx] if k[0] in ops]
    r = runs[(ctx, lab)]
    strict_rank = np.array([r[k]["strict_rank"] for k in ks])
    n_perfect = np.array([r[k]["n_perfect"] for k in ks], dtype=float)
    n_ans = np.array([r[k]["n_ans_correct"] for k in ks], dtype=float)
    return strict_rank, n_perfect, n_ans


def paired_ci(stat_a, stat_b, n, alphas=(0.05, 0.0056)):
    """stat_* : fn(index_array)->float. Returns point diff + CIs (a-b)."""
    point = stat_a(np.arange(n)) - stat_b(np.arange(n))
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        diffs[b] = stat_a(idx) - stat_b(idx)
    out = {"point": point}
    for al in alphas:
        lo, hi = np.percentile(diffs, [100 * al / 2, 100 * (1 - al / 2)])
        out[f"ci{al}"] = (lo, hi)
    # bootstrap "p-value" (two-sided, proportion crossing zero)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    out["p_boot"] = max(p, 1 / B)
    return out


def passk_stat(strict_rank, k):
    succ = (strict_rank < k).astype(float)
    return lambda idx: succ[idx].mean()


def hack_stat(n_perfect, n_ans):
    hacked = np.maximum(0, n_ans - n_perfect)
    return lambda idx: hacked[idx].sum() / max(n_ans[idx].sum(), 1e-9)


results = []

HARD = {16, 18, 20}
EDGE = {12, 14}
ALL = {12, 14, 16, 18, 20}

# 1a. teachers hard strict pass@1: process - outcome ; and outcome - base
sr_out, _, _ = arrays("teachers", "outcome", HARD)
n_hard = len(sr_out)
for lab in ["mix_0.5_0.5", "mix_0.2_0.8", "strict"]:
    sr, _, _ = arrays("teachers", lab, HARD)
    r = paired_ci(passk_stat(sr, 1), passk_stat(sr_out, 1), n_hard)
    results.append(("teachers hard p@1", f"{lab} - outcome", r))
sr_base, _, _ = arrays("teachers", "base", HARD)
results.append(("teachers hard p@1", "outcome - base",
                paired_ci(passk_stat(sr_out, 1), passk_stat(sr_base, 1), n_hard)))

# teachers EDGE p@1 mix_0.5 - outcome (the +6.4 claim)
sr_oe, _, _ = arrays("teachers", "outcome", EDGE)
sr_me, _, _ = arrays("teachers", "mix_0.5_0.5", EDGE)
results.append(("teachers edge p@1", "mix_0.5_0.5 - outcome",
                paired_ci(passk_stat(sr_me, 1), passk_stat(sr_oe, 1), len(sr_oe))))

# 1b. teachers overall hacking rate: process - outcome ; outcome - base
_, np_out, na_out = arrays("teachers", "outcome", ALL)
n_all = len(np_out)
for lab in ["mix_0.5_0.5", "mix_0.2_0.8", "strict", "base"]:
    _, npf, na = arrays("teachers", lab, ALL)
    r = paired_ci(hack_stat(npf, na), hack_stat(np_out, na_out), n_all)
    results.append(("teachers overall hack", f"{lab} - outcome", r))

# 1c. zoo edge pass@64: base - outcome
sr_zb, _, _ = arrays("zoo", "base", EDGE)
sr_zo, _, _ = arrays("zoo", "outcome", EDGE)
results.append(("zoo edge p@64", "base - outcome",
                paired_ci(passk_stat(sr_zb, 64), passk_stat(sr_zo, 64), len(sr_zb))))
# zoo FULL pass@64 base - outcome too
sr_zb2, _, _ = arrays("zoo", "base", ALL)
sr_zo2, _, _ = arrays("zoo", "outcome", ALL)
results.append(("zoo overall p@64", "base - outcome",
                paired_ci(passk_stat(sr_zb2, 64), passk_stat(sr_zo2, 64), n_all)))

# teachers hard pass@64: outcome - base (the big RL claim)
results.append(("teachers hard p@64", "outcome - base",
                paired_ci(passk_stat(sr_out, 64), passk_stat(sr_base, 64), n_hard)))

print(f"\n{'scope':24s} {'comparison':28s} {'diff':>7s}  {'95% CI':>18s}  {'99.44% CI':>18s}  {'p_boot':>7s}  verdict")
for scope, comp, r in results:
    lo, hi = r["ci0.05"]
    lo2, hi2 = r["ci0.0056"]
    verdict = "EXCLUDES 0" if lo > 0 or hi < 0 else "overlaps 0"
    v2 = "+bonf" if (lo2 > 0 or hi2 < 0) else ""
    print(f"{scope:24s} {comp:28s} {100*r['point']:+7.2f}  [{100*lo:+6.2f},{100*hi:+6.2f}]  "
          f"[{100*lo2:+6.2f},{100*hi2:+6.2f}]  {r['p_boot']:7.4f}  {verdict} {v2}")

# 2. sign test: teachers per-op strict p@1, process vs outcome (15 comparisons)
from itertools import product
wins = ties = 0
for lab, op in product(["mix_0.5_0.5", "mix_0.2_0.8", "strict"], sorted(ALL)):
    sr, _, _ = arrays("teachers", lab, {op})
    sro, _, _ = arrays("teachers", "outcome", {op})
    d = (sr < 1).mean() - (sro < 1).mean()
    wins += d > 0
    ties += d == 0
print(f"\nsign test raw: wins={wins} ties={ties} losses={15-wins-ties} of 15 (correlated! see caveat)")
# exact binomial on 5 ops using mean-across-variants sign per op (more honest unit)
op_signs = []
for op in sorted(ALL):
    ds = []
    for lab in ["mix_0.5_0.5", "mix_0.2_0.8", "strict"]:
        sr, _, _ = arrays("teachers", lab, {op})
        sro, _, _ = arrays("teachers", "outcome", {op})
        ds.append((sr < 1).mean() - (sro < 1).mean())
    op_signs.append(np.mean(ds) > 0)
k = sum(op_signs)
p_sign = sum(math.comb(5, i) for i in range(k, 6)) / 2**5
print(f"per-op (averaged over variants) wins: {k}/5, one-sided exact sign test p={p_sign:.4f}")

# 4. ratio-estimator bias check for hacking rate (jackknife bias estimate, teachers overall, strict)
_, npf, na = arrays("teachers", "strict", ALL)
full = hack_stat(npf, na)(np.arange(n_all))
jk = np.array([hack_stat(npf, na)(np.delete(np.arange(n_all), i)) for i in range(n_all)])
bias = (n_all - 1) * (jk.mean() - full)
print(f"\nhacking-rate ratio estimator (teachers/strict): point={100*full:.2f}%, jackknife bias={100*bias:+.4f} pp")
