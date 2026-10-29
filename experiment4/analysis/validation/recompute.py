"""Clean-room recomputation of E4 metrics from raw per-sample files.

Written independently from the definitions; no code shared with analyze_e4.py.
"""
import json
import glob
import os
from collections import defaultdict

ROOT = "/Users/jayitsaha/Documents/COURSES/SEM1/DLT/FINAL_PROJ/experiment4"
GEN = os.path.join(ROOT, "results", "generations")
GRA = os.path.join(ROOT, "results", "graph_results")
CTXS = ["zoo", "teachers"]
LABELS = ["base", "outcome", "mix_0.5_0.5", "mix_0.2_0.8", "strict"]
OPS = [12, 14, 16, 18, 20]
EDGE, HARD = {12, 14}, {16, 18, 20}
NPER = 200
KS = [1, 64]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def compute_run(ctx, label):
    # --- generations: per-prompt ordered exact_match list -------------------
    gen_files = sorted(glob.glob(os.path.join(GEN, ctx, label, "*_ood_generations.jsonl")))
    assert len(gen_files) == 1, gen_files
    ans = defaultdict(list)          # (op, pid) -> [em, em, ...] in file order
    for rec in read_jsonl(gen_files[0]):
        op = int(rec["op"])
        pid = int(rec["__idx"]) % NPER
        ans[(op, pid)].append(int(rec["exact_match"]))

    # --- graph results: per-prompt entries ----------------------------------
    gra_files = sorted(glob.glob(os.path.join(GRA, ctx, label, "*.graph_results.jsonl")))
    assert len(gra_files) == 1, gra_files
    strict_entries = defaultdict(list)  # (op,pid) -> [(sample_rank, perfect)]
    for rec in read_jsonl(gra_files[0]):
        op = int(rec["op"])
        pid = int(rec["index"])
        strict_entries[(op, pid)].append((int(rec["sample_rank"]), bool(rec["perfect"])))

    # --- per-prompt reductions ----------------------------------------------
    prompts = sorted(ans.keys())
    stats = {}
    for key in prompts:
        ems = ans[key]
        # answer-only first-success index (file order)
        a_rank = next((i for i, e in enumerate(ems) if e), None)
        entries = strict_entries.get(key, [])
        s_ranks = [r for r, p in entries if p]
        s_rank = min(s_ranks) if s_ranks else None
        n_ans = sum(ems)
        n_perf = sum(1 for _, p in entries if p)
        stats[key] = dict(a_rank=a_rank, s_rank=s_rank, n_ans=n_ans, n_perf=n_perf,
                          n_samples=len(ems), n_scored=len(entries))
    return stats


def scope_keys(stats, ops):
    return [k for k in stats if k[0] in ops]


def pass_at(stats, keys, k, which):
    n = len(keys)
    if n == 0:
        return float("nan")
    succ = sum(1 for key in keys
               if stats[key][which] is not None and stats[key][which] < k)
    return succ / n


def hacking(stats, keys):
    num = sum(max(0, stats[k]["n_ans"] - stats[k]["n_perf"]) for k in keys)
    den = sum(stats[k]["n_ans"] for k in keys)
    return num / den if den else float("nan")


def main():
    results = {}
    sanity = {}
    for ctx in CTXS:
        for label in LABELS:
            st = compute_run(ctx, label)
            keys_all = list(st.keys())
            sanity[(ctx, label)] = dict(
                n_prompts=len(keys_all),
                n_samples=sum(v["n_samples"] for v in st.values()),
                n_scored=sum(v["n_scored"] for v in st.values()),
                ops=sorted({k[0] for k in keys_all}),
                per_op_prompts={op: sum(1 for k in keys_all if k[0] == op) for op in OPS},
                max_samples=max(v["n_samples"] for v in st.values()),
                min_samples=min(v["n_samples"] for v in st.values()),
            )
            row = {}
            scopes = {"overall": set(OPS), "edge": EDGE, "hard": HARD}
            for op in OPS:
                scopes[f"op{op}"] = {op}
            for sname, ops in scopes.items():
                keys = scope_keys(st, ops)
                for k in KS:
                    row[f"strict@{k}|{sname}"] = pass_at(st, keys, k, "s_rank")
                    row[f"ans@{k}|{sname}"] = pass_at(st, keys, k, "a_rank")
                row[f"hack|{sname}"] = hacking(st, keys)
            results[(ctx, label)] = row
    out = {"results": {f"{c}/{l}": v for (c, l), v in results.items()},
           "sanity": {f"{c}/{l}": v for (c, l), v in sanity.items()}}
    with open(os.path.join(os.path.dirname(__file__), "cleanroom_metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote cleanroom_metrics.json")
    for k, v in out["sanity"].items():
        print(k, v)


if __name__ == "__main__":
    main()
