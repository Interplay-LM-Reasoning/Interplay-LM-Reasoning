"""Qualitative audit: pair generations with graph verdicts, re-score sampled
cases from scratch with the authors' parser, and extract readable evidence."""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRATCH = Path(__file__).resolve().parents[1]
REPO = SCRATCH / "interplay-repo"
sys.path.insert(0, str(REPO))
from utils.solution_dependency_graph import DependencyGraphData, SolutionParser  # noqa

E4RES = Path("/Users/jayitsaha/Documents/COURSES/SEM1/DLT/FINAL_PROJ/experiment4/results")
AUDIT = Path(__file__).parent

GEN_FILE = {
    ("zoo", "base"): "base_ood_generations.jsonl",
    ("zoo", "outcome"): "contextzoo_0.9zoo_0.1teacher_ood_generations.jsonl",
    ("zoo", "strict"): "contextzoo_0.9zoo_0.1teacher_process_strict_ood_generations.jsonl",
    ("teachers", "base"): "base_ood_generations.jsonl",
    ("teachers", "outcome"): "contextzoo_0.9zoo_0.1teacher_ood_generations.jsonl",
    ("teachers", "strict"): "contextzoo_0.9zoo_0.1teacher_process_strict_ood_generations.jsonl",
}
OPS = (12, 20)
parser = SolutionParser()


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


# gold graphs: (ctx, op, line_idx) -> (graph, gold_row)
gold = {}
gold_rows = {}
for ctx in ("zoo", "teachers"):
    for op in OPS:
        graphs = load_jsonl(AUDIT / "gold" / ctx / f"op{op}_200_graphs.jsonl")
        rows = load_jsonl(AUDIT / "eval" / ctx / f"op{op}_200.jsonl")
        for i, (g, r) in enumerate(zip(graphs, rows)):
            gold[(ctx, op, i)] = DependencyGraphData.from_json(g["graph"])
            gold_rows[(ctx, op, i)] = r


def pair_run(ctx, label):
    """Return list of merged records: gen fields + graph verdict fields."""
    gens = load_jsonl(E4RES / "generations" / ctx / label / GEN_FILE[(ctx, label)])
    gname = GEN_FILE[(ctx, label)]
    graphs = load_jsonl(E4RES / "graph_results" / ctx / label / f"{gname}.graph_results.jsonl")
    gmap = {}
    for e in graphs:
        gmap[(int(e["op"]), int(e["index"]), int(e["sample_rank"]))] = e
    counters = defaultdict(int)
    merged = []
    for rec in gens:
        op = int(rec["op"])
        idx = int(rec["__idx"]) % 200
        rank = counters[(op, idx)]
        counters[(op, idx)] += 1
        e = gmap.get((op, idx, rank))
        if e is None:
            continue
        m = dict(rec)
        m.update({"ctx": ctx, "label": label, "idx": idx, "rank": rank,
                  "perfect": e["perfect"], "vm": e["value_mismatch_count"],
                  "dm": e["dependency_mismatch_count"], "am": e["answer_mismatch"],
                  "xn": e["extra_nodes"], "mn": e["missing_nodes"]})
        merged.append(m)
    return merged


def rescore(ctx, op, idx, text):
    """From-scratch verdict using the authors' parser + compare()."""
    g = gold[(ctx, op, idx)]
    parsed = parser.parse(text)
    pred = parser.build_graph(parsed)
    rep = g.compare(pred, value_tolerance=1e-6)
    perfect = not (rep["value_mismatches"] or rep["dependency_mismatches"]
                   or rep["answer_mismatch"] is not None or rep["missing_in_pred"])
    return rep, pred, perfect


def evidence(ctx, op, idx, rec):
    rep, pred, perfect = rescore(ctx, op, idx, rec["gen_solution_answer"])
    text_l = rec["gen_solution_answer"].lower()
    missing_but_mentioned = [n for n in rep["missing_in_pred"] if n.lower() in text_l]
    return {
        "rescored_perfect": perfect,
        "stored_perfect": rec["perfect"],
        "missing_in_pred": rep["missing_in_pred"],
        "missing_but_text_mentions": missing_but_mentioned,
        "extra_in_pred": rep["extra_in_pred"][:6],
        "value_mismatches": [(n, gv, pv) for n, gv, pv in rep["value_mismatches"]][:6],
        "dep_mismatches": rep["dependency_mismatches"][:6],
        "answer_mismatch": rep["answer_mismatch"],
        "pred_nodes": len(pred.nodes), "gold_nodes": len(gold[(ctx, op, idx)].nodes),
    }


def main():
    random.seed(7)
    all_runs = {}
    for (ctx, label) in GEN_FILE:
        all_runs[(ctx, label)] = pair_run(ctx, label)
        print(f"[pair] {ctx}/{label}: {len(all_runs[(ctx,label)])} merged records",
              file=sys.stderr)

    # sanity: rescore agreement on a random 200-record sample across runs
    agree = tot = 0
    for (ctx, label), recs in all_runs.items():
        for rec in random.sample([r for r in recs if int(r["op"]) in OPS], 40):
            ev = evidence(ctx, int(rec["op"]), rec["idx"], rec)
            tot += 1
            agree += (ev["rescored_perfect"] == bool(rec["perfect"]))
    print(f"[sanity] rescore agreement with stored verdicts: {agree}/{tot}", file=sys.stderr)

    # ---- sample hacked: em==1, perfect==False : 3 per (label,ctx,op) = 36
    cases = []
    for (ctx, label), recs in all_runs.items():
        for op in OPS:
            pool = [r for r in recs if int(r["op"]) == op
                    and r["exact_match"] == 1 and not r["perfect"]]
            for rec in random.sample(pool, min(3, len(pool))):
                cases.append(("hacked", rec))
    # ---- sample perfect: 15 total spread
    for (ctx, label), recs in all_runs.items():
        for op in OPS:
            pool = [r for r in recs if int(r["op"]) == op and r["perfect"]]
            if pool and len([c for c in cases if c[0] == "perfect"]) < 16:
                cases.append(("perfect", random.choice(pool)))

    out = []
    for kind, rec in cases:
        ctx, label, op, idx = rec["ctx"], rec["label"], int(rec["op"]), rec["idx"]
        ev = evidence(ctx, op, idx, rec)
        gr = gold_rows[(ctx, op, idx)]
        out.append({
            "kind": kind, "ctx": ctx, "label": label, "op": op, "idx": idx,
            "rank": rec["rank"], "mode": gr.get("mode"),
            "exact_match": rec["exact_match"],
            "stored": {"perfect": rec["perfect"], "vm": rec["vm"], "dm": rec["dm"],
                       "mn": rec["mn"], "xn": rec["xn"], "am": rec["am"]},
            "evidence": ev,
            "gold_answer": rec["gold_answer"], "gen_answer": rec["gen_answer"],
            "gen_text": rec["gen_solution_answer"],
            "gold_solution": gr["solution"],
        })
    (AUDIT / "cases.json").write_text(json.dumps(out, indent=1))

    # compact summary table to stderr
    for i, c in enumerate(out):
        e = c["evidence"]
        print(f"{i:2d} {c['kind']:7s} {c['ctx']:8s} {c['label']:7s} op{c['op']} idx{c['idx']} "
              f"r{c['rank']} mode={c['mode']} | rescored_perfect={e['rescored_perfect']} "
              f"(stored={c['stored']['perfect']}) miss={len(e['missing_in_pred'])} "
              f"missButMentioned={len(e['missing_but_text_mentions'])} "
              f"vm={len(e['value_mismatches'])} dm={len(e['dep_mismatches'])} "
              f"am={'Y' if e['answer_mismatch'] else 'N'} "
              f"nodes={e['pred_nodes']}/{e['gold_nodes']}", file=sys.stderr)

    # ---- aggregate artifact stats over ALL hacked samples (not just sampled):
    # among hacked, how often is a missing gold node actually mentioned in text?
    art = {}
    for (ctx, label), recs in all_runs.items():
        hs = [r for r in recs if r["exact_match"] == 1 and not r["perfect"]
              and int(r["op"]) in OPS]
        sample = random.sample(hs, min(150, len(hs)))
        n_missmention = n_withmiss = 0
        cat = defaultdict(int)
        for r in sample:
            ev = evidence(ctx, int(r["op"]), r["idx"], r)
            if ev["missing_in_pred"]:
                n_withmiss += 1
                if ev["missing_but_text_mentions"]:
                    n_missmention += 1
            key = ("dep" if ev["dep_mismatches"] else "") + \
                  ("val" if ev["value_mismatches"] else "") + \
                  ("miss" if ev["missing_in_pred"] else "") or "other"
            cat[key] += 1
        art[f"{ctx}/{label}"] = {
            "n": len(sample), "with_missing": n_withmiss,
            "missing_but_mentioned": n_missmention,
            "cats": dict(sorted(cat.items(), key=lambda kv: -kv[1])),
        }
    (AUDIT / "artifact_stats.json").write_text(json.dumps(art, indent=1))
    print(json.dumps(art, indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
