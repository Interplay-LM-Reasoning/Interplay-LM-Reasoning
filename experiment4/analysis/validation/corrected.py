"""Artifact-corrected strict metrics: strip the equation tail ('We know ...')
before parsing BOTH gold and generated solutions, killing the phantom-letter
dependency leak. Recompute hacking rate + strict pass@1 on ops 12/20."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRATCH / "interplay-repo"))
from utils.solution_dependency_graph import SolutionParser  # noqa

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
TAIL_RE = re.compile(r"We know\b.*$", re.DOTALL)
parser = SolutionParser()


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def core_graph(text):
    """Parse with the equation tail removed; return {name: (deps, value)}."""
    parsed = parser.parse(TAIL_RE.sub("", text))
    g = parser.build_graph(parsed)
    return {n: (frozenset(i.dependencies), i.value) for n, i in g.nodes.items()}


def core_perfect(goldg, predg, tol=1e-6):
    for name, (gd, gv) in goldg.items():
        if name not in predg:
            return False
        pd, pv = predg[name]
        if gd != pd:
            return False
        if gv is None:
            continue  # gold unknown value (reverse-mode unknowns resolved in tail)
        if pv is None or abs(gv - pv) > tol:
            return False
    return True


# sanitized gold graphs
gold = {}
for ctx in ("zoo", "teachers"):
    for op in OPS:
        for i, r in enumerate(load_jsonl(AUDIT / "eval" / ctx / f"op{op}_200.jsonl")):
            gold[(ctx, op, i)] = core_graph(r["solution"])

rows = []
for (ctx, label), gname in GEN_FILE.items():
    gens = load_jsonl(E4RES / "generations" / ctx / label / gname)
    graphs = load_jsonl(E4RES / "graph_results" / ctx / label / f"{gname}.graph_results.jsonl")
    gmap = {(int(e["op"]), int(e["index"]), int(e["sample_rank"])): e for e in graphs}
    counters = defaultdict(int)
    stats = defaultdict(int)
    p1 = {"orig": defaultdict(lambda: [0, 0]), "corr": defaultdict(lambda: [0, 0])}
    for rec in gens:
        op = int(rec["op"])
        if op not in OPS:
            continue
        idx = int(rec["__idx"]) % 200
        rank = counters[(op, idx)]
        counters[(op, idx)] += 1
        e = gmap[(op, idx, rank)]
        em = rec["exact_match"] == 1
        stored_perfect = bool(e["perfect"])
        cp = core_perfect(gold[(ctx, op, idx)], core_graph(rec["gen_solution_answer"]))
        corr_strict = cp and em
        stats["n"] += 1
        stats["em"] += em
        stats["stored_perfect"] += stored_perfect
        stats["corr_strict"] += corr_strict
        if em and not stored_perfect:
            stats["hacked_orig"] += 1
            if corr_strict:
                stats["hacked_pure_artifact"] += 1
        if em and not corr_strict:
            stats["hacked_corr"] += 1
        if rank == 0:
            p1["orig"][(op, idx)][0] += stored_perfect
            p1["corr"][(op, idx)][0] += corr_strict
    n_prompts = len(counters)
    row = {"run": f"{ctx}/{label}", **stats,
           "hack_rate_orig": stats["hacked_orig"] / max(1, stats["em"]),
           "hack_rate_corr": stats["hacked_corr"] / max(1, stats["em"]),
           "artifact_share_of_hacked": stats["hacked_pure_artifact"] / max(1, stats["hacked_orig"]),
           "strict_p1_orig": sum(v[0] for v in p1["orig"].values()) / n_prompts,
           "strict_p1_corr": sum(v[0] for v in p1["corr"].values()) / n_prompts}
    rows.append(row)
    print(json.dumps(row), flush=True)

(AUDIT / "corrected_metrics.json").write_text(json.dumps(rows, indent=1))
