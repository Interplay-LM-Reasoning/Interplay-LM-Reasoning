#!/usr/bin/env python3
"""Analyze average problem length for the zero_context dataset.

Scans the directory tree under the given root and looks for `.jsonl` files.
For each JSON object it attempts to extract a human-readable problem/question string
and computes character and word length statistics. Outputs a short report and
optionally writes CSV per-file summaries.

Example:
  python gsm_infinite/analyze_zero_context_lengths.py \
      --root gsm_infinite/gsm-infinite/data/realistic/Igsm/zero_context \
      --output csv_report.csv

"""
import argparse
import json
import os
import glob
import csv
from collections import defaultdict


PREFERRED_KEYS = [
    # "question",
    "problem",
    "solution",
    # "prompt",
    # "text",
    # "input",
    # "query",
    # "question_text",
    # "body",
]


def extract_text(obj):
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return str(obj)
    for k in PREFERRED_KEYS:
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            return obj[k].strip()
    # Sometimes the question is nested under a key like "example" or "data"
    for v in obj.values():
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Fallback: dump the object
    return json.dumps(obj, ensure_ascii=False)


def analyze_file(path, return_examples=False):
    counts = {"n": 0, "chars": 0, "words": 0}
    solution_counts = {"n": 0, "chars": 0, "words": 0}
    example_lengths = [] if return_examples else None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # treat line as plain text
                    txt = line
                    solution_txt = ""
                else:
                    txt = extract_text(obj)
                    # Extract solution separately
                    if isinstance(obj, dict) and "solution" in obj:
                        solution_txt = obj["solution"]
                    else:
                        solution_txt = ""

                l_chars = len(txt)
                l_words = len(txt.split())
                sol_chars = len(solution_txt)
                sol_words = len(solution_txt.split())

                counts["n"] += 1
                counts["chars"] += l_chars
                counts["words"] += l_words
                
                if solution_txt:
                    solution_counts["n"] += 1
                    solution_counts["chars"] += sol_chars
                    solution_counts["words"] += sol_words

                if return_examples:
                    example_lengths.append({
                        "chars": l_chars, "words": l_words, "text": txt,
                        "sol_chars": sol_chars, "sol_words": sol_words, "solution": solution_txt
                    })
    except Exception as e:
        print(f"Failed reading {path}: {e}")
    
    result = {"problem": counts, "solution": solution_counts}
    if return_examples:
        return (result, example_lengths)
    return result

def merge_counts(a, b):
    a["n"] += b["n"]
    a["chars"] += b["chars"]
    a["words"] += b["words"]

def merge_results(a, b):
    merge_counts(a["problem"], b["problem"])
    merge_counts(a["solution"], b["solution"])

def pretty_report_with_solution(agg):
    lines = []
    for key, result in sorted(agg.items()):
        prob_c = result["problem"]
        sol_c = result["solution"]
        prob_n = prob_c["n"]
        sol_n = sol_c["n"]
        
        if prob_n == 0:
            lines.append(f"{key}: no examples")
            continue
            
        prob_avg_chars = prob_c["chars"] / prob_n
        prob_avg_words = prob_c["words"] / prob_n
        
        if sol_n > 0:
            sol_avg_chars = sol_c["chars"] / sol_n
            sol_avg_words = sol_c["words"] / sol_n
            lines.append(f"{key}: examples={prob_n}, problem_avg_chars={prob_avg_chars:.1f}, problem_avg_words={prob_avg_words:.1f}, solution_avg_chars={sol_avg_chars:.1f}, solution_avg_words={sol_avg_words:.1f}")
        else:
            lines.append(f"{key}: examples={prob_n}, problem_avg_chars={prob_avg_chars:.1f}, problem_avg_words={prob_avg_words:.1f}, no solutions")
    return "\n".join(lines)

def analyze_groups(root, pattern):
    files = sorted(glob.glob(os.path.join(root, pattern), recursive=True))
    per_group = defaultdict(lambda: {"problem": {"n": 0, "chars": 0, "words": 0}, "solution": {"n": 0, "chars": 0, "words": 0}})
    for p in files:
        result = analyze_file(p)
        rel = os.path.relpath(p, root)
        parts = rel.split(os.sep)
        # Use the second part as group (e.g., "hard/2" -> group="2")
        group = parts[1] if len(parts) > 1 else "unknown"
        merge_results(per_group[group], result)
    return per_group

def pretty_group_report_with_solution(per_group, label):
    lines = [f"{label}:"]
    for i in range(2, 21):
        key = str(i)
        result = per_group.get(key, {"problem": {"n": 0, "chars": 0, "words": 0}, "solution": {"n": 0, "chars": 0, "words": 0}})
        prob_c = result["problem"]
        sol_c = result["solution"]
        prob_n = prob_c["n"]
        sol_n = sol_c["n"]
        
        if prob_n == 0:
            lines.append(f"  {key}: no examples")
            continue
            
        prob_avg_chars = prob_c["chars"] / prob_n
        prob_avg_words = prob_c["words"] / prob_n
        
        if sol_n > 0:
            sol_avg_chars = sol_c["chars"] / sol_n
            sol_avg_words = sol_c["words"] / sol_n
            lines.append(f"  {key}: examples={prob_n}, prob={prob_avg_chars:.0f}c|{prob_avg_words:.0f}w, sol={sol_avg_chars:.0f}c|{sol_avg_words:.0f}w")
        else:
            lines.append(f"  {key}: examples={prob_n}, prob={prob_avg_chars:.0f}c|{prob_avg_words:.0f}w, no solutions")
    return "\n".join(lines)

def format_table_with_solution(datasets_results, title):
    """Format results from multiple datasets into a table with problem and solution stats."""
    lines = [f"\n===== {title} ====="]
    
    # Header
    header = f"{'Group':<6} {'Zero_Ctx':<25} {'8k':<25} {'16k':<25} {'32k':<25}"
    lines.append(header)
    lines.append("-" * len(header))
    
    # Data rows for groups 2-20
    for i in range(2, 21):
        key = str(i)
        row_parts = [f"{key:<6}"]
        
        for dataset in ['zero_context', '8k', '16k', '32k']:
            if dataset in datasets_results:
                result = datasets_results[dataset].get(key, {"problem": {"n": 0, "chars": 0, "words": 0}, "solution": {"n": 0, "chars": 0, "words": 0}})
                prob_c = result["problem"]
                sol_c = result["solution"]
                prob_n = prob_c["n"]
                sol_n = sol_c["n"]
                
                if prob_n == 0:
                    cell = "no data"
                else:
                    prob_avg_chars = prob_c["chars"] / prob_n
                    prob_avg_words = prob_c["words"] / prob_n
                    if sol_n > 0:
                        sol_avg_chars = sol_c["chars"] / sol_n
                        sol_avg_words = sol_c["words"] / sol_n
                        cell = f"{prob_n:>2}|{prob_avg_chars:>3.0f}|{prob_avg_words:>2.0f}|{sol_avg_chars:>3.0f}|{sol_avg_words:>2.0f}"
                    else:
                        cell = f"{prob_n:>2}|{prob_avg_chars:>3.0f}|{prob_avg_words:>2.0f}|no_sol"
                row_parts.append(f"{cell:<25}")
            else:
                row_parts.append(f"{'---':<25}")
        
        lines.append("".join(row_parts))
    
    # Footer explaining format
    lines.append("")
    lines.append("Format: n|prob_chars|prob_words|sol_chars|sol_words")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="gsm_infinite/gsm-infinite/data/realistic/Igsm/zero_context",
                        help="Root directory for zero_context (default: gsm_infinite/gsm-infinite/data/realistic/Igsm/zero_context)")
    parser.add_argument("--output", default=None, help="Optional CSV output path for per-file stats")
    parser.add_argument("--pattern", default="**/*.jsonl", help="Glob pattern for files to analyze")
    parser.add_argument("--detailed", action="store_true", help="Print per-file average lengths grouped by top-level dir")
    parser.add_argument("--per-example", action="store_true", help="Also print per-example lengths for each file (can be large)")
    args = parser.parse_args()

    root = args.root
    glob_pattern = os.path.join(root, args.pattern)
    files = sorted(glob.glob(glob_pattern, recursive=True))
    per_file_rows = []  # Initialize the list for per-file statistics
    if not files:
        print(f"No files found under {root} matching pattern {args.pattern}")
        return

    per_group = analyze_groups(root, args.pattern)
    overall_problem = {"n": 0, "chars": 0, "words": 0}
    overall_solution = {"n": 0, "chars": 0, "words": 0}
    for result in per_group.values():
        merge_counts(overall_problem, result["problem"])
        merge_counts(overall_solution, result["solution"])
    overall = {"problem": overall_problem, "solution": overall_solution}

    print("===== zero_context length analysis =====")
    print(pretty_report_with_solution({"overall": overall}))
    print("")
    print(pretty_group_report_with_solution(per_group, "zero_context groups 2-20"))

    # Analyze 8k, 16k, 32k datasets
    other_roots = {
        "8k": "gsm_infinite/gsm-infinite/data/realistic/Igsm/8k",
        "16k": "gsm_infinite/gsm-infinite/data/realistic/Igsm/16k",
        "32k": "gsm_infinite/gsm-infinite/data/realistic/Igsm/32k",
    }
    
    all_datasets_results = {"zero_context": per_group}
    
    for label, r in other_roots.items():
        per_group_other = analyze_groups(r, args.pattern)
        all_datasets_results[label] = per_group_other
        print(pretty_group_report_with_solution(per_group_other, f"{label} groups 2-20"))

    # Print summary table
    print(format_table_with_solution(all_datasets_results, "Summary Table by Group and Dataset"))
    if args.output:
        try:
            # Populate per_file_rows with data if it's empty
            if not per_file_rows:
                for file_path in files:
                    counts = analyze_file(file_path)
                    if counts["n"] > 0:
                        per_file_rows.append({
                            "file": os.path.relpath(file_path, root),
                            "n": counts["n"],
                            "avg_chars": counts["chars"] / counts["n"],
                            "avg_words": counts["words"] / counts["n"]
                        })
                        
            with open(args.output, "w", newline="", encoding="utf-8") as csvf:
                writer = csv.DictWriter(csvf, fieldnames=["file", "n", "avg_chars", "avg_words"])
                writer.writeheader()
                for r in per_file_rows:
                    writer.writerow(r)
            print(f"Wrote per-file CSV to {args.output}")
            print(f"Wrote per-file CSV to {args.output}")
        except Exception as e:
            print(f"Failed to write CSV: {e}")


if __name__ == "__main__":
    main()
