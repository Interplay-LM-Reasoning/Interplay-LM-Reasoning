#!/usr/bin/env python3
"""Compute pass@k metrics from graph_results JSONL files.

For every directory beneath a root that contains one or more
``*.graph_results.jsonl`` files, this script aggregates per-operator,
per-template pass@k statistics and writes them to ``results.json``.

The output JSON groups metrics for each source JSONL file separately,
recording pass@k values both per-operator and overall.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
import re
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_PASS_AT: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_TEMPLATE_NAME = "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate pass@k metrics for graph_results JSONL files."
    )
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("graph_results"),
        help="Root directory to scan for *.graph_results.jsonl files.",
    )
    parser.add_argument(
        "--pass-at",
        dest="ks",
        type=int,
        action="append",
        default=None,
        help=(
            "Pass@k values to compute. May be provided multiple times. "
            "Defaults to 1,2,4,8,16,32,84,128."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results.json files instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and log actions without writing any results.json files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (level=INFO instead of WARNING).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=48,
        help="Number of worker processes to use (default: CPU count). Use 1 to disable multiprocessing.",
    )
    return parser.parse_args()


class PromptStats:
    """Track the earliest successful rank for a single prompt."""

    __slots__ = ("min_success_rank", "max_rank")

    def __init__(self) -> None:
        self.min_success_rank: int | None = None
        self.max_rank: int = -1

    def update(self, rank: int | None, perfect: bool) -> None:
        if rank is None:
            return
        if perfect and (self.min_success_rank is None or rank < self.min_success_rank):
            self.min_success_rank = rank
        if rank > self.max_rank:
            self.max_rank = rank


PromptKey = Tuple[str, Union[str, int, float, bool, None]]


def normalise_prompt_value(value: object) -> Union[str, int, float, bool, None]:
    """Convert prompt_key_value into a hashable form."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return repr(value)


class ResultAccumulator:
    """Accumulate prompt-level stats for multiple aggregation views."""

    def __init__(self) -> None:
        self.template_op_prompts: MutableMapping[
            str, MutableMapping[str, MutableMapping[PromptKey, PromptStats]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(PromptStats)))
        self.template_total_prompts: MutableMapping[
            str, MutableMapping[Tuple[str, PromptKey], PromptStats]
        ] = defaultdict(lambda: defaultdict(PromptStats))
        self.op_prompts: MutableMapping[
            str, MutableMapping[Tuple[str, PromptKey], PromptStats]
        ] = defaultdict(lambda: defaultdict(PromptStats))
        self.global_prompts: MutableMapping[
            Tuple[str, str, PromptKey], PromptStats
        ] = defaultdict(PromptStats)
        self.max_rank: int = -1

    def ingest_record(self, record: Mapping[str, object]) -> None:
        op = record.get("op")
        if op is None:
            return
        op_str = str(op)

        template_raw = record.get("template")
        template = (
            str(template_raw) if template_raw not in (None, "") else DEFAULT_TEMPLATE_NAME
        )

        prompt_type = str(record.get("prompt_key_type", "idx"))
        prompt_value = normalise_prompt_value(record.get("prompt_key_value"))
        prompt_key: PromptKey = (prompt_type, prompt_value)

        rank_raw = record.get("sample_rank")
        try:
            rank = int(rank_raw)
        except (TypeError, ValueError):
            rank = None
        perfect = bool(record.get("perfect", False))

        self.template_op_prompts[template][op_str][prompt_key].update(rank, perfect)
        template_total_key = (op_str, prompt_key)
        self.template_total_prompts[template][template_total_key].update(rank, perfect)
        op_total_key = (template, prompt_key)
        self.op_prompts[op_str][op_total_key].update(rank, perfect)
        global_key = (template, op_str, prompt_key)
        self.global_prompts[global_key].update(rank, perfect)

        if isinstance(rank, int) and rank > self.max_rank:
            self.max_rank = rank


def compute_pass_rates(
    prompts: Mapping[object, PromptStats], ks: Sequence[int]
) -> Dict[str, float]:
    total = len(prompts)
    if total == 0:
        return {f"pass@{k}": 0.0 for k in ks}

    successes = {k: 0 for k in ks}
    for stats in prompts.values():
        min_rank = stats.min_success_rank
        if min_rank is None:
            continue
        idx = bisect_right(ks, min_rank)
        for k in ks[idx:]:
            successes[k] += 1

    return {f"pass@{k}": successes[k] / total for k in ks}


def summarise_accumulator(
    acc: ResultAccumulator, ks: Sequence[int]
) -> Tuple[Dict[str, object], Dict[str, object]]:
    per_template: Dict[str, object] = {}
    per_template_counts: Dict[str, object] = {}

    for template, per_op in sorted(acc.template_op_prompts.items()):
        template_entry = {}
        template_columns: MutableMapping[str, Dict[str, float]] = defaultdict(dict)
        template_counts = {}

        for op in sort_op_keys(per_op.keys()):
            prompts = per_op[op]
            pass_rates = compute_pass_rates(prompts, ks)
            template_entry[op] = pass_rates
            for metric_key, value in pass_rates.items():
                template_columns[metric_key][op] = value
            template_counts[op] = len(prompts)

        total_prompts = acc.template_total_prompts.get(template, {})
        template_total_metrics = compute_pass_rates(total_prompts, ks)
        per_template[template] = {
            "per_op_pass_at_k": template_entry,
            "per_op_pass_at_columns": dict(template_columns),
            "total_pass_at_k": template_total_metrics,
        }
        per_template_counts[template] = {
            "per_op": template_counts,
            "total": len(total_prompts),
        }

    total_per_op = {}
    total_columns: MutableMapping[str, Dict[str, float]] = defaultdict(dict)
    total_counts_op = {}
    for op in sort_op_keys(acc.op_prompts.keys()):
        prompts = acc.op_prompts[op]
        pass_rates = compute_pass_rates(prompts, ks)
        total_per_op[op] = pass_rates
        for metric_key, value in pass_rates.items():
            total_columns[metric_key][op] = value
        total_counts_op[op] = len(prompts)

    total_metrics = compute_pass_rates(acc.global_prompts, ks)
    total_counts = len(acc.global_prompts)

    totals = {
        "per_op_pass_at_k": total_per_op,
        "per_op_pass_at_columns": dict(total_columns),
        "total_pass_at_k": total_metrics,
    }
    totals_counts = {
        "per_op": total_counts_op,
        "total": total_counts,
    }

    return (
        {"per_template": per_template, "total": totals},
        {"per_template": per_template_counts, "total": totals_counts},
    )


def discover_run_directories(root: Path) -> Dict[Path, List[Path]]:
    run_to_files: Dict[Path, List[Path]] = {}
    for path in root.rglob("*.graph_results.jsonl"):
        run_to_files.setdefault(path.parent, []).append(path)
    return run_to_files


def filter_pass_ks(candidate_ks: Iterable[int], max_rank: int) -> List[int]:
    cleaned = sorted({k for k in candidate_ks if isinstance(k, int) and k > 0})
    if max_rank >= 0:
        cleaned = [k for k in cleaned if k <= max_rank + 1]
        if not cleaned:
            cleaned = [max_rank + 1]
        elif cleaned[-1] != max_rank + 1 and cleaned[-1] < max_rank + 1:
            cleaned.append(max_rank + 1)
    return cleaned


_OP_SORT_RE = re.compile(r"^(\d+)")


def sort_op_keys(keys: Iterable[str]) -> List[str]:
    def key_func(op: str) -> Tuple[int, int, str]:
        if op.isdigit():
            return (0, int(op), op)
        m = _OP_SORT_RE.match(op)
        if m:
            return (1, int(m.group(1)), op)
        return (2, 0, op)

    return sorted((str(op) for op in keys), key=key_func)


_CHECKPOINT_SORT_RE = re.compile(r"checkpoint-(\d+)")


def sort_result_files(files: Iterable[Path]) -> List[Path]:
    """Sort result files so checkpoint-N entries use numeric order.

    This ensures, for example, that ``checkpoint-403`` appears before
    ``checkpoint-13042`` in the generated results.json.
    """

    def key_func(path: Path) -> Tuple[int, int, str]:
        name = path.name
        m = _CHECKPOINT_SORT_RE.search(name)
        if m:
            return (0, int(m.group(1)), name)
        # Non-checkpoint files are sorted after checkpoint files,
        # falling back to lexicographic ordering by filename.
        return (1, 0, name)

    return sorted(files, key=key_func)


def process_file(path: Path, ks: Sequence[int]) -> Tuple[Dict[str, object], List[int]]:
    acc = ResultAccumulator()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON line in %s", path)
                continue
            if not isinstance(record, dict):
                continue
            acc.ingest_record(record)

    if acc.max_rank < 0:
        return {}, []

    effective_ks = filter_pass_ks(ks, acc.max_rank)
    metrics, counts = summarise_accumulator(acc, effective_ks)
    metrics["ks"] = effective_ks
    metrics["prompt_counts"] = counts
    return metrics, effective_ks


def write_results(run_dir: Path, data: Mapping[str, object], *, overwrite: bool, dry_run: bool) -> None:
    output_path = run_dir / "results.json"
    if output_path.exists() and not overwrite:
        logger.info("Skipping %s (results.json already exists)", run_dir)
        return
    if dry_run:
        logger.info("Dry run: would write %s", output_path)
        return
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    logger.info("Wrote %s", output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    ks = tuple(args.ks) if args.ks else DEFAULT_PASS_AT
    if not ks:
        raise ValueError("No pass@k values specified.")

    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"Root directory {root} does not exist.")

    run_dirs = discover_run_directories(root)
    if not run_dirs:
        logger.warning("No *.graph_results.jsonl files found under %s", root)
        return

    logger.info("Discovered %d run directories under %s", len(run_dirs), root)

    requested_workers = args.workers if args.workers is not None else 0
    if requested_workers <= 0:
        requested_workers = os.cpu_count() or 1

    for run_dir in sorted(run_dirs):
        files = sort_result_files(run_dirs[run_dir])
        logger.info("Processing %s (%d files)", run_dir, len(files))
        run_results: Dict[str, object] = {"files": {}}

        if not files:
            continue

        if requested_workers <= 1 or len(files) == 1:
            for file_path in files:
                metrics, effective_ks = process_file(file_path, ks)
                if not effective_ks:
                    logger.warning("No valid data in %s", file_path)
                    continue
                run_results["files"][file_path.name] = metrics
        else:
            max_workers = min(requested_workers, len(files))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_file, file_path, ks): file_path
                    for file_path in files
                }
                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        metrics, effective_ks = future.result()
                    except Exception:  # pragma: no cover - propagate details via logging
                        logger.exception("Failed to process %s", file_path)
                        continue
                    if not effective_ks:
                        logger.warning("No valid data in %s", file_path)
                        continue
                    run_results["files"][file_path.name] = metrics

        if not run_results["files"]:
            logger.info("No metrics generated for %s; skipping write", run_dir)
            continue

        write_results(run_dir, run_results, overwrite=args.overwrite, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
