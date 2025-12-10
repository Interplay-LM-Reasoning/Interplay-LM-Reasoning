"""Compare generated solutions against gold dependency graphs.

The comparison ignores variable symbol choices (A vs K) and focuses on the
parameter names, their numeric values, and dependency structure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from multiprocessing import Pool, cpu_count
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.solution_dependency_graph import DependencyGraphData, SolutionParser


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)



def append_jsonl(path: Path, payload: dict) -> None:
    """Append a JSON-serializable payload as a single JSONL line."""
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload))
        handle.write('\n')


def discover_generation_files(target: Path) -> List[Path]:
    if target.is_file(): 
        
        return [target]
    if target.is_dir():
        files = sorted(
            p for p in target.rglob('*.jsonl') if p.name.endswith('generations.jsonl')
        )
        return files
    raise FileNotFoundError(f"Generation path {target} not found")


def resolve_result_path(generated_file: Path, args: argparse.Namespace) -> Path | None:
    if getattr(args, 'no_save', False):
        return None
    results_dir = Path(args.results_dir)
    base_root = args.generated_path if args.generated_path.is_dir() else generated_file.parent
    try:
        relative = generated_file.relative_to(base_root)
    except ValueError:
        relative = Path(generated_file.name)
    relative_path = relative if isinstance(relative, Path) else Path(relative)
    result_path = results_dir / relative_path
    return result_path.with_suffix(relative_path.suffix + '.graph_results.jsonl')


def _dry_run_report(generation_files: Iterable[Path], args: argparse.Namespace) -> None:
    skipped_entries: List[dict] = []
    pending_entries: List[dict] = []

    for generated_file in generation_files:
        result_path = resolve_result_path(generated_file, args)
        if result_path is None:
            pending_entries.append({
                'file': str(generated_file),
                'status': 'process',
                'reason': 'no_save enabled; results not persisted',
            })
            continue

        if result_path.exists():
            skipped_entries.append({
                'file': str(generated_file),
                'status': 'skip',
                'results_path': str(result_path),
                'reason': 'existing results detected',
            })
        else:
            pending_entries.append({
                'file': str(generated_file),
                'status': 'process',
                'results_path': str(result_path),
                'reason': 'results missing',
            })

    for entry in skipped_entries:
        print(json.dumps(entry))
    for entry in pending_entries:
        print(json.dumps(entry))

    summary = {
        'dry_run': True,
        'total_files': len(generation_files),
        'skip_count': len(skipped_entries),
        'process_count': len(pending_entries),
    }
    print(json.dumps(summary))


_WORKER_STATE: Dict[str, object] = {}


def load_gold_graphs(directory: Path):
    by_id: Dict[Tuple[str, str], list] = defaultdict(list)
    by_idx: Dict[Tuple[str, int], DependencyGraphData] = {}
    total = 0
    op_counts: Dict[str, int] = defaultdict(int)
    for path in sorted(directory.glob("*.jsonl")):
        for line_index, record in enumerate(load_jsonl(path)):
            graph_payload = record.get("graph")
            if not graph_payload:
                continue
            total += 1
            graph = DependencyGraphData.from_json(graph_payload)
            op = str(record.get("op"))
            op_counts[op] += 1
            id_value = record.get("id")
            if id_value is not None:
                by_id[(op, str(id_value))].append(graph)
            idx_value = record.get("line_index", line_index)
            by_idx[(op, int(idx_value))] = graph
    return by_id, by_idx, dict(op_counts), total


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score generated solutions using gold dependency graphs.")
    parser.add_argument("gold_dir", type=Path, help="Directory containing *_graphs.jsonl files")
    parser.add_argument("generated_path", type=Path, help="JSONL file or directory containing generated solutions")
    parser.add_argument("--solution-key", default="gen_solution_answer", help="Field holding the generated solution text")
    parser.add_argument("--op-field", default="op", help="Field storing the operation identifier")
    parser.add_argument("--id-field", default="id", help="Field storing the example identifier")
    parser.add_argument("--index-field", default="__idx", help="Fallback index field when id is absent")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory to store per-instance comparison results")
    parser.add_argument("--no-save", action="store_true", help="Disable writing per-instance comparison results")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Numeric tolerance for value comparisons")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of parallel worker processes")
    parser.add_argument("--dry-run", action="store_true", help="List files that would be skipped without scoring")
    args = parser.parse_args(list(argv) if argv is not None else None)

    gold_by_id, gold_by_idx, op_counts, total_gold = load_gold_graphs(args.gold_dir)
    if not gold_by_id and not gold_by_idx:
        raise FileNotFoundError(f"No gold graphs found under {args.gold_dir}")

    generation_files = discover_generation_files(args.generated_path)
    if not generation_files:
        raise FileNotFoundError(
            f"No *_generations.jsonl files found under {args.generated_path}"
        )

    aggregate = {
        'files': 0,
        'total': 0,
        'missing_solution': 0,
        'missing_gold': 0,
        'parse_fail': 0,
        'perfect': 0,
        'value_mismatch': 0,
        'dependency_mismatch': 0,
        'answer_mismatch': 0,
        'extra_nodes': 0,
        'missing_nodes': 0,
        'perfect_with_extra': 0,
    }
    aggregate['skipped'] = 0
    overall_used_graphs = set()
    stats_keys = ('total', 'missing_solution', 'missing_gold', 'parse_fail', 'perfect', 'value_mismatch', 'dependency_mismatch', 'answer_mismatch', 'extra_nodes', 'missing_nodes', 'perfect_with_extra')

    args.results_dir = Path(args.results_dir) 
    args.generated_path = Path(args.generated_path)
    if args.dry_run:
        _dry_run_report(generation_files, args)
        return
    args.results_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.workers))
    args.workers = workers

    report_records: List[Dict[str, object]] = []
    template_aggregate: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def update_aggregate(stats, used_keys):
        if stats.get('skipped'):
            aggregate['skipped'] += 1
            return
        aggregate['files'] += 1
        for key in stats_keys:
            aggregate[key] += int(stats.get(key, 0))
        overall_used_graphs.update(used_keys)
        for template_key, template_stats in stats.get('template_stats', {}).items():
            bucket = template_aggregate[template_key]
            for stat_key, value in template_stats.items():
                bucket[stat_key] += int(value)

    if workers == 1:
        args.record_progress = True
        for generated_file in tqdm(generation_files, desc='Scoring files'):
            stats, used_keys = score_file(
                generated_file,
                args,
                gold_by_id,
                gold_by_idx,
                op_counts,
                total_gold,
            )
            print(json.dumps(stats, indent=2))
            report_records.append(dict(stats))
            update_aggregate(stats, used_keys)
    else:
        args.record_progress = False
        worker_args = {
            'solution_key': args.solution_key,
            'op_field': args.op_field,
            'id_field': args.id_field,
            'index_field': args.index_field,
            'results_dir': str(args.results_dir),
            'no_save': args.no_save,
            'tolerance': args.tolerance,
            'record_progress': False,
        }
        with Pool(processes=workers, initializer=_init_worker, initargs=(gold_by_id, gold_by_idx, op_counts, total_gold, worker_args, str(args.generated_path))) as pool:
            for stats, used_list in tqdm(pool.imap_unordered(_score_worker, generation_files), total=len(generation_files), desc='Scoring files'):
                used_keys = {tuple(item) for item in used_list}
                print(json.dumps(stats, indent=2))
                report_records.append(dict(stats))
                update_aggregate(stats, used_keys)

    aggregate['unused_gold_entries'] = max(total_gold - len(overall_used_graphs), 0)
    aggregate['aggregated'] = True
    report_filename = f"{args.generated_path.name}_score_report.jsonl"
    report_path = args.results_dir / report_filename
    legacy_report_path = args.results_dir / f"{args.generated_path.name}_score_report.json"
    if legacy_report_path.exists() and not report_path.exists():
        with legacy_report_path.open('r', encoding='utf-8') as legacy_handle:
            legacy_payload = json.load(legacy_handle)
        append_jsonl(report_path, legacy_payload)
    aggregate['report_path'] = str(report_path)
    aggregate['template_stats'] = {
        template: dict(values) for template, values in template_aggregate.items()
    }

    report_payload = {
        'gold_dir': str(args.gold_dir),
        'generated_path': str(args.generated_path),
        'results_dir': str(args.results_dir),
        'per_file': report_records,
        'aggregate': aggregate,
    }
    append_jsonl(report_path, report_payload)
    print(json.dumps(aggregate, indent=2))


def score_file(
    generated_file: Path,
    args: argparse.Namespace,
    gold_by_id: Dict[Tuple[str, str], List[DependencyGraphData]],
    gold_by_idx: Dict[Tuple[str, int], DependencyGraphData],
    op_counts: Dict[str, int],
    total_gold: int,
) -> Tuple[Dict[str, object], set]:
    solution_parser = SolutionParser()

    id_usage = defaultdict(int)
    sample_counters = defaultdict(int)

    stats = {
        'file': str(generated_file),
        'total': 0,
        'missing_solution': 0,
        'missing_gold': 0,
        'parse_fail': 0,
        'perfect': 0,
        'value_mismatch': 0,
        'dependency_mismatch': 0,
        'answer_mismatch': 0,
        'extra_nodes': 0,
        'missing_nodes': 0,
        "perfect_with_extra": 0
    }
    template_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    result_handle = None
    result_path = resolve_result_path(generated_file, args)
    if result_path is not None and result_path.exists():
        stats['skipped'] = True
        stats['results_path'] = str(result_path)
        return stats, set()

    if result_path is not None:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_handle = result_path.open('w', encoding='utf-8')

    iterator = load_jsonl(generated_file)
    if getattr(args, 'record_progress', True):
        iterator = tqdm(iterator, desc=f"Records: {generated_file.name}", leave=False)
    used_graph_keys = set()
    for record in iterator:
        stats['total'] += 1
        solution_text = record.get(args.solution_key)
        template = record.get("template")
        template_key = template if isinstance(template, str) and template else "__none__"
        template_bucket = template_stats[template_key]
        template_bucket['total'] += 1
        if not isinstance(solution_text, str) or not solution_text.strip():
            stats['missing_solution'] += 1
            template_bucket['missing_solution'] += 1
            continue

        op_value = str(record.get(args.op_field))
        id_value = record.get(args.id_field)
        gold_graph = None
        match_source = None
        normalized_idx = None
        raw_idx_value = None
        if id_value is not None:
            id_key = (op_value, str(id_value))
            if id_key in gold_by_id:
                queue = gold_by_id[id_key]
                pos = id_usage[id_key]
                if queue:
                    slot = pos % len(queue)
                    gold_graph = queue[slot]
                    id_usage[id_key] += 1
                    match_source = 'id'
                    used_graph_keys.add(('id', op_value, str(id_value), slot))
        if gold_graph is None:
            idx_field = record.get(args.index_field)
            if idx_field is not None:
                try:
                    raw_idx = int(idx_field)
                except (TypeError, ValueError):
                    raw_idx = None
                if raw_idx is not None:
                    raw_idx_value = raw_idx
                    size = op_counts.get(op_value)
                    if size:
                        raw_idx %= size
                    normalized_idx = raw_idx
                    idx_key = (op_value, raw_idx)
                    gold_graph = gold_by_idx.get(idx_key)
                    if gold_graph is not None:
                        match_source = 'idx'
                        used_graph_keys.add(('idx', op_value, normalized_idx))
        if gold_graph is None:
            stats['missing_gold'] += 1
            template_bucket['missing_gold'] += 1
            continue

        try:
            parsed = solution_parser.parse(solution_text)
            pred_graph = solution_parser.build_graph(parsed)
        except Exception:
            stats['parse_fail'] += 1
            template_bucket['parse_fail'] += 1
            continue

        report = gold_graph.compare(pred_graph, value_tolerance=args.tolerance)

        has_mismatch = False
        if report['value_mismatches']:
            stats['value_mismatch'] += 1
            template_bucket['value_mismatch'] += 1
            has_mismatch = True
        if report['dependency_mismatches']:
            stats['dependency_mismatch'] += 1
            template_bucket['dependency_mismatch'] += 1
            has_mismatch = True
        if report['answer_mismatch'] is not None:
            stats['answer_mismatch'] += 1
            template_bucket['answer_mismatch'] += 1
            has_mismatch = True
        if report['extra_in_pred']:
            stats['extra_nodes'] += 1
            template_bucket['extra_nodes'] += 1
        if report['missing_in_pred']:
            stats['missing_nodes'] += 1
            template_bucket['missing_nodes'] += 1
            has_mismatch = True

        perfect = not has_mismatch
        if perfect:
            stats['perfect'] += 1
            template_bucket['perfect'] += 1
            if report['extra_in_pred']:
                stats['perfect_with_extra'] += 1
                template_bucket['perfect_with_extra'] += 1
        if result_handle is not None:
            if match_source == 'id' and id_value is not None:
                prompt_key = ('id', op_value, str(id_value))
                prompt_type = 'id'
                prompt_value = str(id_value)
            else:
                prompt_type = 'idx'
                key_value = normalized_idx if normalized_idx is not None else raw_idx_value
                prompt_key = ('idx', op_value, key_value)
                prompt_value = key_value
            sample_rank = sample_counters[prompt_key]
            sample_counters[prompt_key] += 1

            entry = {
                'op': op_value,
                'id': str(id_value) if id_value is not None else None,
                'template': template,
                'prompt_key_type': prompt_type,
                'prompt_key_value': prompt_value,
                'raw_index': raw_idx_value,
                'index': normalized_idx,
                'sample_rank': sample_rank,
                'match_source': match_source,
                'perfect': perfect,
                'value_mismatch_count': len(report['value_mismatches']),
                'dependency_mismatch_count': len(report['dependency_mismatches']),
                'answer_mismatch': report['answer_mismatch'] is not None,
                'extra_nodes': len(report['extra_in_pred']),
                'missing_nodes': len(report['missing_in_pred']),
            }
            if report['answer_mismatch'] is not None:
                answer_gold, answer_pred = report['answer_mismatch']
                entry['answer_mismatch_detail'] = [answer_gold, answer_pred]
            result_handle.write(json.dumps(entry))
            result_handle.write('\n')

    stats['unused_gold_entries'] = max(total_gold - len(used_graph_keys), 0)
    stats['template_stats'] = {
        template: dict(values) for template, values in template_stats.items()
    }
    if result_path is not None:
        stats['results_path'] = str(result_path)

    if result_handle is not None:
        result_handle.close()

    return stats, used_graph_keys


def _init_worker(gold_by_id, gold_by_idx, op_counts, total_gold, args_dict, generated_root):
    args_ns = SimpleNamespace(**args_dict)
    args_ns.results_dir = Path(args_ns.results_dir)
    args_ns.generated_path = Path(generated_root)
    _WORKER_STATE['gold_by_id'] = gold_by_id
    _WORKER_STATE['gold_by_idx'] = gold_by_idx
    _WORKER_STATE['op_counts'] = op_counts
    _WORKER_STATE['total_gold'] = total_gold
    _WORKER_STATE['args'] = args_ns


def _score_worker(file_path):
    state = _WORKER_STATE
    generated_file = Path(file_path)
    stats, used_keys = score_file(
        generated_file,
        state['args'],
        state['gold_by_id'],
        state['gold_by_idx'],
        state['op_counts'],
        state['total_gold'],
    )
    return stats, [list(item) for item in used_keys]

if __name__ == "__main__":
    main()
