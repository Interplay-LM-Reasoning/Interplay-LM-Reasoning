"""Build gold dependency graphs from ground-truth solutions.

Reads JSONL datasets (as produced by gsm-infinite) and serialises dependency
information for each example. The resulting files can be reused when evaluating
model generations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.solution_dependency_graph import DependencyGraphData, NodeInfo, SolutionParser


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def maybe_int(value: Optional[float]) -> Optional[float | int]:
    if value is None:
        return None
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value


def build_graph_record(
    record: Dict[str, Any],
    parser: SolutionParser,
    line_index: Optional[int] = None,
) -> Dict[str, Any]:
    raw_solution = record.get("solution")
    if not isinstance(raw_solution, str):
        return {
            "op": record.get("op"),
            "id": record.get("id"),
            "template": record.get("template"),
            "graph": None,
            "error": "missing-solution",
        }
    parsed = parser.parse(raw_solution)
    graph_data: DependencyGraphData = parser.build_graph(parsed)
    nodes: List[Dict[str, Any]] = []
    for step in parsed.steps:
        info: NodeInfo = graph_data.nodes[step.parameter_name]
        nodes.append(
            {
                "name": info.name,
                "variable": info.variable,
                "value": maybe_int(info.value),
                "dependencies": sorted(info.dependencies),
            }
        )
    try:
        topo = graph_data.topological_order()
    except ValueError:
        topo = None
    output = {
        "op": record.get("op"),
        "id": record.get("id"),
        "template": record.get("template"),
        "graph": {
            "nodes": nodes,
            "answer": maybe_int(parsed.answer),
            "topological_order": topo,
        },
    }
    if line_index is not None:
        output["line_index"] = line_index
    return output


def process_file(input_path: Path, output_path: Path, parser: SolutionParser) -> None:
    generated: List[str] = []
    for line_index, record in enumerate(load_jsonl(input_path)):
        graph_record = build_graph_record(record, parser, line_index=line_index)
        generated.append(json.dumps(graph_record, ensure_ascii=False))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(generated) + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Construct gold dependency graphs from datasets.")
    parser.add_argument("input_dir", type=Path, help="Directory containing JSONL datasets")
    parser.add_argument("output_dir", type=Path, help="Directory to write graph JSONL files")
    parser.add_argument(
        "--suffix",
        default="_graphs.jsonl",
        help="Suffix appended to each input filename for the graph output",
    )
    args = parser.parse_args(argv)

    solution_parser = SolutionParser()

    input_files = sorted(p for p in args.input_dir.glob("*.jsonl"))
    if not input_files:
        raise FileNotFoundError(f"No JSONL files found in {args.input_dir}")

    for path in input_files:
        output_path = args.output_dir / f"{path.stem}{args.suffix}"
        process_file(path, output_path, solution_parser)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
