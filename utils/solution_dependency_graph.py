"""Utilities to reconstruct dependency graphs from model solutions.

This module parses step-by-step solutions written in the GSM-infinite format
(e.g. "Define ... as X; so X = ...") and recovers the variable dependency
structure. The resulting dependency graph can be used to sanity-check whether
an answer is consistent with the intermediate calculations.

Limitations:
 - Currently tailored to the templated "Define ..." style solutions produced
   by gsm_infinite forward/reverse generators.
 - Expressions are assumed to contain only basic arithmetic (+, -, *).
 - More free-form reasoning text is ignored; we only use explicit assignment
   sentences.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import importlib.util

MODULE_PATH = REPO_ROOT / 'gsm_infinite' / 'gsm-infinite' / 'data' / 'realistic' / 'DependencyGraph.py'
_dependency_graph = None
if MODULE_PATH.exists():
    spec = importlib.util.spec_from_file_location('realistic_dependency_graph', MODULE_PATH)
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            _dependency_graph = module
        except ModuleNotFoundError as exc:
            if exc.name != 'pydot':
                raise

if _dependency_graph is not None:
    AbstractParameter = _dependency_graph.AbstractParameter
    DependencyGraph = _dependency_graph.DependencyGraph
    InstanceParameter = _dependency_graph.InstanceParameter
else:
    AbstractParameter = None
    DependencyGraph = None
    InstanceParameter = None

ASSIGNMENT_RE = re.compile(r"([A-Za-z])\s*=\s*([^.;]+)")
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
# Allow slightly malformed prompts like "Define ... a X;" where "as" was mistyped.
DEFINE_RE = re.compile(
    r"Define\s+(.*?)\s+(?:as|a)\s+(?:[A-Za-z]+\s+)*([A-Za-z]);",
    re.IGNORECASE | re.DOTALL,
)
TAG_RE = re.compile(r"<[^>]+>")
ALLOWED_EVAL_RE = re.compile(r"^[0-9+\-*/().\s]+$")


@dataclass
class StepParseWarning:
    message: str
    statement: Optional[str] = None


@dataclass
class SolutionStep:
    parameter_name: str
    variable: str
    raw_body: str
    statements: List[str]
    dependencies: Set[str] = field(default_factory=set)
    value: Optional[float] = None
    expressions: List[str] = field(default_factory=list)
    warnings: List[StepParseWarning] = field(default_factory=list)

    def add_warning(self, msg: str, statement: Optional[str] = None) -> None:
        self.warnings.append(StepParseWarning(msg, statement))


@dataclass
class ParsedSolution:
    steps: List[SolutionStep]
    answer: Optional[float]
    preamble: str

    @property
    def parameter_dependencies(self) -> Dict[str, Set[str]]:
        return {step.parameter_name: step.dependencies for step in self.steps}


@dataclass
class NodeInfo:
    name: str
    variable: str
    value: Optional[float]
    dependencies: Set[str] = field(default_factory=set)


@dataclass
class DependencyGraphData:
    nodes: Dict[str, NodeInfo]
    answer: Optional[float]

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "DependencyGraphData":
        nodes: Dict[str, NodeInfo] = {}
        for node in payload.get("nodes", []):
            name = node["name"]
            nodes[name] = NodeInfo(
                name=name,
                variable=node.get("variable"),
                value=node.get("value"),
                dependencies=set(node.get("dependencies", [])),
            )
        return cls(nodes=nodes, answer=payload.get("answer"))
    def compare(
        self,
        other: "DependencyGraphData",
        value_tolerance: float = 1e-6,
    ) -> Dict[str, Any]:
        """Compare two graphs ignoring variable naming differences."""

        def num(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        report: Dict[str, Any] = {
            "missing_in_pred": [],
            "extra_in_pred": [],
            "value_mismatches": [],
            "dependency_mismatches": [],
            "answer_mismatch": None,
        }

        gold_names = set(self.nodes)
        pred_names = set(other.nodes)

        for name in sorted(gold_names - pred_names):
            report["missing_in_pred"].append(name)
        for name in sorted(pred_names - gold_names):
            report["extra_in_pred"].append(name)

        shared = gold_names & pred_names
        for name in sorted(shared):
            gold_info = self.nodes[name]
            pred_info = other.nodes[name]

            gold_val = num(gold_info.value)
            pred_val = num(pred_info.value)
            if gold_val is None and pred_val is not None:
                report["value_mismatches"].append((name, gold_val, pred_val))
            elif gold_val is not None and pred_val is None:
                report["value_mismatches"].append((name, gold_val, pred_val))
            elif gold_val is not None and pred_val is not None:
                if abs(gold_val - pred_val) > value_tolerance:
                    report["value_mismatches"].append((name, gold_val, pred_val))

            gold_deps = set(gold_info.dependencies)
            pred_deps = set(pred_info.dependencies)
            if gold_deps != pred_deps:
                report["dependency_mismatches"].append(
                    {
                        "name": name,
                        "gold": sorted(gold_deps),
                        "pred": sorted(pred_deps),
                    }
                )

        gold_answer = num(self.answer)
        pred_answer = num(other.answer)
        if gold_answer is None and pred_answer is not None:
            report["answer_mismatch"] = (gold_answer, pred_answer)
        elif gold_answer is not None and pred_answer is None:
            report["answer_mismatch"] = (gold_answer, pred_answer)
        elif gold_answer is not None and pred_answer is not None:
            if abs(gold_answer - pred_answer) > value_tolerance:
                report["answer_mismatch"] = (gold_answer, pred_answer)

        return report

    def topological_order(self) -> List[str]:
        """Return a topological ordering of parameters if the graph is acyclic."""
        indegree = {name: 0 for name in self.nodes}
        dependents: Dict[str, Set[str]] = {name: set() for name in self.nodes}
        for name, info in self.nodes.items():
            for dep in info.dependencies:
                if dep in indegree:
                    indegree[name] += 1
                    dependents[dep].add(name)
        queue = deque([name for name, deg in indegree.items() if deg == 0])
        order: List[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for follower in dependents.get(current, set()):
                indegree[follower] -= 1
                if indegree[follower] == 0:
                    queue.append(follower)
        if len(order) != len(self.nodes):
            raise ValueError("Dependency graph contains a cycle or unresolved nodes")
        return order

    def to_dependency_graph(self) -> DependencyGraph:
        if DependencyGraph is None or AbstractParameter is None or InstanceParameter is None:
            raise ImportError('DependencyGraph utilities are unavailable (missing dependency such as pydot).')
        graph = DependencyGraph()
        created_nodes: Dict[str, InstanceParameter | AbstractParameter] = {}

        def make_node(name: str) -> InstanceParameter | AbstractParameter:
            if name in created_nodes:
                return created_nodes[name]
            if " per " in name:
                node: InstanceParameter | AbstractParameter = AbstractParameter(name)
            else:
                node = InstanceParameter(name)
            info = self.nodes[name]
            node.value = info.value
            node.variable = info.variable
            created_nodes[name] = node
            return node

        for name in self.nodes:
            make_node(name)

        for name, info in self.nodes.items():
            node = created_nodes[name]
            for dep_name in info.dependencies:
                if dep_name not in created_nodes:
                    continue
                graph.add_edge(created_nodes[dep_name], node)

        for node in created_nodes.values():
            if isinstance(node, AbstractParameter):
                if node not in graph.abstractparameters:
                    graph.abstractparameters.append(node)
            else:
                if node not in graph.instanceparameters:
                    graph.instanceparameters.append(node)

        return graph


class SolutionParser:
    """Parse gsm-infinite style solutions into dependency graphs."""

    def __init__(self) -> None:
        # Maps variable -> parameter name for already processed steps
        self.var_to_param: Dict[str, str] = {}
        # Numeric value known for variables (parameter or intermediate)
        self.variable_values: Dict[str, float] = {}

    def parse(self, raw_solution: str) -> ParsedSolution:
        answer = self._extract_answer(raw_solution)
        cleaned = TAG_RE.sub("", raw_solution)
        cleaned = cleaned.strip()
        # Reset parser state for each new solution
        self.var_to_param = {}
        self.variable_values = {}
        preamble, body = self._split_preamble(cleaned)

        steps: List[SolutionStep] = []
        matches = list(DEFINE_RE.finditer(body))
        if not matches:
            return ParsedSolution([], answer=answer, preamble=preamble)

        for idx, match in enumerate(matches):
            raw_param = " ".join(match.group(1).split())
            param_name = self._normalize_parameter_name(raw_param)
            variable = match.group(2).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
            raw_body = body[start:end].strip()
            raw_body = self._strip_answer_tail(raw_body)
            statements = [s.strip() for s in raw_body.split(";") if s.strip()]
            step = SolutionStep(param_name, variable, raw_body, statements)
            self._process_step(step)
            steps.append(step)

        final_values = dict(self.variable_values)
        self.variable_values = final_values
        for step in steps:
            if step.value is None and step.expressions:
                for expr in reversed(step.expressions):
                    value = self._evaluate_expression(expr, {})
                    if value is not None:
                        step.value = value
                        break
            if step.value is None and step.variable in final_values:
                step.value = final_values[step.variable]
            if step.value is not None:
                final_values[step.variable] = step.value
                self.variable_values[step.variable] = step.value

        return ParsedSolution(steps=steps, answer=answer, preamble=preamble)

    @staticmethod
    def _normalize_parameter_name(name: str) -> str:
        """Attempt to undo minor rendering glitches in parameter names."""
        cleaned = " ".join(name.split())
        cleaned = cleaned.replace("\ufffd", "").strip(" .;")
        matches = list(re.finditer(r"\bDefine\s+", cleaned, flags=re.IGNORECASE))
        if matches:
            cleaned = cleaned[matches[-1].end():]
        return cleaned.strip(" .;")

    def build_graph(self, parsed: ParsedSolution) -> DependencyGraphData:
        nodes: Dict[str, NodeInfo] = {}
        for step in parsed.steps:
            nodes[step.parameter_name] = NodeInfo(
                name=step.parameter_name,
                variable=step.variable,
                value=step.value,
                dependencies=set(step.dependencies),
            )
        return DependencyGraphData(nodes=nodes, answer=parsed.answer)

    def _process_step(self, step: SolutionStep) -> None:
        current_var = step.variable
        intermediate_deps: Dict[str, Set[str]] = {}
        intermediate_values: Dict[str, float] = {}
        last_value: Optional[float] = None

        for match in ASSIGNMENT_RE.finditer(step.raw_body):
            target_var = match.group(1)
            expr = match.group(2).strip()
            expr_eval = expr.split("=")[-1].strip()
            source_deps = self._collect_dependencies(expr, intermediate_deps)

            if target_var == current_var:
                step.dependencies.update(source_deps)
                if expr_eval:
                    step.expressions.append(expr_eval)
            else:
                intermediate_deps.setdefault(target_var, set()).update(source_deps)

            value = self._evaluate_expression(expr_eval, intermediate_values)
            if value is not None:
                intermediate_values[target_var] = value
                self.variable_values[target_var] = value
                if target_var == current_var:
                    last_value = value
            else:
                fallback_value = self._extract_last_number(match.group(0))
                if fallback_value is not None:
                    intermediate_values[target_var] = fallback_value
                    self.variable_values[target_var] = fallback_value
                    if target_var == current_var:
                        last_value = fallback_value

        if last_value is None:
            last_value = self._extract_last_number(step.raw_body)
            if last_value is not None:
                self.variable_values[current_var] = last_value
        step.value = last_value
        self.var_to_param[current_var] = step.parameter_name

    def _collect_dependencies(
        self,
        expr: str,
        intermediate_deps: Dict[str, Set[str]],
    ) -> Set[str]:
        deps: Set[str] = set()
        for token in re.findall(r"[A-Za-z]", expr):
            if token in self.var_to_param:
                deps.add(self.var_to_param[token])
            if token in intermediate_deps:
                deps.update(intermediate_deps[token])
        return deps

    def _evaluate_expression(
        self,
        expr: str,
        intermediate_values: Dict[str, float],
    ) -> Optional[float]:
        tokens = set(re.findall(r"[A-Za-z]", expr))
        substituted = expr
        lookup: Dict[str, float] = {}
        lookup.update(self.variable_values)
        lookup.update(intermediate_values)

        for tok in tokens:
            if tok not in lookup:
                return None
            substituted = re.sub(rf"\b{tok}\b", str(lookup[tok]), substituted)

        substituted = substituted.strip()
        if not substituted:
            return None
        if not ALLOWED_EVAL_RE.match(substituted):
            return None
        try:
            return float(eval(substituted, {"__builtins__": {}}))
        except Exception:
            return None

    @staticmethod
    def _strip_answer_tail(body: str) -> str:
        answer_idx = body.find("Answer:")
        if answer_idx != -1:
            return body[:answer_idx].strip()
        return body.strip()

    @staticmethod
    def _extract_last_number(text: str) -> Optional[float]:
        numbers = NUMBER_RE.findall(text)
        if not numbers:
            return None
        try:
            return float(numbers[-1])
        except ValueError:
            return None

    @staticmethod
    def _extract_answer(solution: str) -> Optional[float]:
        tag_match = re.search(r"<answer>\s*([-+]?\d+(?:\.\d+)?)", solution, re.IGNORECASE)
        if tag_match:
            try:
                return float(tag_match.group(1))
            except ValueError:
                return None
        match = re.search(r"Answer:\s*([-+]?\d+(?:\.\d+)?)", solution)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    @staticmethod
    def _split_preamble(solution: str) -> Tuple[str, str]:
        first_define = DEFINE_RE.search(solution)
        if not first_define:
            return solution, ""
        start = first_define.start()
        return solution[:start].strip(), solution[start:]


def load_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct dependency graph from solutions")
    parser.add_argument("jsonl", type=Path, help="Path to JSONL file with solutions")
    parser.add_argument(
        "--solution-key",
        default="gen_solution_answer",
        help="Key in each JSON record containing the solution text",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of examples to parse (0 means all)",
    )
    args = parser.parse_args(argv)

    parser_obj = SolutionParser()
    count = 0
    for record in load_jsonl(args.jsonl):
        raw_solution = record.get(args.solution_key)
        if not isinstance(raw_solution, str):
            continue
        parsed = parser_obj.parse(raw_solution)
        graph_data = parser_obj.build_graph(parsed)
        print("#" * 80)
        print(f"Example {count}")
        for step in parsed.steps:
            print(f"- {step.parameter_name} (var {step.variable}) depends on: {sorted(step.dependencies)} value={step.value}")
        print(f"Answer reported: {parsed.answer}")
        count += 1
        if args.limit and count >= args.limit:
            break


if __name__ == "__main__":
    main()
