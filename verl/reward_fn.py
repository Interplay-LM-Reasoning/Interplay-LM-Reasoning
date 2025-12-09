import re
from typing import Any, Dict, Optional, Union

from utils.solution_dependency_graph import SolutionParser


def extract_answer(text: str) -> str:
    """Extract answer from generated text.
    Priority:
      1) Between <answer>...</answer>
      2) If <answer> exists but no closing tag, take text after it up to a tag start '<' or newline.
    """
    # 1) Proper tags
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 2) Open tag only
    pos = text.lower().rfind("<answer>")
    if pos != -1:
        tail = text[pos + len("<answer>"):]
        # find the numeric text up to the next tag or at the end, if not found return ""
        m2 = re.search(r"(.*?)(<|\n|$)", tail, flags=re.DOTALL)
        if m2:
            return m2.group(1).strip()
    return ""




def compute_score(solution_str: str, ground_truth: Dict[str,str], data_source=None, extra_info=None) -> bool:
    gold_answer = ground_truth['answer']
    answer = extract_answer(solution_str)
    ret_score = answer.strip().rstrip(".") == gold_answer.strip().rstrip(".")  

    return ret_score

def parse_graph(solution_str):
    global solution_parser
    if solution_parser is None:
        solution_parser = SolutionParser()
    parsed = solution_parser.parse(solution_str)
    graph = solution_parser.build_graph(parsed)
    setattr(graph, "steps", len(parsed.steps))
    setattr(graph, "parsed_solution", parsed)
    return graph

solution_parser = None

def _normalise_answer(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip().rstrip(".")


def _compute_step_process_reward(
    solution_str: str,
    ground_truth: Union[Dict, str],
    extra_info: Optional[Dict[str, Any]],
    *,
    answer_weight: float,
    step_weight: float,
    value_tolerance: float,
    zero_on_process_mismatch: bool = False,
) -> Dict[str, Any]:
    extra_info_dict = extra_info if isinstance(extra_info, dict) else {}

    gold_answer: Optional[str]
    gold_solution: Optional[str]
    if isinstance(ground_truth, dict):
        gold_answer = ground_truth.get("answer")
        gold_solution = ground_truth.get("solution") or ground_truth.get("gold_solution")
    else:
        gold_answer = ground_truth
        gold_solution = None

    if not gold_answer:     
        gold_answer = extra_info_dict.get("gold_answer")
    if not gold_solution:
        gold_solution = extra_info_dict.get("gold_solution")

    predicted_answer = extract_answer(solution_str)
    outcome_reward = 0.0
    if gold_answer:
        outcome_reward = 1.0 if _normalise_answer(predicted_answer) == _normalise_answer(gold_answer) else 0.0

    process_reward = 0.0
    gold_graph = None
    process_mismatch = False
    if isinstance(gold_solution, str) and gold_solution.strip():
        try:
            gold_graph = parse_graph(gold_solution)
        except Exception:
            gold_graph = None

    if gold_graph is not None:
        try:
            pred_graph = parse_graph(solution_str)
        except Exception:
            pred_graph = None

        if pred_graph is not None:
            report = gold_graph.compare(pred_graph, value_tolerance=value_tolerance)
            total_nodes = max(len(gold_graph.nodes), 1)
            structural_penalty = (
                len(report["value_mismatches"])
                + len(report["dependency_mismatches"])
                + len(report["missing_in_pred"])
            )
            structural_penalty += min(len(report["extra_in_pred"]), total_nodes)
            if report["answer_mismatch"] is not None:
                structural_penalty += 1
            process_reward = max(0.0, 1.0 - structural_penalty / total_nodes)
            process_mismatch = structural_penalty > 0
        else:
            process_reward = 0.0
            process_mismatch = True
    else:
        # Fall back to outcome reward if process supervision is unavailable.
        process_reward = outcome_reward
        process_mismatch = False

    if extra_info_dict is not None:
        extra_info_dict["outcome_reward"] = outcome_reward
        extra_info_dict["process_reward"] = process_reward
        extra_info_dict["process_reward_has_gold"] = gold_graph is not None
        if zero_on_process_mismatch:
            extra_info_dict["process_zeroed_due_to_mismatch"] = False

    total_weight = answer_weight + step_weight
    if total_weight <= 0:
        answer_weight = step_weight = 0.5
        total_weight = 1.0

    answer_component = (answer_weight / total_weight) * outcome_reward
    step_component = (step_weight / total_weight) * process_reward
    reward = max(0.0, min(1.0, answer_component + step_component))
    if outcome_reward < 1.0:
        reward = 0.0
    zeroed_due_to_mismatch = False
    if (
        zero_on_process_mismatch
        and gold_graph is not None
        and outcome_reward >= 1.0
        and process_mismatch
    ):
        reward = 0.0
        zeroed_due_to_mismatch = True
    if extra_info_dict is not None and zero_on_process_mismatch:
        extra_info_dict["process_zeroed_due_to_mismatch"] = zeroed_due_to_mismatch

    return {
        "score": reward,
        "outcome_reward": outcome_reward,
        "process_reward": process_reward,
        "process_reward_has_gold": gold_graph is not None,
        "process_zeroed_due_to_mismatch": zeroed_due_to_mismatch,
    }


def compute_score_with_step_process(
    solution_str: str,
    ground_truth: Union[Dict, str],
    data_source: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    *,
    answer_weight: float = 0.2,
    step_weight: float = 0.8,
    value_tolerance: float = 1e-6,
    zero_on_process_mismatch: bool = False,
) -> Dict[str, Any]:
    """
    Blend answer-level accuracy with a process reward derived from dependency graphs.

    Args:
        solution_str: Model generated solution text.
        ground_truth: Ground-truth payload; may be either a string answer or a dict containing
            keys like ``answer`` and ``solution``.
        data_source: Identifier for the example (unused, but kept for compatibility).
        extra_info: Mutable metadata dict that can be enriched with reward diagnostics.
        answer_weight: Relative weight assigned to the outcome reward.
        step_weight: Relative weight assigned to the process reward.
        value_tolerance: Tolerance for numeric comparisons during graph matching.
        zero_on_process_mismatch: If True, zero the blended reward when the answer is correct but the process mismatches.

    Returns:
        A dictionary containing the blended ``score`` alongside outcome/process diagnostics.
    """

    return _compute_step_process_reward(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        answer_weight=answer_weight,
        step_weight=step_weight,
        value_tolerance=value_tolerance,
        zero_on_process_mismatch=zero_on_process_mismatch,
    )


def compute_score_with_step_process_batched(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    *,
    answer_weight: float = 0.2,
    step_weight: float = 0.8,
    value_tolerance: float = 1e-6,
    zero_on_process_mismatch: bool = False,
):
    """
    Batched variant of ``compute_score_with_step_process`` compatible with ``BatchRewardManager``.
    """

    _ = data_sources  # kept for API symmetry with other reward fns
    results: list[Dict[str, Any]] = []
    for solution_str, ground_truth, extra_info in zip(
        solution_strs, ground_truths, extra_infos, strict=True
    ):
        results.append(
            _compute_step_process_reward(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                answer_weight=answer_weight,
                step_weight=step_weight,
                value_tolerance=value_tolerance,
                zero_on_process_mismatch=zero_on_process_mismatch,
            )
        )
    return results


def compute_score_with_step_process_strict(
    solution_str: str,
    ground_truth: Union[Dict, str],
    data_source: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    *,
    answer_weight: float = 0.2,
    step_weight: float = 0.8,
    value_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Variant of ``compute_score_with_step_process`` that zeros the reward if the process mismatches despite a correct answer.
    """

    _ = data_source  # kept for API compatibility
    return _compute_step_process_reward(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        answer_weight=answer_weight,
        step_weight=step_weight,
        value_tolerance=value_tolerance,
        zero_on_process_mismatch=True,
    )


def compute_score_with_step_process_strict_batched(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    *,
    answer_weight: float = 0.2,
    step_weight: float = 0.8,
    value_tolerance: float = 1e-6,
):
    """
    Batched variant of ``compute_score_with_step_process_strict``.
    """

    return compute_score_with_step_process_batched(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
        answer_weight=answer_weight,
        step_weight=step_weight,
        value_tolerance=value_tolerance,
        zero_on_process_mismatch=True,
    )
