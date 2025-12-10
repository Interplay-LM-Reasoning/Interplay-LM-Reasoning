
from typing import Tuple


def _split_solution(sol: str) -> Tuple[str, str]:
    """Split solution text into body and answer parts based on 'Answer:' marker.

    Returns (solution_body, answer_text)
    """
    if not sol:
        return "", ""
    if "Answer:" not in sol:
        return sol.strip(), ""
    pre, ans = sol.rsplit("Answer:", 1)
    ans = ans.strip().splitlines()[0].strip().rstrip(".")
    return pre.strip(), ans

def compose_text(obj: dict, add_special_tokens: bool = True) -> str:
    """Compose a training text string from a raw example.

    When add_special_tokens=True, format as:
      <question> <problem+question> </question> <solution> <solution_body> </solution> <answer> <answer> </answer>
    falling back to plain fields if required keys are missing.
    """
    problem = (obj.get("problem") or "").strip()
    question = (obj.get("question") or "").strip()
    solution = (obj.get("solution") or "").strip()

    if add_special_tokens and (problem or question or solution):
        sol_body, answer = _split_solution(solution)
        pq = (problem + " " + question).strip()
        parts = []
        if pq:
            parts.extend(["<question>", pq, "</question>"])
        if sol_body:
            parts.extend(["<solution>", sol_body, "</solution>"])
        if answer:
            parts.extend(["<answer>", answer, "</answer>"])
        text = " ".join([p for p in parts if p]).strip()
        if text:
            return text