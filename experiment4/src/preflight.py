#!/usr/bin/env python3
"""Fail-fast environment preflight for E4. Run BEFORE any GPU phase.

Checks, in order:
  1. package versions match the pinned known-good matrix (requirements.txt);
  2. the authors' repo is importable and its scripts parse;
  3. the exact transformers API vLLM 0.10.1.1 needs exists (the call that
     crashed the first H100 job), exercised on a real downloaded tokenizer
     when one is available — all on CPU.

Exit code 0 = safe to proceed. Any failure prints a one-line fix.
Usage: python src/preflight.py [--repo PATH] [--models-dir PATH] [--gpu]
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str, fix: str = "") -> None:
    mark = "ok " if ok else "FAIL"
    print(f"[preflight] {mark} {name}: {detail}")
    if not ok:
        FAILURES.append(f"{name}: {detail}" + (f" — fix: {fix}" if fix else ""))


def version_tuple(v: str) -> tuple:
    # Leading digits only: '2.7.1+cu126' -> (2,7,1), '4.56.0.dev0' -> (4,56,0).
    import re
    parts = []
    for tok in v.split(".")[:3]:
        m = re.match(r"\d+", tok)
        parts.append(int(m.group(0)) if m else 0)
    return tuple(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=None, help="authors' repo root")
    ap.add_argument("--models-dir", type=Path, default=None,
                    help="downloaded checkpoints root (tokenizer dry-run if present)")
    ap.add_argument("--gpu", action="store_true", help="also require CUDA visibility")
    args = ap.parse_args()

    # --- 1. version matrix --------------------------------------------------
    reinstall = "pip install -r requirements.txt (re-run scripts/run_all_pace.sbatch does this automatically)"

    try:
        import transformers
        v = transformers.__version__
        check("transformers", version_tuple(v) >= (4, 55, 0) and version_tuple(v) < (5,),
              f"{v} (need >=4.55,<5 — v5 removed APIs vLLM 0.10.x uses)", reinstall)
    except ImportError as e:
        check("transformers", False, str(e), reinstall)

    try:
        import vllm
        check("vllm", vllm.__version__ == "0.10.1.1",
              f"{vllm.__version__} (need ==0.10.1.1)", reinstall)
    except ImportError as e:
        check("vllm", False, str(e), reinstall)

    try:
        import torch
        check("torch", version_tuple(torch.__version__) == (2, 7, 1),
              f"{torch.__version__} (need 2.7.1, pinned by vllm)", reinstall)
        if args.gpu:
            check("cuda", torch.cuda.is_available(),
                  f"cuda_available={torch.cuda.is_available()}"
                  + (f" device={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""),
                  "run on a GPU node / check --gres")
    except ImportError as e:
        check("torch", False, str(e), reinstall)

    try:
        import numpy
        check("numpy", version_tuple(numpy.__version__) < (2, 3),
              f"{numpy.__version__} (need <2.3 for numba/vllm)", reinstall)
    except ImportError as e:
        check("numpy", False, str(e), reinstall)

    try:
        import pandas
        check("pandas", version_tuple(pandas.__version__) < (3,),
              f"{pandas.__version__} (analysis tested on 2.x)", reinstall)
    except ImportError as e:
        check("pandas", False, str(e), reinstall)

    for mod in ("matplotlib", "tqdm", "safetensors", "huggingface_hub"):
        try:
            m = importlib.import_module(mod)
            check(mod, True, getattr(m, "__version__", "?"))
        except ImportError as e:
            check(mod, False, str(e), reinstall)

    # --- 2. authors' repo ---------------------------------------------------
    if args.repo:
        repo = args.repo
        ok = (repo / "scripts" / "eval_checkpoints.py").is_file()
        check("authors-repo", ok, str(repo), "bash scripts/00_setup_env.sh (clones it)")
        if ok:
            sys.path.insert(0, str(repo))
            try:
                from utils.solution_dependency_graph import SolutionParser  # noqa: F401
                import py_compile
                for s in ("eval_checkpoints.py", "build_gold_dependency_graphs.py",
                          "score_dependency_graphs.py", "aggregate_graph_pass_results.py"):
                    py_compile.compile(str(repo / "scripts" / s), doraise=True)
                check("authors-scripts", True, "all 4 eval scripts compile; utils import")
            except Exception as e:  # noqa: BLE001
                check("authors-scripts", False, repr(e), "re-clone the repo")

    # --- 3. the exact API that crashed job 6588817, exercised on CPU --------
    try:
        from transformers import AutoTokenizer
        tok_path = None
        if args.models_dir and args.models_dir.is_dir():
            hits = sorted(args.models_dir.rglob("tokenizer.json"))
            if hits:
                tok_path = hits[0].parent
        if tok_path is not None:
            tok = AutoTokenizer.from_pretrained(str(tok_path))
            _ = tok.all_special_tokens_extended      # AttributeError on transformers 5.x
            _ = tok.convert_tokens_to_ids("</answer>")
            check("tokenizer-dry-run", True,
                  f"{type(tok).__name__} from {tok_path} has all_special_tokens_extended")
        else:
            from transformers import PreTrainedTokenizerBase
            ok = hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended") or \
                version_tuple(__import__("transformers").__version__) < (5,)
            check("tokenizer-api", ok,
                  "all_special_tokens_extended available (no local tokenizer to load yet)")
    except Exception as e:  # noqa: BLE001
        check("tokenizer-dry-run", False, repr(e), reinstall)

    print()
    if FAILURES:
        print("[preflight] BLOCKED — fix before booking GPU time:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("[preflight] all checks passed — safe to run the pipeline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
