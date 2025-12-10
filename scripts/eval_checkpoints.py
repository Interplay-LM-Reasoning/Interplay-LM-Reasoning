#!/usr/bin/env python
"""
Evaluate all checkpoints on test_id and test_ood splits with torchrun.

Features:
- Discovers checkpoints (checkpoint-*) under a root dir.
- Loads test_200_id and test_200_ood from data/split/difficulty/... (configurable).
- Uses official DataLoader with DistributedSampler to shard across ranks for HF backend.
- Two evaluations per split:
    1) Generation: prompt until <solution>, generate solution+answer. Metrics: answer EM and avg response length.
    2) Loss (teacher forcing): compute average model loss over gold full sequence (labels=input_ids). Skipped when using vLLM.
- Reports per-op accuracy and split-level metrics, plus aggregated total across both splits.

Backends:
- HF (default): transformers.generate
- vLLM: fast generation via vLLM engine (single process recommended; use --vllm-tensor-parallel-size for multi-GPU)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import math
import multiprocessing as mp
import queue
import traceback
from collections import defaultdict
import shutil

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch.nn.functional as F
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x=None, *args, **kwargs):
        return x if x is not None else range(0)

import sys
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))
if str(PROJ_ROOT / "..") not in sys.path:
    # Support legacy layouts where scripts lived under an extra subdir.
    sys.path.insert(0, str(PROJ_ROOT / ".."))

from utils.text_preprocess import compose_text
from datetime import timedelta




# Optional vLLM import
_VLLM_AVAILABLE = False
try:
    from vllm import LLM, SamplingParams  # type: ignore
    _VLLM_AVAILABLE = True
except Exception:
    _VLLM_AVAILABLE = False


def init_dist_if_needed():
    if dist.is_available() and not dist.is_initialized():
        try:
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                timeout=timedelta(seconds=1800)
            )
        except Exception:
            pass
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


def _normalize_pattern(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() == "none":
        return None
    return stripped


def find_checkpoints(root: Path, pattern: Optional[str] = "checkpoint-*") -> List[Path]:
    if not root.exists():
        return []
    pattern = _normalize_pattern(pattern)
    if pattern is None:
        return [root] if root.is_dir() else []
    cks = sorted(
        [p for p in root.glob(pattern) if p.is_dir()],
        key=lambda p: int(re.findall(r"(\d+)$", p.name)[0]) if re.findall(r"(\d+)$", p.name) else -1
    )
    if not cks and root.is_dir():
        # Fall back to treating the root itself as a checkpoint directory when globbing fails.
        marker_files = {"config.json", "model.safetensors", "pytorch_model.bin"}
        if any((root / marker).exists() for marker in marker_files):
            return [root]
    return cks


def construct_prompts_from_rows(rows: List[Dict]) -> List[str]:
    prompts = []
    for r in rows:
        query = compose_text(r).split("<solution>")[0] + "<solution>"
        prompts.append(query)
    return prompts


def get_visible_cuda_devices() -> List[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        return [dev.strip() for dev in env.split(",") if dev.strip()]
    try:
        count = torch.cuda.device_count()
    except Exception:
        count = 0
    return [str(i) for i in range(count)]


def extract_problem_from_sequence(seq: str) -> str:
    try:
        m = re.search(r"<question>(.*?)</question>", seq, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return ""


def extract_answer_flexible(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    pos = text.lower().rfind("<answer>")
    if pos != -1:
        tail = text[pos + len("<answer>"):]
        m2 = re.search(r"(.*?)(<|\n|$)", tail, flags=re.DOTALL)
        if m2:
            return m2.group(1).strip()
    return ""


def _numeric_from_string(s: str) -> Optional[float]:
    from fractions import Fraction
    import re as _re
    if not s:
        return None
    t = s.strip().rstrip(".")
    if not t:
        return None
    m = _re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", t)
    if m:
        try:
            return float(Fraction(int(m.group(1)), int(m.group(2))))
        except Exception:
            return None
    t2 = _re.sub(r"(?<=\d),(?=\d)", "", t)
    m = _re.fullmatch(r"[+-]?(?:\d+\.?\d*|\d*\.\d+)", t2)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    m2 = _re.findall(r"[-+]?\d+(?:[\,\s]?\d+)*(?:\.\d+)?", s)
    if m2:
        token = m2[-1].replace(",", "")
        try:
            return float(token)
        except Exception:
            return None
    return None


def _resolve_vllm_dtype(requested_dtype: str) -> str:
    """
    vLLM defaults to the dtype recorded in the checkpoint (often bfloat16).
    Some GPUs (e.g. non-A100 Ampere cards) cannot execute bfloat16 kernels,
    which can cause the vLLM engine to die during its start-up handshake.
    We proactively fall back to float16 in that case.
    """
    normalized = (requested_dtype or "").strip().lower()
    if normalized and normalized not in {"auto"}:
        return normalized

    if not torch.cuda.is_available():
        return normalized or "auto"

    try:
        current_device = torch.cuda.current_device()
    except Exception:
        current_device = 0

    try:
        with torch.cuda.device(current_device):
            torch.zeros(1, dtype=torch.bfloat16, device="cuda")
    except Exception:
        return "float16"
    else:
        return "bfloat16"


def load_model_and_tokenizer(model_path: Path, device: torch.device):
    tok_dir = model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=None,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def safe_max_length(tokenizer) -> int:
    ml = getattr(tokenizer, "model_max_length", 2048)
    try:
        if not isinstance(ml, int) or ml > 100000:
            return 2048
    except Exception:
        return 2048
    return min(ml, 4096)


def distributed_all_gather_dict(d: Dict) -> List[Dict]:
    if not (dist.is_available() and dist.is_initialized()):
        return [d]
    gathered: List[Dict] = [None for _ in range(dist.get_world_size())]  # type: ignore
    dist.all_gather_object(gathered, d)
    return gathered


def merge_op_counts(dicts: List[Dict[str, Dict[str, int]]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for d in dicts:
        for op, cnt in d.items():
            if op not in out:
                out[op] = {"total": 0, "correct": 0, "resp_tokens": 0}
            out[op]["total"] += cnt.get("total", 0)
            out[op]["correct"] += cnt.get("correct", 0)
            out[op]["resp_tokens"] += cnt.get("resp_tokens", 0)
    return out


def _aggregate_pass_counts(
    examples: List[Dict[str, Any]], *, key_prefix: str = ""
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Dict[str, Tuple[int, int]]]]:
    overall: Dict[str, List[int]] = defaultdict(list)
    per_op: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    prefix = key_prefix or ""
    for record in examples:
        idx = record.get("__idx")
        if idx is None:
            continue
        key = f"{prefix}{idx}"
        em_val = record.get("exact_match", 0)
        try:
            em = int(em_val)
        except Exception:
            continue
        overall[key].append(em)
        op = str(record.get("op", "unknown"))
        per_op[op][key].append(em)

    counts = {k: (len(vals), sum(vals)) for k, vals in overall.items() if vals}
    per_op_counts: Dict[str, Dict[str, Tuple[int, int]]] = {}
    for op, idx_map in per_op.items():
        per_op_counts[op] = {k: (len(vals), sum(vals)) for k, vals in idx_map.items() if vals}
    return counts, per_op_counts


def _pass_probability(n: int, successes: int, k: int) -> float:
    if k <= 0 or n < k or successes <= 0:
        return 0.0
    failures = n - successes
    if failures < k:
        return 1.0
    try:
        return 1.0 - math.comb(failures, k) / math.comb(n, k)
    except ValueError:
        return 0.0


def _build_k_values(max_samples: int, max_k: Optional[int]) -> List[int]:
    if max_samples <= 0:
        return []
    target = max_k if max_k and max_k > 0 else max_samples
    target = max(1, target)
    ks: List[int] = [1]
    cur = 2
    while cur < target:
        ks.append(cur)
        cur *= 2
    if ks[-1] != target:
        ks.append(target)
    return sorted({k for k in ks if k <= max_samples})


def _compute_pass_series(counts: Iterable[Tuple[int, int]], max_k: Optional[int]) -> Dict[int, float]:
    counts = list(counts)
    if not counts:
        return {}
    max_samples = max(n for n, _ in counts)
    ks = _build_k_values(max_samples, max_k)
    results: Dict[int, float] = {}
    for k in ks:
        eligible = [(n, s) for n, s in counts if n >= k]
        if not eligible:
            results[k] = 0.0
            continue
        total = sum(_pass_probability(n, s, k) for n, s in eligible)
        results[k] = total / len(eligible)
    return results


def _format_pass(series: Dict[int, float]) -> Dict[str, float]:
    return {f"pass@{k}": float(v) for k, v in sorted(series.items())}


def _parse_int(value: Any) -> Optional[int]:
    try:
        if isinstance(value, bool):  # bool is subclass of int; exclude explicitly
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            return int(stripped)
    except Exception:
        return None
    return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        for obj in iter_jsonl(path):
            if isinstance(obj, dict):
                rows.append(obj)
    except Exception:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows


def _categorize_op(op: Optional[int], threshold: Optional[int]) -> str:
    if threshold is None or threshold < 0 or op is None:
        return "id"
    return "id" if op <= threshold else "ood"


def load_eval_rows(
    data_root: Path,
    id_op_threshold: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load evaluation rows from a data root.

    Supports multiple layouts:
      1) Single jsonl file containing all rows (treated as ID).
      2) Directory with test_200_id.jsonl / test_200_ood.jsonl.
      3) Directory with test_id/ and test_ood/ subdirectories.
      4) Directory containing per-op jsonl files (e.g., op2-*.jsonl).
    """
    if data_root.is_file():
        rows = _read_jsonl(data_root)
        return rows, []

    id_file = data_root / "test_200_id.jsonl"
    ood_file = data_root / "test_200_ood.jsonl"
    if id_file.exists() or ood_file.exists():
        return _read_jsonl(id_file), _read_jsonl(ood_file)

    legacy_dirs = [
        (data_root / "test_id", data_root / "test_ood"),
        (data_root / "val_id", data_root / "val_ood"),
    ]
    for id_dir, ood_dir in legacy_dirs:
        if id_dir.exists() or ood_dir.exists():
            id_rows = load_split(id_dir) if id_dir.exists() else []
            ood_rows = load_split(ood_dir) if ood_dir.exists() else []
            return id_rows, ood_rows

    jsonl_files = sorted(data_root.glob("*.jsonl"))
    if not jsonl_files:
        return [], []

    op_pat = re.compile(r"op(\d+)", flags=re.IGNORECASE)
    id_rows: List[Dict[str, Any]] = []
    ood_rows: List[Dict[str, Any]] = []

    for file in jsonl_files:
        m = op_pat.search(file.stem)
        op_from_file: Optional[int] = None
        if m:
            try:
                op_from_file = int(m.group(1))
            except Exception:
                op_from_file = None
        rows = _read_jsonl(file)
        for row in rows:
            op_value = _parse_int(row.get("op"))
            if op_value is None and op_from_file is not None:
                row.setdefault("op", op_from_file)
                op_value = op_from_file
            bucket = _categorize_op(op_value, id_op_threshold)
            if bucket == "id":
                id_rows.append(row)
            else:
                ood_rows.append(row)

    return id_rows, ood_rows


class RowsDataset(Dataset):
    def __init__(self, rows: List[Dict]):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx: int) -> Dict:
        return self.rows[idx]


class LMFullWithOpDataset(Dataset):
    """Returns tokenized tensors plus op label for per-op loss stats."""
    def __init__(self, rows: List[Dict], tokenizer, max_length: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx: int):
        row = self.rows[idx]
        text = compose_text(row)
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = item["input_ids"].clone()
        op = str(row.get("op", "unknown"))
        return item, op


@torch.no_grad()
def eval_generation(
    model,
    tokenizer,
    rows: List[Dict],
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    sample_k: int = 1,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 0,
    return_examples: bool = False,
):
    eos_id = tokenizer.convert_tokens_to_ids("</answer>")
    if eos_id == tokenizer.unk_token_id or eos_id is None:
        eos_id = tokenizer.eos_token_id

    total = 0
    correct = 0
    total_response_tokens = 0
    per_op: Dict[str, Dict[str, int]] = {}

    ds = RowsDataset(rows)
    sampler = DistributedSampler(ds, shuffle=False) if dist.is_available() and dist.is_initialized() else None
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False, collate_fn=lambda xs: xs)
    collected: List[Dict] = []

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    iterator = tqdm(dl, total=len(dl), desc="gen", disable=(rank != 0))
    for batch_rows in iterator:
        prompts = construct_prompts_from_rows(batch_rows)
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(getattr(tokenizer, "model_max_length", 2048), 2048),
        )
        if "token_type_ids" in enc:
            del enc["token_type_ids"]
        enc = {k: v.to(device) for k, v in enc.items()}

        gen_kwargs = dict(
            **enc,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        if sample_k > 1:
            gen_kwargs.update(dict(
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                num_return_sequences=int(sample_k)
            ))
            if top_k is not None and top_k > 0:
                gen_kwargs["top_k"] = int(top_k)
        else:
            gen_kwargs.update(dict(do_sample=False))

        gen = model.generate(**gen_kwargs)

        B = len(batch_rows)
        if sample_k > 1:
            seq_len = gen.size(1)
            gen = gen.view(B, int(sample_k), seq_len)
            for j, row in enumerate(batch_rows):
                prompt_len = int(enc["attention_mask"][j].sum().item())
                for s in range(int(sample_k)):
                    new_ids = gen[j, s, prompt_len:]

                    resp_len = len(new_ids)
                    try:
                        if eos_id is not None:
                            pos = (new_ids == eos_id).nonzero(as_tuple=False)
                            if pos.numel() > 0:
                                resp_len = min(resp_len, pos[0].item())
                    except Exception:
                        pass
                    total_response_tokens += resp_len

                    trunc_ids = new_ids
                    try:
                        if eos_id is not None:
                            pos = (new_ids == eos_id).nonzero(as_tuple=False)
                            if pos.numel() > 0:
                                trunc_ids = new_ids[:pos[0].item()]
                    except Exception:
                        pass
                    generated_text = tokenizer.decode(trunc_ids, skip_special_tokens=False)

                    pred_answer_raw = extract_answer_flexible(generated_text).strip()
                    gold_answer_raw = (row.get("answer") or "").strip()

                    def _parse_numeric(s: str) -> Optional[float]:
                        from fractions import Fraction
                        import re as _re
                        t = s.strip().rstrip(".")
                        if not t:
                            return None
                        m = _re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", t)
                        if m:
                            try:
                                return float(Fraction(int(m.group(1)), int(m.group(2))))
                            except Exception:
                                return None
                        t = _re.sub(r"(?<=\d),(?=\d)", "", t)
                        m = _re.fullmatch(r"[+-]?(?:\d+\.?\d*|\d*\.\d+)", t)
                        if m:
                            try:
                                return float(m.group(0))
                            except Exception:
                                return None
                        return None

                    pa = pred_answer_raw
                    ga = gold_answer_raw
                    em = 0
                    if ga:
                        ax = _parse_numeric(pa)
                        bx = _parse_numeric(ga)
                        if ax is not None and bx is not None:
                            if abs(ax - bx) <= 1e-8 * max(1.0, abs(ax), abs(bx)):
                                em = 1
                        elif pa.rstrip(".") == ga.rstrip("."):
                            em = 1

                    op = str(row.get("op", "unknown"))
                    if op not in per_op:
                        per_op[op] = {"total": 0, "correct": 0, "resp_tokens": 0}
                    per_op[op]["total"] += 1
                    per_op[op]["correct"] += em
                    per_op[op]["resp_tokens"] += resp_len
                    total += 1
                    correct += em

                    if return_examples:
                        prob = extract_problem_from_sequence(row.get("text", ""))
                        gen_solution = tokenizer.decode(trunc_ids, skip_special_tokens=False).strip()
                        collected.append({
                            "op": op,
                            "__idx": row.get("__idx"),
                            "problem": prob,
                            "gold_answer": gold_answer_raw,   # keep raw for validation utils
                            "gen_answer": pred_answer_raw,    # keep raw for validation utils
                            "gold_answer_raw": gold_answer_raw,
                            "gen_answer_raw": pred_answer_raw,
                            "resp_tokens": int(resp_len),
                            "gen_solution_answer": gen_solution,
                            "exact_match": em,
                            "template": row.get("template"),
                        })
        else:
            for j, row in enumerate(batch_rows):
                prompt_len = int(enc["attention_mask"][j].sum().item())
                new_ids = gen[j][prompt_len:]
                resp_len = len(new_ids)
                try:
                    if eos_id is not None:
                        pos = (new_ids == eos_id).nonzero(as_tuple=False)
                        if pos.numel() > 0:
                            resp_len = min(resp_len, pos[0].item())
                except Exception:
                    pass
                total_response_tokens += resp_len

                trunc_ids = new_ids
                try:
                    if eos_id is not None:
                        pos = (new_ids == eos_id).nonzero(as_tuple=False)
                        if pos.numel() > 0:
                            trunc_ids = new_ids[:pos[0].item()]
                except Exception:
                    pass
                generated_text = tokenizer.decode(trunc_ids, skip_special_tokens=False)

                pred_answer_raw = extract_answer_flexible(generated_text).strip()
                gold_answer_raw = (row.get("answer") or "").strip()

                def _parse_numeric(s: str) -> Optional[float]:
                    from fractions import Fraction
                    import re as _re
                    t = s.strip().rstrip(".")
                    if not t:
                        return None
                    m = _re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", t)
                    if m:
                        try:
                            return float(Fraction(int(m.group(1)), int(m.group(2))))
                        except Exception:
                            return None
                    t = _re.sub(r"(?<=\d),(?=\d)", "", t)
                    m = _re.fullmatch(r"[+-]?(?:\d+\.?\d*|\d*\.\d+)", t)
                    if m:
                        try:
                            return float(m.group(0))
                        except Exception:
                            return None
                    return None

                pa = pred_answer_raw
                ga = gold_answer_raw
                em = 0
                if ga:
                    ax = _parse_numeric(pa)
                    bx = _parse_numeric(ga)
                    if ax is not None and bx is not None:
                        if abs(ax - bx) <= 1e-8 * max(1.0, abs(ax), abs(bx)):
                            em = 1
                    elif pa.rstrip(".") == ga.rstrip("."):
                        em = 1

                op = str(row.get("op", "unknown"))
                if op not in per_op:
                    per_op[op] = {"total": 0, "correct": 0, "resp_tokens": 0}
                per_op[op]["total"] += 1
                per_op[op]["correct"] += em
                per_op[op]["resp_tokens"] += resp_len
                total += 1
                correct += em

                if return_examples:
                    prob = extract_problem_from_sequence(row.get("text", ""))
                    gen_solution = tokenizer.decode(trunc_ids, skip_special_tokens=False).strip()
                    collected.append({
                        "op": op,
                        "__idx": row.get("__idx"),
                        "problem": prob,
                        "gold_answer": gold_answer_raw,
                        "gen_answer": pred_answer_raw,
                        "gold_answer_raw": gold_answer_raw,
                        "gen_answer_raw": pred_answer_raw,
                        "resp_tokens": int(resp_len),
                        "gen_solution_answer": gen_solution,
                        "exact_match": em,
                        "template": row.get("template"),
                    })

    stats = {
        "count": total,
        "correct": correct,
        "avg_response_len": (total_response_tokens / total) if total else 0.0,
        "resp_tokens_sum": float(total_response_tokens),
        "per_op": per_op,
    }
    if return_examples:
        return stats, collected
    return stats


def eval_generation_vllm(
    model_path: Path,
    rows: List[Dict],
    max_new_tokens: int,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    sample_k: int = 1,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = -1,
    return_examples: bool = False,
    gpu_memory_utilization: Optional[float] = None,
    max_num_seqs: Optional[int] = None,
):
    if not _VLLM_AVAILABLE:
        raise RuntimeError("vLLM not installed. pip install vllm")
    resolved_dtype = _resolve_vllm_dtype(dtype)
    prompts = construct_prompts_from_rows(rows)
    max_num_seqs_adj: Optional[int] = None
    if max_num_seqs is not None:
        max_num_seqs_adj = max(sample_k, int(max_num_seqs))
    gpu_mem_adj: Optional[float] = None
    if gpu_memory_utilization is not None:
        gpu_mem_adj = float(max(0.1, min(gpu_memory_utilization, 0.99)))
    engine_kwargs = dict(
        model=str(model_path),
        tensor_parallel_size=max(1, int(tensor_parallel_size)),
        dtype=resolved_dtype,
        tokenizer=str(model_path),
        max_model_len=2048,
        enable_prefix_caching=True,
    )
    if gpu_mem_adj is not None:
        engine_kwargs["gpu_memory_utilization"] = gpu_mem_adj
    if max_num_seqs_adj is not None:
        engine_kwargs["max_num_seqs"] = int(max_num_seqs_adj)
    engine = LLM(**engine_kwargs)
    sampling_kwargs = dict(
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_new_tokens),
        n=int(sample_k),
        # use_parallel_sampling=True if int(sample_k) > 1 else False,
        skip_special_tokens=False,
        stop=["</answer>"],
        # stop_token_ids=[eos_id, tokenizer_eos],  # add if you resolve ids here
    )
    if isinstance(top_k, int) and top_k >= 0:
        sampling_kwargs["top_k"] = int(top_k)
    sampling = SamplingParams(**sampling_kwargs)

    outs = engine.generate(prompts, sampling)
    total = 0
    correct = 0
    total_response_tokens = 0
    per_op: Dict[str, Dict[str, int]] = {}
    collected: List[Dict] = []

    for out, row in zip(outs, rows):
        for cand in out.outputs:
            resp_text = cand.text
            resp_tok_ct = len(getattr(cand, "token_ids", []) or [])

            pred_answer_raw = extract_answer_flexible(resp_text).strip()
            # Use the same gold field as HF path for consistency
            gold_answer_raw = (row['solution'].split("Answer: ")[-1]).strip().rstrip(".")

            def _parse_numeric(s: str) -> Optional[float]:
                from fractions import Fraction
                import re as _re
                t = s.strip().rstrip(".")
                if not t:
                    return None
                m = _re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", t)
                if m:
                    try:
                        return float(Fraction(int(m.group(1)), int(m.group(2))))
                    except Exception:
                        return None
                t = _re.sub(r"(?<=\d),(?=\d)", "", t)
                m = _re.fullmatch(r"[+-]?(?:\d+\.?\d*|\d*\.\d+)", t)
                if m:
                    try:
                        return float(m.group(0))
                    except Exception:
                        return None
                return None

            pa = pred_answer_raw
            ga = gold_answer_raw
            em = 0
            if ga:
                ax = _parse_numeric(pa)
                bx = _parse_numeric(ga)
                if ax is not None and bx is not None:
                    if abs(ax - bx) <= 1e-8 * max(1.0, abs(ax), abs(bx)):
                        em = 1
                elif pa.rstrip(".") == ga.rstrip("."):
                    em = 1

            op = str(row.get("op", "unknown"))
            d = per_op.setdefault(op, {"total": 0, "correct": 0, "resp_tokens": 0})
            d["total"] += 1
            d["correct"] += em
            d["resp_tokens"] += int(resp_tok_ct)
            total += 1
            correct += em
            total_response_tokens += int(resp_tok_ct)

            if return_examples:
                collected.append({
                    "op": op,
                    "__idx": row.get("__idx"),
                    "prompt": out.prompt,
                    "gold_answer": gold_answer_raw,
                    "gen_answer": pred_answer_raw,
                    "gold_answer_raw": gold_answer_raw,
                    "gen_answer_raw": pred_answer_raw,
                    "gen_solution_answer": resp_text.strip(),
                    "resp_tokens": int(resp_tok_ct),
                    "exact_match": em,
                    "template": row.get("template"),
                })

    stats = {
        "count": total,
        "correct": correct,
        "avg_response_len": (total_response_tokens / total) if total else 0.0,
        "resp_tokens_sum": float(total_response_tokens),
        "per_op": per_op,
    }
    if return_examples:
        return stats, collected
    return stats


def _split_rows_evenly(rows: List[Dict], num_splits: int) -> List[List[Dict]]:
    if num_splits <= 1:
        return [rows]
    shards: List[List[Dict]] = [[] for _ in range(num_splits)]
    for idx, row in enumerate(rows):
        shards[idx % num_splits].append(row)
    return shards


def _eval_generation_vllm_worker(payload):
    device_id = payload["device_id"]
    model_path_str = payload["model_path"]
    rows = payload["rows"]
    max_new_tokens = payload["max_new_tokens"]
    dtype = payload["dtype"]
    sample_k = payload["sample_k"]
    temperature = payload["temperature"]
    top_p = payload["top_p"]
    top_k = payload["top_k"]
    return_examples = payload["return_examples"]
    gpu_memory_utilization = payload["gpu_memory_utilization"]
    max_num_seqs = payload.get("max_num_seqs")
    task_id = payload["task_id"]
    # Restrict vLLM to the designated device for this worker.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ["VLLM_TENSOR_PARALLEL_SIZE"] = "1"
    # Ensure torch picks up the remapped device 0 for this worker.
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
    res = eval_generation_vllm(
        Path(model_path_str),
        rows,
        max_new_tokens,
        tensor_parallel_size=1,
        dtype=dtype,
        sample_k=sample_k,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        return_examples=return_examples,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
    )
    if isinstance(res, tuple):
        stats, examples = res
    else:
        stats, examples = res, []
    return task_id, stats, examples


def _eval_generation_vllm_worker_entry(payload, out_queue):
    task_id = payload.get("task_id")
    try:
        task_id_ret, stats, examples = _eval_generation_vllm_worker(payload)
        out_queue.put(("ok", task_id_ret, stats, examples))
    except Exception as exc:
        out_queue.put(("err", task_id, repr(exc), traceback.format_exc()))


def _merge_stats_dicts(dicts: List[Dict]) -> Dict:
    merged = {
        "count": 0,
        "correct": 0,
        "resp_tokens_sum": 0.0,
        "avg_response_len": 0.0,
        "per_op": {},
    }
    per_op_inputs: List[Dict[str, Dict[str, int]]] = []
    for d in dicts:
        if not d:
            continue
        merged["count"] += int(d.get("count", 0))
        merged["correct"] += int(d.get("correct", 0))
        resp_sum = d.get("resp_tokens_sum")
        if resp_sum is None:
            resp_sum = float(d.get("avg_response_len", 0.0)) * int(d.get("count", 0))
        merged["resp_tokens_sum"] += float(resp_sum)
        per_op_inputs.append(d.get("per_op", {}))
    if merged["count"]:
        merged["avg_response_len"] = merged["resp_tokens_sum"] / merged["count"]
    if per_op_inputs:
        merged["per_op"] = merge_op_counts(per_op_inputs)
    return merged


def eval_generation_vllm_data_parallel(
    model_path: Path,
    rows: List[Dict],
    max_new_tokens: int,
    dtype: str,
    sample_k: int,
    temperature: float,
    top_p: float,
    top_k: int,
    return_examples: bool,
    devices: List[str],
    gpu_memory_utilization: float,
    max_num_seqs: Optional[int],
    cache_dir: Optional[Path] = None,
):
    usable_devices = [str(d).strip() for d in devices if str(d).strip()]
    if len(usable_devices) <= 1:
        return eval_generation_vllm(
            model_path,
            rows,
            max_new_tokens,
            tensor_parallel_size=1,
            dtype=dtype,
            sample_k=sample_k,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            return_examples=return_examples,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
        )

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    shards = _split_rows_evenly(rows, len(usable_devices))
    task_params: Dict[str, Dict] = {}
    task_cache_paths: Dict[str, Path] = {}
    worker_results: List[Tuple[str, Dict, List[Dict]]] = []
    pending_tasks: List[Dict] = []

    for shard_idx, (device_id, shard_rows) in enumerate(zip(usable_devices, shards)):
        if not shard_rows:
            continue
        task_id = f"shard{shard_idx}"
        cache_path: Optional[Path] = None
        if cache_dir is not None:
            cache_path = cache_dir / f"{task_id}.json"
        rows_signature = [str(r.get("__idx", idx)) for idx, r in enumerate(shard_rows)]
        params = {
            "model_path": str(model_path),
            "max_new_tokens": int(max_new_tokens),
            "dtype": str(dtype),
            "sample_k": int(sample_k),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "return_examples": bool(return_examples),
            "gpu_memory_utilization": None if gpu_memory_utilization is None else float(gpu_memory_utilization),
            "max_num_seqs": None if max_num_seqs is None else int(max_num_seqs),
            "rows_signature": rows_signature,
        }
        task_params[task_id] = params

        cache_hit = False
        if cache_path is not None and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("version") == 1 and cached.get("params") == params:
                    stats = cached.get("stats", {})
                    examples = cached.get("examples", []) if return_examples else []
                    worker_results.append((task_id, stats, examples))
                    cache_hit = True
            except Exception:
                cache_hit = False

        if cache_hit:
            continue

        payload = {
            "task_id": task_id,
            "device_id": device_id,
            "model_path": str(model_path),
            "rows": shard_rows,
            "max_new_tokens": max_new_tokens,
            "dtype": dtype,
            "sample_k": sample_k,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "return_examples": return_examples,
            "gpu_memory_utilization": None if gpu_memory_utilization is None else float(gpu_memory_utilization),
            "max_num_seqs": max_num_seqs,
        }
        pending_tasks.append(payload)
        if cache_path is not None:
            task_cache_paths[task_id] = cache_path

    if not pending_tasks:
        stats_list = [res[1] for res in worker_results]
        merged_stats = _merge_stats_dicts(stats_list)
        if return_examples:
            merged_examples: List[Dict] = []
            for _, _, ex in worker_results:
                merged_examples.extend(ex)
            return merged_stats, merged_examples
        return merged_stats

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes: List[mp.Process] = []
    for payload in pending_tasks:
        p = ctx.Process(target=_eval_generation_vllm_worker_entry, args=(payload, result_queue))
        p.daemon = False
        p.start()
        processes.append(p)

    remaining = len(processes)
    try:
        while remaining > 0:
            try:
                status, task_id, payload1, payload2 = result_queue.get(timeout=60.0)
            except queue.Empty:
                for proc in processes:
                    if proc.exitcode is not None and proc.exitcode != 0:
                        for other in processes:
                            if other.is_alive():
                                other.terminate()
                        raise RuntimeError(
                            f"vLLM worker {proc.pid} exited with code {proc.exitcode} before returning results"
                        )
                continue

            if status == "err":
                for proc in processes:
                    if proc.is_alive():
                        proc.terminate()
                raise RuntimeError(f"vLLM worker {task_id} failed: {payload1}\n{payload2}")

            worker_results.append((task_id, payload1, payload2))
            cache_path = task_cache_paths.get(task_id)
            if cache_path is not None:
                try:
                    cache_record = {
                        "version": 1,
                        "params": task_params[task_id],
                        "stats": payload1,
                    }
                    if return_examples:
                        cache_record["examples"] = payload2
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache_record, f, ensure_ascii=False)
                except Exception as cache_exc:
                    print(f"[warn] Failed to write cache for {cache_path}: {cache_exc}")

            remaining -= 1
    finally:
        for proc in processes:
            proc.join()
        result_queue.close()

    stats_list = [res[1] for res in worker_results]
    merged_stats = _merge_stats_dicts(stats_list)

    if return_examples:
        merged_examples: List[Dict] = []
        for _, _, ex in worker_results:
            merged_examples.extend(ex)
        return merged_stats, merged_examples
    return merged_stats


@torch.no_grad()
def eval_loss(model, dataset: Dataset, batch_size: int, device: torch.device) -> Tuple[float, int]:
    sampler = DistributedSampler(dataset, shuffle=False) if dist.is_available() and dist.is_initialized() else None
    dl = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    total_examples = 0
    sum_loss_x_n = 0.0
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss.detach().float().item()
        n = batch["input_ids"].shape[0]
        sum_loss_x_n += loss * n
        total_examples += n
    return sum_loss_x_n, total_examples


@torch.no_grad()
def eval_loss_per_op(model, dataset: Dataset, batch_size: int, device: torch.device):
    sampler = DistributedSampler(dataset, shuffle=False) if dist.is_available() and dist.is_initialized() else None

    def collate_with_op(batch):
        items = [b[0] for b in batch]
        ops = [b[1] for b in batch]
        keys = items[0].keys()
        out = {k: torch.stack([it[k] for it in items], dim=0) for k in keys}
        return out, ops

    dl = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, collate_fn=collate_with_op)
    total_examples = 0
    sum_loss_x_n = 0.0
    per_op: Dict[str, Dict[str, float]] = {}

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    iterator = tqdm(dl, total=len(dl), desc="loss", disable=(rank != 0))
    for batch, ops in iterator:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss.detach().float().item()
        n = batch["input_ids"].shape[0]
        sum_loss_x_n += loss * n
        total_examples += n

        logits = out.logits  # (B, T, V)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch["labels"][:, 1:].contiguous()
        mask = batch["attention_mask"][:, 1:].contiguous()
        B, Tm1, V = shift_logits.size(0), shift_logits.size(1), shift_logits.size(2)
        loss_tok = F.cross_entropy(
            shift_logits.view(-1, V), shift_labels.view(-1), reduction='none'
        ).view(B, Tm1)
        loss_tok = loss_tok * mask
        sum_per_ex = loss_tok.sum(dim=1)  # (B,)
        tok_per_ex = mask.sum(dim=1).clamp_min(1)
        mean_per_ex = (sum_per_ex / tok_per_ex).detach().float().cpu().tolist()
        for op, l in zip(ops, mean_per_ex):
            d = per_op.setdefault(op, {"loss_sum": 0.0, "count": 0})
            d["loss_sum"] += float(l)
            d["count"] += 1

    return sum_loss_x_n, total_examples, per_op


def reduce_scalar(sum_val: float) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return sum_val
    t = torch.tensor([sum_val], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints-root", default="trainer/difficulty/model_output")
    ap.add_argument("--checkpoints-pattern", default="checkpoint-*")
    ap.add_argument("--data-root", default="data/split/difficulty/zero_context_medium/difficulty-5B")
    ap.add_argument(
        "--id-op-threshold",
        type=int,
        default=10,
        help="Ops <= threshold treated as ID when loading per-op jsonl directories; set <0 to treat all ops as ID.",
    )
    ap.add_argument("--gen-batch-size", type=int, default=1)
    ap.add_argument("--loss-batch-size", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--output-dir", default="results/eval_difficulty")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--gen-backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--skip-loss", action="store_true")
    # Sampling / multi-response options
    ap.add_argument("--sample-k", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--val-top-p", type=float, default=0.7)  # kept for downstream compatibility
    ap.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    ap.add_argument("--vllm-dtype", default="auto")
    ap.add_argument(
        "--vllm-data-parallel-size",
        type=int,
        default=0,
        help="How many vLLM workers to launch for data-parallel prompt evaluation (0=auto).",
    )
    ap.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction of GPU memory vLLM may claim; leave unset to use vLLM's default.",
    )
    ap.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=None,
        help="Maximum concurrent sequences for vLLM (default: heuristic based on --sample-k).",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summary-filename", default="summary")
    ap.add_argument(
        "--group-by-template",
        nargs="+",
        default=[
            "crazy_zootopia",
            "teachers_in_school",
            "movie_festival_awards",
        ],
        help="Aggregate final metrics by template values (default: crazy_zootopia, teachers_in_school, movie_festival_awards)",
    )
    args = ap.parse_args()

    # Device / dist
    rank, world_size, local_rank = init_dist_if_needed()
    if args.device == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = (args.device == "cuda")
    if use_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda" if use_cuda else "cpu")
    if rank == 0:
        print(f"[eval] world_size={world_size} device={device}")

    vllm_data_parallel_devices: List[str] = []
    if args.gen_backend == "vllm":
        visible_devices = get_visible_cuda_devices()
        if args.vllm_data_parallel_size > 0:
            data_parallel_size = args.vllm_data_parallel_size
        else:
            data_parallel_size = len(visible_devices) if visible_devices else 1
        if (args.vllm_tensor_parallel_size or 1) > 1 and data_parallel_size > 1:
            if rank == 0:
                print("[warn] Disabling vLLM data parallel because tensor parallel > 1")
            data_parallel_size = 1
        if visible_devices and data_parallel_size > len(visible_devices):
            if rank == 0:
                print(
                    f"[warn] Requested vLLM data parallel size {data_parallel_size} but only "
                    f"{len(visible_devices)} devices visible; clipping."
                )
            data_parallel_size = len(visible_devices)
        if data_parallel_size > 1 and visible_devices:
            vllm_data_parallel_devices = visible_devices[:data_parallel_size]
        if rank == 0:
            if vllm_data_parallel_devices:
                print(
                    f"[vllm] data_parallel_size={len(vllm_data_parallel_devices)} "
                    f"devices={','.join(vllm_data_parallel_devices)}"
                )
            else:
                print("[vllm] data_parallel_size=1")

    data_root = Path(args.data_root)
    id_threshold = args.id_op_threshold
    if id_threshold is not None and id_threshold < 0:
        id_threshold = None

    test_id_rows, test_ood_rows = load_eval_rows(data_root, id_threshold)

    if rank == 0:
        print(
            f"[data] Loaded ID={len(test_id_rows)} OOD={len(test_ood_rows)} "
            f"rows from {data_root}"
        )
        if test_id_rows:
            id_ops_present = sorted({str(r.get("op", "unknown")) for r in test_id_rows})
            print(f"        ID ops: {', '.join(id_ops_present)}")
        if test_ood_rows:
            ood_ops_present = sorted({str(r.get("op", "unknown")) for r in test_ood_rows})
            print(f"        OOD ops: {', '.join(ood_ops_present)}")

    ck_root = Path(args.checkpoints_root)
    checkpoints = list(reversed(find_checkpoints(ck_root, args.checkpoints_pattern)))
    if rank == 0:
        print(f"[ckpt] Found {len(checkpoints)} checkpoints under {ck_root}")
    if not checkpoints:
        if rank == 0:
            print("No checkpoints found. Exiting.")
        return

    out_root = Path(args.output_dir)
    if rank == 0:
        out_root.mkdir(parents=True, exist_ok=True)

    summary_records: List[Dict] = []
    ck_iter = tqdm(checkpoints, desc="checkpoints", disable=(rank != 0))

    expect_id_outputs = len(test_id_rows) > 0
    expect_ood_outputs = len(test_ood_rows) > 0

    def existing_outputs_ready(checkpoint_name: str) -> Tuple[bool, List[Path]]:
        candidates: List[Path] = [out_root / f"{checkpoint_name}_metrics.json"]
        if expect_id_outputs:
            candidates.append(out_root / f"{checkpoint_name}_id_generations.jsonl")
        if expect_ood_outputs:
            candidates.append(out_root / f"{checkpoint_name}_ood_generations.jsonl")
        missing = [path for path in candidates if not path.exists()]
        return len(missing) == 0, missing

    for ck in ck_iter:
        all_outputs_present, missing_outputs = existing_outputs_ready(ck.name)

        if all_outputs_present:
            if rank == 0:
                print(f"[skip] {ck.name} results already present in {out_root}")
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            continue

        if args.resume:
            metrics_path = out_root / f"{ck.name}_metrics.json"
            if metrics_path.exists():
                if rank == 0:
                    print(f"[resume] Skip {ck.name} (found {metrics_path})")
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                continue

        if rank == 0:
            print(f"\n=== Evaluating {ck.name} ===")

        model = None
        tokenizer = None
        if args.gen_backend == "hf":
            model, tokenizer = load_model_and_tokenizer(ck, device)

        # Tag original indices for pass@k-style aggregation
        for i, r in enumerate(test_id_rows):
            if "__idx" not in r:
                r["__idx"] = i
        for i, r in enumerate(test_ood_rows):
            if "__idx" not in r:
                r["__idx"] = i

        def shard_rows(rows: List[Dict]) -> List[Dict]:
            if not (dist.is_available() and dist.is_initialized()):
                return rows
            if args.gen_backend == "vllm":
                return rows if rank == 0 else []
            return [rows[i] for i in range(rank, len(rows), world_size)]

        shard_id_rows = shard_rows(test_id_rows)
        shard_ood_rows = shard_rows(test_ood_rows)

        id_gen_stats = {"count": 0, "correct": 0, "avg_response_len": 0.0, "per_op": {}}
        ood_gen_stats = {"count": 0, "correct": 0, "avg_response_len": 0.0, "per_op": {}}
        id_examples_part: List[Dict] = []
        ood_examples_part: List[Dict] = []

        ck_cache_root = out_root / ".cache" / ck.name

        if shard_id_rows:
            if args.gen_backend == "vllm":
                if vllm_data_parallel_devices:
                    res = eval_generation_vllm_data_parallel(
                        ck,
                        shard_id_rows,
                        args.max_new_tokens,
                        dtype=args.vllm_dtype,
                        sample_k=args.sample_k,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        return_examples=True,
                        devices=vllm_data_parallel_devices,
                        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                        max_num_seqs=args.vllm_max_num_seqs,
                        cache_dir=ck_cache_root / "id",
                    )
                else:
                    res = eval_generation_vllm(
                        ck,
                        shard_id_rows,
                        args.max_new_tokens,
                        tensor_parallel_size=args.vllm_tensor_parallel_size,
                        dtype=args.vllm_dtype,
                        sample_k=args.sample_k,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        return_examples=True,
                        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                        max_num_seqs=args.vllm_max_num_seqs,
                    )
            else:
                res = eval_generation(
                    model, tokenizer, shard_id_rows, device,
                    args.gen_batch_size, args.max_new_tokens,
                    sample_k=args.sample_k,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    return_examples=True,
                )
            id_gen_stats, id_examples_part = res

        if shard_ood_rows:
            if args.gen_backend == "vllm":
                if vllm_data_parallel_devices:
                    res = eval_generation_vllm_data_parallel(
                        ck,
                        shard_ood_rows,
                        args.max_new_tokens,
                        dtype=args.vllm_dtype,
                        sample_k=args.sample_k,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        return_examples=True,
                        devices=vllm_data_parallel_devices,
                        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                        max_num_seqs=args.vllm_max_num_seqs,
                        cache_dir=ck_cache_root / "ood",
                    )
                else:
                    res = eval_generation_vllm(
                        ck,
                        shard_ood_rows,
                        args.max_new_tokens,
                        tensor_parallel_size=args.vllm_tensor_parallel_size,
                        dtype=args.vllm_dtype,
                        sample_k=args.sample_k,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        return_examples=True,
                        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                        max_num_seqs=args.vllm_max_num_seqs,
                    )
            else:
                res = eval_generation(
                    model, tokenizer, shard_ood_rows, device,
                    args.gen_batch_size, args.max_new_tokens,
                    sample_k=args.sample_k,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    return_examples=True,
                )
            ood_gen_stats, ood_examples_part = res

        # Reduce basic scalars
        for key in ("count", "correct"):
            id_gen_stats[key] = int(reduce_scalar(float(id_gen_stats.get(key, 0))))
            ood_gen_stats[key] = int(reduce_scalar(float(ood_gen_stats.get(key, 0))))
        # Weighted average for response length
        id_resp_tokens_sum = id_gen_stats.get("resp_tokens_sum")
        if id_resp_tokens_sum is None:
            id_resp_tokens_sum = id_gen_stats.get("avg_response_len", 0.0) * (id_gen_stats.get("count", 0) or 1)
        ood_resp_tokens_sum = ood_gen_stats.get("resp_tokens_sum")
        if ood_resp_tokens_sum is None:
            ood_resp_tokens_sum = ood_gen_stats.get("avg_response_len", 0.0) * (ood_gen_stats.get("count", 0) or 1)
        id_resp_tokens_sum = reduce_scalar(id_resp_tokens_sum)
        ood_resp_tokens_sum = reduce_scalar(ood_resp_tokens_sum)

        # Per-op maps via all_gather_object
        id_per_op_gather = distributed_all_gather_dict(id_gen_stats.get("per_op", {}))
        ood_per_op_gather = distributed_all_gather_dict(ood_gen_stats.get("per_op", {}))

        # Loss eval (HF)
        do_loss = (not args.skip_loss)
        id_loss_sum, id_n = (0.0, 0)
        ood_loss_sum, ood_n = (0.0, 0)
        id_loss_per_op_local: Dict[str, Dict[str, float]] = {}
        ood_loss_per_op_local: Dict[str, Dict[str, float]] = {}
        if do_loss:
            if model is None or tokenizer is None:
                model, tokenizer = load_model_and_tokenizer(ck, device)
            max_len = safe_max_length(tokenizer)
            id_ds = LMFullWithOpDataset(test_id_rows, tokenizer, max_length=max_len) if test_id_rows else None
            ood_ds = LMFullWithOpDataset(test_ood_rows, tokenizer, max_length=max_len) if test_ood_rows else None
            if id_ds is not None and len(id_ds) > 0:
                s, n, per_op_loss = eval_loss_per_op(model, id_ds, args.loss_batch_size, device)
                id_loss_sum += s; id_n += n; id_loss_per_op_local = per_op_loss
            if ood_ds is not None and len(ood_ds) > 0:
                s, n, per_op_loss = eval_loss_per_op(model, ood_ds, args.loss_batch_size, device)
                ood_loss_sum += s; ood_n += n; ood_loss_per_op_local = per_op_loss

            id_loss_sum = reduce_scalar(id_loss_sum)
            id_n = int(reduce_scalar(float(id_n)))
            ood_loss_sum = reduce_scalar(ood_loss_sum)
            ood_n = int(reduce_scalar(float(ood_n)))

        if model is not None and args.gen_backend == "vllm":
            try:
                del model
                torch.cuda.empty_cache()
            except Exception:
                pass
            model = None

        # Merge per-op counts across ranks
        if rank == 0:
            id_per_op = merge_op_counts(id_per_op_gather)
            ood_per_op = merge_op_counts(ood_per_op_gather)
        else:
            id_per_op = {}
            ood_per_op = {}

        # Gather per-op loss stats
        id_loss_per_op_gather = distributed_all_gather_dict(id_loss_per_op_local)
        ood_loss_per_op_gather = distributed_all_gather_dict(ood_loss_per_op_local)

        def merge_op_loss(dicts: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
            out: Dict[str, Dict[str, float]] = {}
            for d in dicts:
                for op, v in d.items():
                    cur = out.setdefault(op, {"loss_sum": 0.0, "count": 0.0})
                    cur["loss_sum"] += float(v.get("loss_sum", 0.0))
                    cur["count"] += float(v.get("count", 0.0))
            return out

        if rank == 0:
            id_per_op_loss = merge_op_loss(id_loss_per_op_gather)
            ood_per_op_loss = merge_op_loss(ood_loss_per_op_gather)
        else:
            id_per_op_loss = {}
            ood_per_op_loss = {}

        # Gather examples across ranks (for validation metrics / pass@k-style)
        if dist.is_available() and dist.is_initialized():
            gathered_id_parts: List[List[Dict]] = [None for _ in range(world_size)]  # type: ignore
            gathered_ood_parts: List[List[Dict]] = [None for _ in range(world_size)]  # type: ignore
            dist.all_gather_object(gathered_id_parts, id_examples_part)
            dist.all_gather_object(gathered_ood_parts, ood_examples_part)
            id_examples_all = [e for part in gathered_id_parts for e in (part or [])]
            ood_examples_all = [e for part in gathered_ood_parts for e in (part or [])]
        else:
            id_examples_all = id_examples_part
            ood_examples_all = ood_examples_part

        # ---- Rank 0: finalize metrics, write files ----
        if rank == 0:
            # Sort examples by original index (grouping helps validation utils)
            def sort_by_idx(records: List[Dict]) -> List[Dict]:
                try:
                    return sorted(records, key=lambda r: (r.get("__idx") is None, r.get("__idx", 0)))
                except Exception:
                    return records

            id_examples_all_sorted = sort_by_idx(id_examples_all)
            ood_examples_all_sorted = sort_by_idx(ood_examples_all)

            id_counts, id_per_op_counts = _aggregate_pass_counts(id_examples_all_sorted, key_prefix="id:")
            ood_counts, ood_per_op_counts = _aggregate_pass_counts(ood_examples_all_sorted, key_prefix="ood:")

            id_pass_series = _compute_pass_series(id_counts.values(), args.sample_k)
            ood_pass_series = _compute_pass_series(ood_counts.values(), args.sample_k)

            id_per_op_pass_series = {
                op: _compute_pass_series(op_counts.values(), args.sample_k)
                for op, op_counts in id_per_op_counts.items()
            }
            ood_per_op_pass_series = {
                op: _compute_pass_series(op_counts.values(), args.sample_k)
                for op, op_counts in ood_per_op_counts.items()
            }

            total_counts = {**id_counts, **ood_counts}
            total_per_op_count_map: Dict[str, Dict[str, Tuple[int, int]]] = {}
            for src in (id_per_op_counts, ood_per_op_counts):
                for op, op_counts in src.items():
                    total_per_op_count_map.setdefault(op, {}).update(op_counts)
            total_pass_series = _compute_pass_series(total_counts.values(), args.sample_k)
            total_per_op_pass_series = {
                op: _compute_pass_series(op_counts.values(), args.sample_k)
                for op, op_counts in total_per_op_count_map.items()
            }

            id_val_metrics: Dict[str, Any] = {}
            ood_val_metrics: Dict[str, Any] = {}

            # Scalar aggregates
            id_avg_loss = (id_loss_sum / id_n) if id_n > 0 else 0.0
            ood_avg_loss = (ood_loss_sum / ood_n) if ood_n > 0 else 0.0
            id_avg_resp_len = (id_resp_tokens_sum / id_gen_stats["count"]) if id_gen_stats["count"] else 0.0
            ood_avg_resp_len = (ood_resp_tokens_sum / ood_gen_stats["count"]) if ood_gen_stats["count"] else 0.0
            id_acc = (id_gen_stats["correct"] / id_gen_stats["count"]) if id_gen_stats["count"] else 0.0
            ood_acc = (ood_gen_stats["correct"] / ood_gen_stats["count"]) if ood_gen_stats["count"] else 0.0

            total_correct = id_gen_stats["correct"] + ood_gen_stats["correct"]
            total_count = id_gen_stats["count"] + ood_gen_stats["count"]
            total_acc = (total_correct / total_count) if total_count else 0.0
            total_avg_loss = 0.0
            total_loss_n = id_n + ood_n
            if total_loss_n > 0:
                total_avg_loss = (id_loss_sum + ood_loss_sum) / total_loss_n

            # Per-op helper mappers
            def per_op_metrics(op_counts: Dict[str, Dict[str, int]]):
                return {op: (v["correct"] / v["total"]) if v["total"] else 0.0 for op, v in op_counts.items()}

            def per_op_avg_resp_len(op_counts: Dict[str, Dict[str, int]]):
                return {op: (v.get("resp_tokens", 0) / v["total"]) if v["total"] else 0.0 for op, v in op_counts.items()}

            def per_op_avg_loss(op_loss: Dict[str, Dict[str, float]]):
                return {op: (v.get("loss_sum", 0.0) / v.get("count", 1.0)) if v.get("count", 0.0) > 0 else 0.0
                        for op, v in op_loss.items()}

            total_per_op_counts = merge_op_counts([id_per_op, ood_per_op])
            total_per_op = per_op_metrics(total_per_op_counts)
            total_per_op_resp_len = per_op_avg_resp_len(total_per_op_counts)

            def merge_two_loss(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]]):
                out: Dict[str, Dict[str, float]] = {}
                for src in (a, b):
                    for op, v in src.items():
                        cur = out.setdefault(op, {"loss_sum": 0.0, "count": 0.0})
                        cur["loss_sum"] += float(v.get("loss_sum", 0.0))
                        cur["count"] += float(v.get("count", 0.0))
                return out
            total_per_op_loss_map = merge_two_loss(id_per_op_loss, ood_per_op_loss)
            total_per_op_loss = per_op_avg_loss(total_per_op_loss_map)

            # Compose metrics dict (keep legacy keys for compatibility)
            metrics = {
                "checkpoint": ck.name,
                "id": {
                    "answer_accuracy": id_acc,
                    "avg_response_len": id_avg_resp_len,
                    "avg_loss": id_avg_loss,
                    "count": id_gen_stats["count"],
                    "validation_metrics": id_val_metrics,
                    "pass_at_k": _format_pass(id_pass_series),
                    "per_op_pass_at_k": {op: _format_pass(series) for op, series in id_per_op_pass_series.items()},
                    "per_op_accuracy": per_op_metrics(id_per_op),
                    "per_op_avg_response_len": per_op_avg_resp_len(id_per_op),
                    "per_op_avg_loss": per_op_avg_loss(id_per_op_loss),
                    "per_op_length": per_op_avg_resp_len(id_per_op),  # alias
                    "per_op_loss": per_op_avg_loss(id_per_op_loss),    # alias
                },
                "ood": {
                    "answer_accuracy": ood_acc,
                    "avg_response_len": ood_avg_resp_len,
                    "avg_loss": ood_avg_loss,
                    "count": ood_gen_stats["count"],
                    "validation_metrics": ood_val_metrics,
                    "pass_at_k": _format_pass(ood_pass_series),
                    "per_op_pass_at_k": {op: _format_pass(series) for op, series in ood_per_op_pass_series.items()},
                    "per_op_accuracy": per_op_metrics(ood_per_op),
                    "per_op_avg_response_len": per_op_avg_resp_len(ood_per_op),
                    "per_op_avg_loss": per_op_avg_loss(ood_per_op_loss),
                    "per_op_length": per_op_avg_resp_len(ood_per_op),  # alias
                    "per_op_loss": per_op_avg_loss(ood_per_op_loss),   # alias
                },
                "total": {
                    "answer_accuracy": total_acc,
                    "avg_loss": total_avg_loss,
                    "count": total_count,
                    "validation_metrics": {
                        "id": id_val_metrics,
                        "ood": ood_val_metrics,
                    },
                    "pass_at_k": _format_pass(total_pass_series),
                    "per_op_pass_at_k": {op: _format_pass(series) for op, series in total_per_op_pass_series.items()},
                    "per_op_accuracy": total_per_op,
                    "per_op_avg_response_len": total_per_op_resp_len,
                    "per_op_avg_loss": total_per_op_loss,
                    "per_op_length": total_per_op_resp_len,  # alias
                    "per_op_loss": total_per_op_loss,        # alias
                },
            }

            template_metrics_by_split: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
            template_group_names = list(dict.fromkeys(args.group_by_template)) if args.group_by_template else []
            if template_group_names:
                template_set = set(template_group_names)

                def _group_by_template(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
                    grouped: Dict[str, List[Dict[str, Any]]] = {name: [] for name in template_group_names}
                    for rec in records:
                        tpl = rec.get("template")
                        if tpl in template_set:
                            grouped[tpl].append(rec)
                    return grouped

                def _template_metrics(records: List[Dict[str, Any]], key_prefix: str) -> Dict[str, Any]:
                    counts, per_op_counts = _aggregate_pass_counts(records, key_prefix=key_prefix)
                    pass_series = _compute_pass_series(counts.values(), args.sample_k)
                    per_op_pass_at_k = {
                        op: _format_pass(_compute_pass_series(op_counts.values(), args.sample_k))
                        for op, op_counts in per_op_counts.items()
                    }
                    resp_tokens_sum = sum(int(rec.get("resp_tokens", 0) or 0) for rec in records)
                    correct = sum(int(rec.get("exact_match", 0) or 0) for rec in records)
                    total_examples = len(records)
                    avg_resp_len = (resp_tokens_sum / total_examples) if total_examples else 0.0
                    accuracy = (correct / total_examples) if total_examples else 0.0
                    return {
                        "count": total_examples,
                        "correct": correct,
                        "answer_accuracy": accuracy,
                        "avg_response_len": avg_resp_len,
                        "resp_tokens_sum": resp_tokens_sum,
                        "pass_at_k": _format_pass(pass_series),
                        "per_op_pass_at_k": per_op_pass_at_k,
                    }

                id_grouped = _group_by_template(id_examples_all_sorted)
                ood_grouped = _group_by_template(ood_examples_all_sorted)
                template_metrics_by_split = {"id": {}, "ood": {}, "total": {}}
                for name in template_group_names:
                    id_metrics = _template_metrics(id_grouped.get(name, []), key_prefix=f"id:{name}:")
                    ood_metrics = _template_metrics(ood_grouped.get(name, []), key_prefix=f"ood:{name}:")
                    combined_records: List[Dict[str, Any]] = []
                    for rec in id_grouped.get(name, []):
                        new_rec = dict(rec)
                        new_rec["__idx"] = f"id_{rec.get('__idx')}"
                        combined_records.append(new_rec)
                    for rec in ood_grouped.get(name, []):
                        new_rec = dict(rec)
                        new_rec["__idx"] = f"ood_{rec.get('__idx')}"
                        combined_records.append(new_rec)
                    total_metrics = _template_metrics(combined_records, key_prefix=f"total:{name}:")
                    template_metrics_by_split["id"][name] = id_metrics
                    template_metrics_by_split["ood"][name] = ood_metrics
                    template_metrics_by_split["total"][name] = total_metrics

                metrics["id"]["per_template"] = template_metrics_by_split["id"]
                metrics["ood"]["per_template"] = template_metrics_by_split["ood"]
                metrics["total"]["per_template"] = template_metrics_by_split["total"]

            print(f"[summary] {ck.name} | "
                  f"ID acc={id_acc:.4f} resp_len={id_avg_resp_len:.1f} loss={id_avg_loss:.4f} | "
                  f"OOD acc={ood_acc:.4f} resp_len={ood_avg_resp_len:.1f} loss={ood_avg_loss:.4f} | "
                  f"Total acc={total_acc:.4f}")

            # Write metrics JSON
            out_file = out_root / f"{ck.name}_metrics.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"[write] {out_file}")

            # Write compact generations
            def write_jsonl(path: Path, records: List[Dict]):
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as wf:
                    for r in records:
                        wf.write(json.dumps(r, ensure_ascii=False) + "\n")

            write_jsonl(out_root / f"{ck.name}_id_generations.jsonl", id_examples_all_sorted)
            write_jsonl(out_root / f"{ck.name}_ood_generations.jsonl", ood_examples_all_sorted)

            # Append row for master summary
            summary_record = {
                "checkpoint": ck.name,
                "id_acc": float(id_acc),
                "ood_acc": float(ood_acc),
                "total_acc": float(total_acc),
                "id_avg_loss": float(id_avg_loss),
                "ood_avg_loss": float(ood_avg_loss),
                "total_avg_loss": float(total_avg_loss),
                "id_avg_resp_len": float(id_avg_resp_len),
                "ood_avg_resp_len": float(ood_avg_resp_len),
                "count_id": int(id_gen_stats["count"]),
                "count_ood": int(ood_gen_stats["count"]),
                "count_total": int(total_count),
            }
            if template_metrics_by_split is not None:
                summary_record["template_metrics"] = template_metrics_by_split["total"]
            summary_records.append(summary_record)

        # Synchronize before cache cleanup
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if rank == 0:
            try:
                if ck_cache_root.exists():
                    shutil.rmtree(ck_cache_root)
                    print(f"[cache] Removed cache directory {ck_cache_root}")
            except Exception as cache_rm_exc:
                print(f"[warn] Failed to remove cache directory {ck_cache_root}: {cache_rm_exc}")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # Final summary files
    if rank == 0 and summary_records:
        try:
            summary_json = out_root / f"{args.summary_filename}.json"
            with open(summary_json, "w", encoding="utf-8") as f:
                json.dump(summary_records, f, indent=2, ensure_ascii=False)

            import csv
            summary_csv = out_root / f"{args.summary_filename}.csv"
            fields = [
                "checkpoint",
                "id_acc", "ood_acc", "total_acc",
                "id_avg_loss", "ood_avg_loss", "total_avg_loss",
                "id_avg_resp_len", "ood_avg_resp_len",
                "count_id", "count_ood", "count_total",
            ]
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for r in summary_records:
                    writer.writerow({k: r.get(k, "") for k in fields})
            print(f"[write] summary files -> {summary_json} , {summary_csv}")
        except Exception as e:
            print(f"[warn] Failed to write overall summary: {e}")


if __name__ == "__main__":
    main()
