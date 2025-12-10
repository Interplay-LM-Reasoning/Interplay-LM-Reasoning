#!/usr/bin/env python
"""
Build an OpenAI/LLaMA-style Byte-level BPE + Regex tokenizer from
zero_context_medium difficulty data.

- Pre-tokenization: Regex pattern (PAT_STR) -> ByteLevel(add_prefix_space=...)
- Normalization: identity (no case-folding / accent stripping)
- Coverage: initial_alphabet = ByteLevel.alphabet() to avoid real OOVs
- Decoder: ByteLevel (reversible)

Collects JSONL under --data-dir, concatenating and wrapping with [BOS] ... [EOS]:
    [BOS]
        <question> (problem + question) </question>
        <solution> solution_body </solution>
        <answer> answer </answer>    # if Answer: found
    [EOS]

Saves a HF-compatible tokenizer directory with:
  tokenizer.json, tokenizer_config.json, special_tokens_map.json,
  vocab.json, merges.txt, SPECIAL_TOKENS.txt, manifest.json

Example:
  python trainer/difficulty/build_tokenizer.py \
      --data-dir data/split/difficulty/zero_context_medium \
      --save-dir trainer/difficulty/zero_context_medium_tokenizer \
      --vocab-size 32000
"""
from __future__ import annotations

import os
import re
import json
import argparse
import shutil
import multiprocessing as mp
from typing import Iterable, List

try:
    from tqdm import tqdm
except ImportError:  # lightweight fallback
    def tqdm(x, **kwargs):  # type: ignore
        return x

from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split, ByteLevel, Sequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast

# --------------------------------------------------------------------
# Special tokens (kept from your original script) + explicit BOS/EOS
# --------------------------------------------------------------------
BOS_EOS = [
    "[BOS]",
    "[EOS]",
]

SPECIAL_TOKENS_USER = [
    "<question>",
    "</question>",
    "<solution>",
    "</solution>",
    "<answer>",
    "</answer>",
]

# Keep UNK/PAD for HF/serving interfaces. We will *not* emit UNK during encoding.
BASE_SPECIAL = ["[UNK]", "[PAD]"]
ALL_SPECIAL = BASE_SPECIAL + BOS_EOS + SPECIAL_TOKENS_USER

# --------------------------------------------------------------------
# Regex pre-tokenizer pattern (OpenAI/LLaMA-style)
# - case-insensitive contractions as atomic pieces
# - numbers split into 1–3 digit groups
# - unicode letters/numbers and "other" symbols
# - explicit whitespace handling
# NOTE: requires the 'regex' engine used by 🤗 tokenizers (supports \p{L}, \p{N}, etc.)
# --------------------------------------------------------------------
PAT_STR = (
    # 1) LaTeX control words & symbols
    r"\\[A-Za-z]+|\\."                         # \frac, \alpha, \left, \_, \%, \{ ...
    r"|\\begin\s*\{[A-Za-z*]+\}|\\end\s*\{[A-Za-z*]+\}"  # \begin{aligned} / \end{aligned}

    # 2) Math operators (unicode & ASCII multi-char)
    r"|<=|>=|!=|:=|->|->>|=>|<=>|::=|==|\\to|\\mapsto|\\implies|\\iff"
    r"|[±≈≅≃≡∼∝∞√∑∏∫∮∇∂∆∈∉⊂⊆⊄⊇⊃∩∪∧∨¬⇒⇔←→↦⟶⟨⟩⋯…]"
    
    # 3) Sub/sup & delimiters as stand-alone
    r"|[_^]"
    r"|[{}()\\[\\]]"                           # braces and brackets

    # 4) Numbers: 1–3 digit groups, decimals, scientific notation (no leading +/− glued)
    r"|(?<![\\p{L}\\p{N}])\\p{N}{1,3}(?:[_,]\\p{N}{1,3})*"  # 1,234 or 1_234 style groups
    r"(?:\\.\\p{N}+)?(?:[eE][+-]?\\p{N}+)?"

    # 5) Keep hyphen/minus separate (don’t merge into identifiers or numbers)
    r"|-" 

    # 6) Words / identifiers (unicode letters), keep optional leading space like GPT/Llama
    r"| ?\\p{L}+"

    # 7) “Other” symbols (punct/emoji/etc.) with optional leading space
    r"| ?[^\\s\\p{L}\\p{N}]+"

    # 8) Whitespace handling (end vs mid)
    r"|\\s+(?!\\S)"
    r"|\\s+"
)

# --------------------------------------------------------------------
# Data plumbing (unchanged in spirit)
# --------------------------------------------------------------------
def extract_answer_and_solution(solution_text: str):
    """Split solution into (solution_body, answer) if an 'Answer:' tag appears."""
    if "Answer:" not in solution_text:
        return solution_text.strip(), None
    parts = solution_text.rsplit("Answer:", 1)
    sol_body = parts[0].rstrip()
    ans_segment = parts[1].strip()
    ans_segment = ans_segment.splitlines()[0].strip()
    ans_segment = re.sub(r"[\s\.]*$", "", ans_segment)
    return sol_body, ans_segment if ans_segment else None

def iter_jsonl_lines(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line == "b":  # skip stray artifacts
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield obj

def _process_jsonl_file(path: str) -> List[str]:
    lines: List[str] = []
    for obj in iter_jsonl_lines(path):
        problem = (obj.get("problem") or "").strip()
        question = (obj.get("question") or "").strip()
        solution = (obj.get("solution") or "").strip()
        pq = (problem + " " + question).strip()
        sol_body, answer = extract_answer_and_solution(solution)
        parts: List[str] = [
            "[BOS]",
            "<question>", pq, "</question>",
            "<solution>", sol_body, "</solution>",
        ]
        if answer:
            parts.extend(["<answer>", answer, "</answer>"])
        parts.append("[EOS]")
        lines.append(" ".join(filter(None, parts)))
    return lines


def _collect_jsonl_files(data_dir: str) -> List[str]:
    files: List[str] = []
    for root, _, filenames in os.walk(data_dir):
        for fn in filenames:
            if fn.endswith(".jsonl"):
                files.append(os.path.join(root, fn))
    files.sort()
    return files


def build_corpus_lines(data_dir: str, workers: int = 1) -> List[str]:
    files = _collect_jsonl_files(data_dir)
    if not files:
        return []

    corpus: List[str] = []
    if workers <= 1:
        for path in tqdm(files, desc="processing files", unit="file"):
            corpus.extend(_process_jsonl_file(path))
        return corpus

    with mp.get_context("spawn").Pool(processes=workers) as pool:
        imap_iter = tqdm(
            pool.imap_unordered(_process_jsonl_file, files),
            total=len(files),
            desc="processing files",
            unit="file",
        )
        for lines in imap_iter:
            corpus.extend(lines)
    return corpus

# --------------------------------------------------------------------
# Training (Byte-level BPE + Regex pre-tokenizer)
# --------------------------------------------------------------------
def train_tokenizer(lines: Iterable[str],
                    vocab_size: int,
                    min_frequency: int,
                    limit_alphabet: int,
                    max_token_length: int,
                    add_prefix_space: bool) -> Tokenizer:
    # No UNK emission at encode-time (we include [UNK] only for API parity)
    tokenizer = Tokenizer(BPE(unk_token=None))

    # Pre-tokenization: Regex split (isolated) -> ByteLevel
    tokenizer.pre_tokenizer = Sequence([
        Split(Regex(PAT_STR), behavior="isolated"),
        ByteLevel(add_prefix_space=add_prefix_space),
    ])

    # ByteLevel decoder ensures reversibility
    tokenizer.decoder = ByteLevelDecoder()

    # Ensure full byte coverage to avoid true OOVs
    initial_alphabet = ByteLevel.alphabet()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=ALL_SPECIAL,
        initial_alphabet=initial_alphabet,
        limit_alphabet=limit_alphabet,
        max_token_length=max_token_length,
    )

    # The library accepts an iterator + length. Ensure we operate on a concrete list once.
    cached = lines if isinstance(lines, list) else list(lines)
    tokenizer.train_from_iterator(cached, trainer=trainer, length=len(cached))
    return tokenizer

def save_transformers_wrapper(tokenizer: Tokenizer, save_dir: str,
                              add_prefix_space: bool):
    """
    Wrap in PreTrainedTokenizerFast and save in HF format.
    This will create:
      - tokenizer.json (copied)
      - tokenizer_config.json
      - special_tokens_map.json
      - vocab.json, merges.txt (for GPT-2 style compatibility)
    """
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        # Expose special tokens to Transformers
        unk_token="[UNK]",              # present for API parity; not emitted by encode
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        additional_special_tokens=SPECIAL_TOKENS_USER,
    )
    # Some consumers care about these flags for byte-level behavior
    fast.init_kwargs["add_prefix_space"] = add_prefix_space
    fast.init_kwargs["model_max_length"] = int(1e9)  # effectively "unset"; set per model later
    fast.save_pretrained(save_dir)

    # also keep a human-friendly list of specials
    with open(os.path.join(save_dir, "SPECIAL_TOKENS.txt"), "w", encoding="utf-8") as f:
        for tok in ALL_SPECIAL:
            f.write(tok + "\n")

def write_manifest(save_dir: str, args, n_lines: int):
    manifest = {
        "type": "byte_level_bpe+regex",
        "pat_str": PAT_STR,
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "limit_alphabet": args.limit_alphabet,
        "max_token_length": args.max_token_length,
        "byte_level_add_prefix_space": args.add_prefix_space,
        "special_tokens": ALL_SPECIAL,
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "corpus_source": args.data_dir,
        "num_training_lines": n_lines,
    }
    with open(os.path.join(save_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/split/difficulty/zero_context_medium",
                    help="Root directory containing JSONL difficulty data.")
    ap.add_argument("--save-dir", default="LLaMA-Factory/configs/difficulty_qwen2",
                    help="Directory to save tokenizer.")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--limit-alphabet", type=int, default=1000,
                    help="Upper bound for initial char alphabet (kept modest; bytes guarantee coverage).")
    ap.add_argument("--max-token-length", type=int, default=100,
                    help="Prevent pathological mega-tokens.")
    ap.add_argument("--add-prefix-space", action="store_true",
                    help="ByteLevel(add_prefix_space=True) like GPT/LLaMA. Default False for continuity.")
    ap.add_argument("--no-overwrite", action="store_true", help="Fail if save directory exists.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2),
                    help="Parallel workers for corpus building (default: half of logical cores).")
    return ap.parse_args()

def main():
    args = parse_args()

    if os.path.exists(args.save_dir):
        if args.no_overwrite:
            raise SystemExit(f"Save directory {args.save_dir} exists. Use a different path or remove it.")
        else:
            print(f"[info] Save directory {args.save_dir} exists. Will overwrite files inside (no removal).")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[info] Collecting corpus from {args.data_dir}")
    corpus_iter = build_corpus_lines(args.data_dir, workers=max(1, args.workers))
    print(f"[info] Collected {len(corpus_iter)} training lines")
    for i, sample in enumerate(corpus_iter[:3]):
        print(f"[sample {i}] {sample[:200]}...")

    print("[info] Training Byte-level BPE + Regex tokenizer...")
    tokenizer = train_tokenizer(
        corpus_iter,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        limit_alphabet=args.limit_alphabet,
        max_token_length=args.max_token_length,
        add_prefix_space=args.add_prefix_space,
    )
    print(f"[info] Training complete")

    print("[info] Saving tokenizer in HF format...")
    save_transformers_wrapper(tokenizer, args.save_dir, add_prefix_space=args.add_prefix_space)

    # record contract for reproducibility
    write_manifest(args.save_dir, args, n_lines=len(corpus_iter))

    print(f"[done] Tokenizer saved to {args.save_dir}")
    print("Files you should see now:")
    print("  - tokenizer.json")
    print("  - tokenizer_config.json")
    print("  - special_tokens_map.json")
    print("  - merges.txt")
    print("  - vocab.json")
    print("  - SPECIAL_TOKENS.txt")
    print("  - manifest.json")

if __name__ == "__main__":
    main()
