#!/usr/bin/env python
"""Generate an exact number of samples per operation for zero_context, medium difficulty.

Guarantees each operation id in the predefined schedule has exactly --samples_per_op
JSONL rows (across both generation modes and all templates combined). It differs from
the original datagenerationworker approach which only approximates counts via TOTAL.

Outputs one jsonl file per op:
  <output_dir>/zero_context/medium/<op>/exact_op<op>_samples<S>.jsonl

Each line is a JSON object: problem, question, solution, op, id, template, mode, length, d.

Example:
  python scripts/difficulty/generate_exact_zero_context_medium.py \
      --samples_per_op 1000 --output_dir data/raw/difficulty/zero_context_medium

You may need required deps:
  pip install transformers termcolor pydot tqdm
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import random
import time
import multiprocessing as mp
from collections import defaultdict

import numpy as np

# Resolve project root (this script lives in scripts/difficulty/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REALISTIC_DIR = os.path.join(PROJECT_ROOT, "gsm_infinite", "gsm-infinite", "data", "realistic")

# Make realistic data dir importable (forward_generator, reverse_generator, etc.)
if REALISTIC_DIR not in sys.path:
    sys.path.insert(0, REALISTIC_DIR)

from forward_generator import drawAll  # type: ignore
from reverse_generator import drawAllEquan  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from termcolor import colored  # type: ignore
try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    tqdm = None
import hashlib
import sqlite3
import sqlite3


GEN_FUNCS = {
    "normalforward": drawAll,
    "forwardreverse": drawAllEquan,
}

TEMPLATES = [
    "crazy_zootopia",
    "teachers_in_school",
    "movie_festival_awards",
]

# Schedule mirrors original shell script tiers: (op_max, [ops])
OP_SCHEDULE = [
    (30, [20, 19]),
    (25, [16, 17, 18]),
    (20, [12, 13, 14, 15]),
    (15, [10, 11]),
    (10, [7, 8, 9]),
    (6, [5, 6]),
    (4, [4]),
    (3, [3]),
    (3, [2]),
]

# Create a mapping from operation to its optimal op_max
OP_TO_OPMAX = {}
for op_max, ops in OP_SCHEDULE:
    for op in ops:
        OP_TO_OPMAX[op] = op_max


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples_per_op", type=int, default=1000000, help="Exact number of samples to produce per operation id")
    p.add_argument("--numprocs", type=int, default=32, help="Number of processes to use for parallel generation")
    p.add_argument("--number_range", type=int, default=5)
    p.add_argument("--mod", type=int, default=-1)
    p.add_argument("--d", type=int, default=2, help="Difficulty (2 = medium)")
    p.add_argument("--target_length", type=str, default="zero_context")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "raw", "difficulty", "zero_context_medium"))
    p.add_argument("--tokenizer_path", type=str, default="/netcache/huggingface/Meta-Llama-3-8B-Instruct")
    p.add_argument("--flush_every", type=int, default=10000, help="Flush buffered samples to disk after this many new items per op")
    p.add_argument("--max_attempts", type=int, default=200_000_000, help="Safety cap on total generation attempts to avoid infinite loops")
    p.add_argument("--resume", action="store_true", help="Resume from existing files, skip completed operations")
    p.add_argument("--force", action="store_true", help="Force regeneration even if files exist (opposite of --resume)")
    p.add_argument("--batch_generate", type=int, default=50, help="Generate samples in batches for efficiency")
    p.add_argument("--aggressive_opmax", action="store_true", help="Use more aggressive opmax optimization for speed")
    p.add_argument("--dedicated_ops", action="store_true", help="Dedicate processes to specific operations for better efficiency")
    return p.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_output_path(base_dir: str, length: str, d: int, op: int, samples_per_op: int, part: int = None, process_id: int = None) -> str:
    directory = os.path.join(base_dir, length, "medium" if d == 2 else "hard", str(op))
    ensure_dir(directory)
    if part is not None:
        if process_id is not None:
            return os.path.join(directory, f"exact_op{op}_samples1000_part{part}_proc{process_id}.jsonl")
        else:
            return os.path.join(directory, f"exact_op{op}_samples1000_part{part}.jsonl")
    else:
        return os.path.join(directory, f"exact_op{op}_samples{samples_per_op}.jsonl")


def count_completed_samples(base_dir: str, length: str, d: int, op: int) -> int:
    """Count how many samples have already been generated for an operation."""
    directory = os.path.join(base_dir, length, "medium" if d == 2 else "hard", str(op))
    if not os.path.exists(directory):
        return 0
    
    total_samples = 0
    # Count samples in all part files for this operation
    for filename in os.listdir(directory):
        if filename.startswith(f"exact_op{op}_") and filename.endswith(".jsonl"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for line in f if line.strip())
                total_samples += line_count
            except Exception as e:
                print(f"[warn] Error reading {filepath}: {e}")
                continue
    
    return total_samples


def filter_incomplete_ops(target_ops: list, samples_per_op: int, output_dir: str, target_length: str, d: int) -> list:
    """Filter out operations that have already been completed."""
    incomplete_ops = []
    
    for op in target_ops:
        completed_samples = count_completed_samples(output_dir, target_length, d, op)
        if completed_samples >= samples_per_op:
            pass  # Skip completed operations
        else:
            remaining = samples_per_op - completed_samples
            incomplete_ops.append(op)
    
    return incomplete_ops


def calculate_offset(num, numprocs):
    """Calculate how many samples each process should generate."""
    return (num + numprocs - 1) // numprocs


def get_optimal_opmax(pending_ops: list) -> int:
    """Get the optimal op_max value for the given pending operations.
    
    Uses the smallest op_max that can generate all pending operations,
    making sampling more efficient by reducing the range of possible operations.
    """
    if not pending_ops:
        return 30  # Default fallback
    
    max_pending_op = max(pending_ops)
    
    # Find the smallest op_max that can handle the largest pending operation
    for op_max, ops in OP_SCHEDULE:
        if max_pending_op in ops or op_max >= max_pending_op:
            return op_max
    
    # Fallback: use the largest op_max if nothing matches
    return max(op_max for op_max, _ in OP_SCHEDULE)


def get_adaptive_opmax(pending_ops: list, successful_samples: int, total_attempts: int, aggressive: bool = False) -> int:
    """Get an adaptive op_max that considers both pending operations and success rate.
    
    If success rate is low, use a more restrictive op_max to improve efficiency.
    """
    if not pending_ops:
        return 30
        
    base_opmax = get_optimal_opmax(pending_ops)
    
    # Aggressive mode: use exact op_max for single operations
    if aggressive and len(pending_ops) == 1:
        op = pending_ops[0]
        if op in OP_TO_OPMAX:
            return OP_TO_OPMAX[op]
    
    # If we have enough attempts to calculate success rate
    if total_attempts > 50:  # Reduced threshold for faster adaptation
        success_rate = successful_samples / total_attempts
        
        # More aggressive success rate thresholds
        if success_rate < 0.05:  # Very low success rate
            # Use the most restrictive possible op_max
            if len(pending_ops) == 1 and pending_ops[0] in OP_TO_OPMAX:
                return OP_TO_OPMAX[pending_ops[0]]
            return min(op_max for op_max, ops in OP_SCHEDULE 
                      if any(op in ops for op in pending_ops))
        elif success_rate < 0.15:  # Low success rate
            # Use a slightly more restrictive op_max
            for op_max, ops in OP_SCHEDULE:
                if all(op in ops or op_max >= op for op in pending_ops):
                    return op_max
    
    return base_opmax


def get_specialized_opmax_per_op(pending_ops: list) -> dict:
    """Get specialized op_max for each operation for maximum efficiency."""
    op_to_opmax = {}
    for op in pending_ops:
        if op in OP_TO_OPMAX:
            op_to_opmax[op] = OP_TO_OPMAX[op]
        else:
            # Find the smallest op_max that can generate this operation
            for op_max, ops in OP_SCHEDULE:
                if op in ops or op_max >= op:
                    op_to_opmax[op] = op_max
                    break
            else:
                op_to_opmax[op] = 30  # Fallback
    return op_to_opmax


def work_function(
    target_ops,
    samples_per_op,
    numprocs,
    process_id,
    number_range,
    target_length,
    d,
    tokenizer_path,
    output_dir,
    seed,
    max_attempts_per_process,
    batch_generate=50,
    aggressive_opmax=False,
    dedicated_ops=False
):
    """Worker function that runs in each process to generate samples."""
    # Import here to avoid issues with multiprocessing
    from forward_generator import drawAll
    from reverse_generator import drawAllEquan
    from transformers import AutoTokenizer
    from termcolor import colored
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    
    # Initialize random seeds uniquely for each process
    process_seed = seed + process_id * 1000
    random.seed(process_seed)
    np.random.seed(process_seed)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    gen_funcs = {
        "normalforward": drawAll,
        "forwardreverse": drawAllEquan,
    }
    
    templates = [
        "crazy_zootopia",
        "teachers_in_school", 
        "movie_festival_awards",
    ]

    # Deduplication is done per-op. For each op we will look for an index file:
    #   <output_dir>/hash_index/op<op>.txt
    # and use a reservation file per op:
    #   <output_dir>/hash_reserved_op<op>.txt
    hash_index_dir = os.path.join(output_dir, "hash_index")

    def compute_sample_hash(problem: str, question: str, solution: str) -> str:
        m = hashlib.sha256()
        m.update(problem.encode("utf-8"))
        m.update(b"\n--FIELD--\n")
        m.update(question.encode("utf-8"))
        m.update(b"\n--FIELD--\n")
        m.update(solution.encode("utf-8"))
        return m.hexdigest()

    # Use a per-op SQLite DB to avoid loading large per-op indexes into memory.
    # For each op we lazily create/open a DB at <hash_index_dir>/op<op>.db,
    # set PRAGMA to WAL and busy timeout, and import lines from op<op>.txt
    # if present. Deduplication is done by attempting to INSERT and catching
    # sqlite3.IntegrityError on primary-key violation.

    db_conn_cache: dict[str, sqlite3.Connection] = {}
    db_imported: set[str] = set()

    def get_db_for_op(op: int) -> sqlite3.Connection:
        key = str(op)
        if key in db_conn_cache:
            return db_conn_cache[key]

        os.makedirs(hash_index_dir, exist_ok=True)
        db_path = os.path.join(hash_index_dir, f"op{op}.db")
        conn = sqlite3.connect(db_path, timeout=30, isolation_level=None, check_same_thread=False)
        # Configure for concurrent access
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            pass

        # Ensure table exists
        conn.execute("CREATE TABLE IF NOT EXISTS hashes(hash TEXT PRIMARY KEY)")

        # Lazy import from op<op>.txt if present and not yet imported in this process
        idx_txt = os.path.join(hash_index_dir, f"op{op}.txt")
        if os.path.exists(idx_txt) and key not in db_imported:
            try:
                with open(idx_txt, "r", encoding="utf-8") as rf:
                    # Use executemany in small batches
                    batch = []
                    for line in rf:
                        h = line.strip()
                        if not h:
                            continue
                        batch.append((h,))
                        if len(batch) >= 1000:
                            conn.executemany("INSERT OR IGNORE INTO hashes(hash) VALUES(?)", batch)
                            batch = []
                    if batch:
                        conn.executemany("INSERT OR IGNORE INTO hashes(hash) VALUES(?)", batch)
                db_imported.add(key)
            except Exception:
                # ignore import errors; DB may still be usable
                pass

        db_conn_cache[key] = conn
        return conn

    def reserve_hash_for_op(h: str, op: int) -> bool:
        """Reserve hash h for op using per-op sqlite DB.

        Returns True if the hash was newly inserted (unique), False if it already existed.
        """
        conn = get_db_for_op(op)
        try:
            # Use a transaction to ensure atomicity; IntegrityError indicates duplicate
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("INSERT INTO hashes(hash) VALUES(?)", (h,))
                conn.execute("COMMIT")
                return True
            except sqlite3.IntegrityError:
                conn.execute("ROLLBACK")
                return False
            except Exception:
                conn.execute("ROLLBACK")
                return False
        except sqlite3.OperationalError:
            # busy or other operational error; conservatively treat as duplicate
            return False
    
    # Each process generates the FULL samples_per_op for its assigned operations
    # We distribute operations among processes, not sample counts
    samples_per_process = samples_per_op
    
    # Tracking for this process - initialize with already completed samples
    counts = defaultdict(int)
    buffers = defaultdict(list)
    
    # Initialize counts with already completed samples
    for op in target_ops:
        completed = count_completed_samples(output_dir, target_length, d, op)
        counts[op] = completed
    
    # Calculate total remaining work for progress bar
    total_remaining = sum(max(0, samples_per_process - counts[op]) for op in target_ops)
    
    if total_remaining == 0:
        return
    
    # Initialize progress bar for this process
    if tqdm:
        pbar = tqdm(
            total=total_remaining,
            desc=f"Process {process_id} (ops: {target_ops})",
            position=process_id,
            leave=True,
            unit="samples",
            smoothing=0.1  # Smoother progress updates
        )
    else:
        pbar = None
    
    attempts = 0
    successful_samples = 0
    last_success_rate = 0
    adaptation_counter = 0
    mode_cycle = list(gen_funcs.keys())
    template_cycle = list(templates)
    
    # Pre-calculate specialized op_max for each operation if using dedicated mode
    if dedicated_ops:
        specialized_opmax = get_specialized_opmax_per_op(target_ops)
    
    remaining_work = {op: samples_per_process - counts[op] for op in target_ops if counts[op] < samples_per_process}
    if not remaining_work:
        if pbar:
            pbar.close()
        return
    
    # Batch generation optimization
    batch_attempts = 0
    last_flush_time = time.time()
    
    while True:
        # Exit condition for this process
        if all(counts[op] >= samples_per_process for op in target_ops):
            break
            
        # Find operations that still need samples
        pending_ops = [op for op in target_ops if counts[op] < samples_per_process]
        if not pending_ops:
            break
            
        # Adaptive generation strategy based on remaining work
        if dedicated_ops and len(pending_ops) == 1:
            # Focus entirely on the single remaining operation
            target_op = pending_ops[0]
            op_max = specialized_opmax.get(target_op, OP_TO_OPMAX.get(target_op, 30))
        else:
            # Use adaptive op_max
            op_max = get_adaptive_opmax(pending_ops, successful_samples, attempts, aggressive_opmax)
        
        # Choose mode & template in round-robin fashion
        mode = mode_cycle[attempts % len(mode_cycle)]
        template = template_cycle[(attempts // len(mode_cycle)) % len(template_cycle)]
        
        # Batch generation for efficiency
        batch_success = False
        for _ in range(min(batch_generate, max_attempts_per_process - attempts)):
            try:
                problem_text, question_text, solution_text, op, oid = gen_funcs[mode](
                    op_max=op_max,
                    ip_max=20,
                    force=True,
                    number_range=number_range,
                    strictline=op_max,
                    mod=-1,
                    target_length=target_length,
                    template=template,
                    d=d,
                    tokenizer=tokenizer,
                    oplist=pending_ops,
                )
                
                attempts += 1
                
                if op in pending_ops:
                    # Deduplication: compute hash and attempt to reserve (per-op)
                    sample_hash = compute_sample_hash(problem_text, question_text, solution_text)
                    # Reserve using sqlite-backed per-op DB (memory-efficient)
                    reserved = reserve_hash_for_op(sample_hash, op)
                    if not reserved:
                        # Duplicate for this op; skip and continue generating
                        attempts += 1
                        continue

                    # Record sample (unique)
                    sample = {
                        "problem": problem_text,
                        "question": question_text,
                        "solution": solution_text,
                        "op": op,
                        "id": f"{oid}_p{process_id}",
                        "template": template,
                        "mode": mode,
                        "length": target_length,
                        "d": d,
                    }
                    counts[op] += 1
                    buffers[op].append(json.dumps(sample))
                    successful_samples += 1
                    batch_success = True
                    
                    # Update progress bar
                    if pbar:
                        pbar.update(1)
                        # Update description with current operation, success rate, and op_max
                        success_rate = (successful_samples / attempts * 100) if attempts > 0 else 0
                        pbar.set_description(f"Process {process_id} (op:{op} max:{op_max} success:{success_rate:.1f}%)")
                    
                    # Check if this operation is now complete
                    if counts[op] >= samples_per_process:
                        # Force flush for completed operation
                        if buffers[op]:
                            current_part = ((counts[op] - 1) // 1000) + 1
                            file_path = build_output_path(output_dir, target_length, d, op, samples_per_op, current_part, process_id)
                            with open(file_path, "a", encoding="utf-8") as f:
                                f.write("\n".join(buffers[op]) + "\n")
                            buffers[op].clear()
                        break  # Exit batch loop for this operation
                
            except Exception:
                attempts += 1
                continue
        
        # Early exit if max attempts reached
        if attempts >= max_attempts_per_process:
            if pbar:
                pbar.set_description(f"Process {process_id} - Max attempts reached")
                pbar.close()
            break
        
        # Periodic flushing for efficiency
        current_time = time.time()
        if current_time - last_flush_time > 30:  # Flush every 30 seconds
            for op, buf in buffers.items():
                if len(buf) >= 1000:  # Only flush if buffer is substantial
                    current_part = ((counts[op] - 1) // 1000) + 1
                    file_path = build_output_path(output_dir, target_length, d, op, samples_per_op, current_part, process_id)
                    with open(file_path, "a", encoding="utf-8") as f:
                        f.write("\n".join(buf) + "\n")
                    buf.clear()
            last_flush_time = current_time
            
    # Final flush for any remaining buffers
    for op, buf in buffers.items():
        if buf:
            current_part = ((counts[op] - 1) // 1000) + 1
            file_path = build_output_path(output_dir, target_length, d, op, samples_per_op, current_part, process_id)
            
            with open(file_path, "a", encoding="utf-8") as f:
                f.write("\n".join(buf) + "\n")
    
    # Close progress bar
    if pbar:
        pbar.set_description(f"Process {process_id} - Completed!")
        pbar.close()


def main():
    args = parse_args()
    
    print(colored(f"[config] samples_per_op={args.samples_per_op} d={args.d} length={args.target_length}", "cyan"))
    print(colored(f"[config] numprocs={args.numprocs} output_dir={args.output_dir}", "cyan"))
    
    # Validate conflicting arguments
    if args.resume and args.force:
        print(colored("[error] Cannot use both --resume and --force flags together", "red"))
        return
    
    # Get all operations we need to generate
    total_ops = {op for _, ops in OP_SCHEDULE for op in ops}
    
    if args.force:
        print(colored("[force] Forcing regeneration of all operations (ignoring existing files)", "yellow"))
        incomplete_ops = list(total_ops)
    elif args.resume:
        # Check for already completed operations and filter them out
        print(colored("[resume] Checking for already completed operations...", "yellow"))
        incomplete_ops = filter_incomplete_ops(
            list(total_ops), 
            args.samples_per_op, 
            args.output_dir, 
            args.target_length, 
            args.d
        )
    else:
        # Default behavior: warn about existing files but continue
        print(colored("[check] Checking for existing files (use --resume to skip completed, --force to overwrite)...", "yellow"))
        incomplete_ops = filter_incomplete_ops(
            list(total_ops), 
            args.samples_per_op, 
            args.output_dir, 
            args.target_length, 
            args.d
        )
        if len(incomplete_ops) < len(total_ops):
            print(colored("[info] Some operations already have files. Use --resume to skip completed operations or --force to regenerate all.", "blue"))
    
    if not incomplete_ops:
        print(colored("[done] All operations already completed! Nothing to generate.", "green"))
        return
    
    print(colored(f"[continue] Processing {len(incomplete_ops)} operations: {sorted(incomplete_ops)}", "cyan"))
    
    # Divide incomplete operations among processes
    ops_list = sorted(incomplete_ops)
    if len(ops_list) < args.numprocs:
        print(colored(f"[adjust] Reducing numprocs from {args.numprocs} to {len(ops_list)} (one process per remaining operation)", "yellow"))
        args.numprocs = len(ops_list)
    
    ops_per_process = len(ops_list) // args.numprocs
    remainder = len(ops_list) % args.numprocs
    
    processes = []
    start_idx = 0
    
    # Calculate max attempts per process
    max_attempts_per_process = args.max_attempts // args.numprocs
    
    for i in range(args.numprocs):
        # Distribute operations among processes
        end_idx = start_idx + ops_per_process
        if i < remainder:
            end_idx += 1
        
        process_ops = ops_list[start_idx:end_idx]
        
        if process_ops:  # Only start process if it has operations to work on
            p = mp.Process(
                target=work_function,
                args=(
                    process_ops,
                    args.samples_per_op,
                    args.numprocs,
                    i,
                    args.number_range,
                    args.target_length,
                    args.d,
                    args.tokenizer_path,
                    args.output_dir,
                    args.seed,
                    max_attempts_per_process,
                    args.batch_generate,
                    args.aggressive_opmax,
                    args.dedicated_ops
                )
            )
            processes.append(p)
            p.start()
            print(colored(f"[main] Started process {i} for operations: {process_ops}", "blue"))
        
        start_idx = end_idx
    
    # Wait for all processes to complete
    for i, p in enumerate(processes):
        p.join()
        print(colored(f"[main] Process {i} completed", "blue"))
    
    print(colored("[done] All processes completed. Check output files for results.", "cyan"))
    print(colored(f"Output directory: {args.output_dir}", "cyan"))


if __name__ == "__main__":
    # Necessary for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)
    main()
