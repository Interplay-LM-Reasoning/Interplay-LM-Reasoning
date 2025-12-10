#!/usr/bin/env python3
"""
Distributed exact dedup (with LSH bins for sharding) for zero_context/medium.
Resume is fast via:
  1) Digest-first checking (xxhash) — no MinHash if already seen.
  2) Per-file DONE marker — skip whole files already processed.

Features:
- Resume support: skips existing samples/files
- Exact dedup inside bins (xxhash digest equality)
- Batched Ray RPCs and LMDB writes
- File-batch and sample progress bars
"""

from __future__ import annotations
import os, sys, argparse, csv
from typing import Iterable, List, Dict, Tuple

import ujson as json
from tqdm import tqdm

import ray, lmdb, xxhash
from datasketch import MinHash

FIELD_SEP = "\n--FIELD--\n"

# -------------------------------
# Tokenization / text utilities
# -------------------------------

def tokenize_words(text: str) -> Iterable[bytes]:
    for tok in text.lower().split():
        if tok:
            yield tok.encode("utf-8")

def tokenize_char_ngrams(text: str, n: int = 5) -> Iterable[bytes]:
    t = text.lower()
    L = len(t)
    if L < n:
        if L > 0:
            yield t.encode("utf-8")
        return
    for i in range(L - n + 1):
        yield t[i:i+n].encode("utf-8")

def compute_minhash(text: str, num_perm: int, tokenizer: str, ngram: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    if tokenizer == "word":
        toks = [tok.encode("utf-8") for tok in text.lower().split() if tok]
    else:
        t = text.lower()
        if len(t) < ngram:
            toks = [t.encode("utf-8")] if t else []
        else:
            toks = [t[i:i+ngram].encode("utf-8") for i in range(len(t) - ngram + 1)]
    if hasattr(mh, "update_batch"):
        mh.update_batch(toks)
    else:
        for tok in toks:
            mh.update(tok)
    return mh

def compute_exact_digest(text: str) -> bytes:
    return xxhash.xxh3_64_digest(text.encode("utf-8"))

# -------------------------------
# LSH banding
# -------------------------------

def compute_band_keys(sig_buf: bytes, bands: int, rows: int) -> List[bytes]:
    import numpy as np
    hv = np.frombuffer(sig_buf, dtype=np.uint64)
    keys = []
    assert len(hv) == bands * rows
    for b in range(bands):
        start = b * rows
        chunk = hv[start:start+rows]
        h = xxhash.xxh3_64_digest(chunk.tobytes())
        keys.append(b.to_bytes(2, "big") + h)
    return keys

# -------------------------------
# Storage (LMDB)
# -------------------------------

class LMDBIndex:
    """
    DBs:
      - db_buckets: band_key -> newline-delimited sample_ids
      - db_sigs:    sample_id -> digest (bytes)
      - db_digests: digest -> sample_id (any one)
      - db_files:   fid -> marker (whole file done)
    """
    def __init__(self, op_dir: str, map_size: int = 128 << 30, readonly: bool = False):
        os.makedirs(op_dir, exist_ok=True)
        self.env = lmdb.open(
            op_dir,
            map_size=map_size,
            max_dbs=4,
            subdir=True,
            readonly=readonly,
            lock=not readonly,
            readahead=False,
            writemap=False,
            map_async=True,
            max_readers=2048,
            sync=False,
            metasync=False,
        )
        self.db_buckets = self.env.open_db(b"buckets")
        self.db_sigs = self.env.open_db(b"sigs")
        self.db_digests = self.env.open_db(b"digests")
        self.db_files = self.env.open_db(b"files")

    def close(self): self.env.close()

    def iter_candidates_for_bandkeys(self, band_keys: List[bytes]) -> Iterable[str]:
        with self.env.begin(db=self.db_buckets, buffers=True) as txn:
            for bk in band_keys:
                v = txn.get(bk)
                if v:
                    for line in bytes(v).split(b"\n"):
                        if line:
                            yield line.decode("utf-8")

    def get_signatures_batch(self, sample_ids: List[str]) -> Dict[str, bytes]:
        out: Dict[str, bytes] = {}
        with self.env.begin(db=self.db_sigs, buffers=True) as txn:
            for sid in sample_ids:
                v = txn.get(sid.encode("utf-8"))
                if v:
                    out[sid] = bytes(v)
        return out

    def has_digests_batch(self, digests: List[bytes]) -> List[bytes]:
        found = []
        with self.env.begin(db=self.db_digests, buffers=True) as txn:
            for d in digests:
                if txn.get(d):
                    found.append(d)
        return found

    def put_signatures_batch(self, sigs: List[tuple[str, bytes]]):
        with self.env.begin(write=True, db=self.db_sigs) as txn:
            for sid, digest in sigs:
                txn.put(sid.encode("utf-8"), digest, overwrite=False)

    def put_digests_batch(self, pairs: List[tuple[bytes, str]]):
        with self.env.begin(write=True, db=self.db_digests) as txn:
            for d, sid in pairs:
                txn.put(d, sid.encode("utf-8"), overwrite=False)

    def add_to_buckets_batch(self, bucket_map: Dict[bytes, List[str]]):
        with self.env.begin(write=True, db=self.db_buckets) as txn:
            for bk, ids in list(bucket_map.items()):
                v = txn.get(bk)
                lines = [] if v is None else bytes(v).split(b"\n")
                if ids:
                    lines.extend([sid.encode("utf-8") for sid in ids])
                if lines:
                    txn.put(bk, b"\n".join(lines), overwrite=True)

    def file_done(self, fid: str) -> bool:
        with self.env.begin(db=self.db_files, buffers=True) as txn:
            return txn.get(fid.encode("utf-8")) is not None

    def mark_file_done(self, fid: str):
        with self.env.begin(write=True, db=self.db_files) as txn:
            txn.put(fid.encode("utf-8"), b"1", overwrite=True)

    def count_sigs(self) -> int:
        with self.env.begin(db=self.db_sigs) as txn:
            return txn.stat()["entries"]

# -------------------------------
# Ray Actor: per-op index shard
# -------------------------------

@ray.remote(max_concurrency=32)
class OpIndexActor:
    def __init__(self, op_key: str, base_dir: str, num_perm: int, bands: int, rows: int,
                 map_size_gb: int = 128):
        self.op_key, self.num_perm, self.bands, self.rows = op_key, num_perm, bands, rows
        op_dir = os.path.join(base_dir, f"op{op_key}.lmdb")
        self.idx = LMDBIndex(op_dir, map_size=map_size_gb << 30)

        self.seen = self.accepted = self.duplicates = 0
        self.sig_buffer: List[tuple[str, bytes]] = []
        self.bucket_buffer: Dict[bytes, List[str]] = {}
        self.digest_buffer: List[tuple[bytes, str]] = []

    def _buffer_insert(self, sample_id: str, digest: bytes, band_keys: List[bytes]):
        self.sig_buffer.append((sample_id, digest))
        self.digest_buffer.append((digest, sample_id))
        for bk in band_keys:
            self.bucket_buffer.setdefault(bk, []).append(sample_id)
        self.accepted += 1

    def flush(self):
        if self.sig_buffer:
            self.idx.put_signatures_batch(self.sig_buffer)
            self.sig_buffer.clear()
        if self.digest_buffer:
            self.idx.put_digests_batch(self.digest_buffer)
            self.digest_buffer.clear()
        if self.bucket_buffer:
            self.idx.add_to_buckets_batch(self.bucket_buffer)
            self.bucket_buffer.clear()
    def digest_exists(self, digest: bytes) -> bool:
        """Check if a digest already exists in LMDB."""
        return bool(self.idx.has_digests_batch([digest]))

    def upsert_batch(self, items: List[Tuple[str, bytes, bytes, str]]) -> tuple[int, int]:
        acc = dup = 0
        for sid, sig_buf, digest, fid in items:
            self.seen += 1
            # check digest existence first
            if self.idx.has_digests_batch([digest]):
                self.duplicates += 1
                dup += 1
                continue
            band_keys = compute_band_keys(sig_buf, self.bands, self.rows)
            self._buffer_insert(sid, digest, band_keys)
            acc += 1
            if len(self.sig_buffer) >= 8192:
                self.flush()
        return acc, dup

    def file_done(self, fid: str) -> bool:
        return self.idx.file_done(fid)

    def mark_file_done(self, fid: str):
        self.idx.mark_file_done(fid)

    def stats(self) -> dict:
        self.flush()
        return {"op": self.op_key, "seen": self.seen, "accepted": self.accepted,
                "duplicates": self.duplicates, "db_signatures": self.idx.count_sigs()}

    def close(self):
        self.flush()
        self.idx.close()
        return True

# -------------------------------
# File scanning
# -------------------------------

def find_jsonl_files(root: str):
    search_root = os.path.join(root, "zero_context", "medium")
    for dirpath, _, filenames in os.walk(search_root):
        for fn in filenames:
            if fn.endswith(".jsonl"):
                yield os.path.join(dirpath, fn)

# -------------------------------
# Driver
# -------------------------------

def process_files_distributed(args):
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    files = list(find_jsonl_files(args.data_dir))
    if not files:
        print(f"[error] no jsonl files found under {args.data_dir}/zero_context/medium")
        return 1

    pbar_samples = tqdm(desc="samples", unit="samp")
    chunk_size = 200
    file_chunks = [files[i:i+chunk_size] for i in range(0, len(files), chunk_size)]

    actors: Dict[str, ray.actor.ActorHandle] = {}
    def get_actor_for_op(op: str):
        if op not in actors:
            actors[op] = OpIndexActor.remote(
                op, os.path.join(args.out_dir, "hash_index_lsh"),
                args.num_perm, args.bands, args.rows,
                map_size_gb=args.map_size_gb
            )
        return actors[op]

    @ray.remote
    def parse_file_batch(file_list: List[str], num_perm: int, tokenizer: str, ngram: int,
                         sep: str, bands: int, rows: int):
        accepted_local = seen_local = duplicates_local = 0
        local_actors = {}

        def get_local_actor(op: str):
            if op not in local_actors:
                local_actors[op] = get_actor_for_op(op)
            return local_actors[op]

        per_op_batch: Dict[str, List[Tuple[str, bytes, bytes, str]]] = {}
        BATCH_RPC = 16384

        for filepath in file_list:
            fid = xxhash.xxh3_64_hexdigest(filepath.encode("utf-8"))
            # pick first op actor to check file_done
            actor0 = get_local_actor("0")
            if ray.get(actor0.file_done.remote(fid)):
                print(f"[resume] skipped whole file {filepath}")
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    op = obj.get("op")
                    if op is None:
                        continue
                    sid = f"{op}|{fid}|{line_idx}"
                    text = f"{obj.get('problem','')}{sep}{obj.get('question','')}{sep}{obj.get('solution','')}"
                    digest = compute_exact_digest(text)
                    actor = get_local_actor(str(op))
                    # fast digest check
                    if ray.get(actor.digest_exists.remote(digest)):
                        duplicates_local += 1
                        continue
                    # compute MinHash only if new
                    mh = compute_minhash(text, num_perm=num_perm, tokenizer=tokenizer, ngram=ngram)
                    sig = mh.hashvalues.tobytes()
                    per_op_batch.setdefault(str(op), []).append((sid, sig, digest, fid))
                    seen_local += 1
                    if len(per_op_batch[str(op)]) >= BATCH_RPC:
                        acc, dup = ray.get(actor.upsert_batch.remote(per_op_batch[str(op)]))
                        accepted_local += acc; duplicates_local += dup
                        per_op_batch[str(op)].clear()

            # flush per-op
            for opk, items in list(per_op_batch.items()):
                if items:
                    actor = get_local_actor(opk)
                    acc, dup = ray.get(actor.upsert_batch.remote(items))
                    accepted_local += acc; duplicates_local += dup
                    per_op_batch[opk].clear()

            # mark file done
            actor0.mark_file_done.remote(fid)

        return {"files": len(file_list), "seen": seen_local,
                "accepted": accepted_local, "duplicates": duplicates_local}

    # driver
    pending, file_results = [], []
    pbar_batches = tqdm(total=len(file_chunks), desc="file-batches", unit="batch")

    for chunk in file_chunks:
        pending.append(parse_file_batch.remote(
            chunk, args.num_perm, args.tokenizer, args.ngram,
            FIELD_SEP, args.bands, args.rows
        ))
        if len(pending) >= args.num_workers:
            ready, pending = ray.wait(pending, num_returns=1)
            res = ray.get(ready)[0]
            file_results.append(res)
            pbar_batches.update(1)
            pbar_samples.update(res["seen"])
            print(f"[done] batch({res['files']} files) seen={res['seen']} "
                  f"accepted={res['accepted']} dup={res['duplicates']}")

    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        res = ray.get(ready)[0]
        file_results.append(res)
        pbar_batches.update(1)
        pbar_samples.update(res["seen"])
        print(f"[done] batch({res['files']} files) seen={res['seen']} "
              f"accepted={res['accepted']} dup={res['duplicates']}")

    pbar_batches.close()
    pbar_samples.close()

    per_op_rows = [ray.get(act.stats.remote()) for act in actors.values()]

    base = os.path.join(args.out_dir, "hash_index_lsh"); os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "summary.csv"), "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["op", "seen", "accepted", "duplicates", "db_signatures"])
        for r in per_op_rows:
            w.writerow([r["op"], r["seen"], r["accepted"], r["duplicates"], r["db_signatures"]])

    _ = ray.get([act.close.remote() for act in actors.values()])
    ray.shutdown()
    print(f"Wrote per-op indices and summaries under {base}")
    return 0

# -------------------------------
# Args
# -------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/raw/difficulty/zero_context_medium")
    p.add_argument("--out_dir", type=str, default="data/raw/difficulty/zero_context_medium")
    p.add_argument("--num_perm", type=int, default=64)
    p.add_argument("--bands", type=int, default=8)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--tokenizer", type=str, choices=["word", "char"], default="char")
    p.add_argument("--ngram", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=32)
    p.add_argument("--map_size_gb", type=int, default=128)
    args = p.parse_args()
    if args.num_perm != args.bands * args.rows:
        print("[error] num_perm must equal bands * rows", file=sys.stderr); sys.exit(2)
    return args

def main():
    args = parse_args()
    rc = process_files_distributed(args)
    sys.exit(rc)

if __name__ == "__main__":
    main()
