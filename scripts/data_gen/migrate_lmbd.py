#!/usr/bin/env python3
"""
Migrate old LMDBs (missing db_digests) by recomputing digests
from the original JSONL files under zero_context/medium/{op}.
"""

import os, lmdb, json, xxhash
from tqdm import tqdm

FIELD_SEP = "\n--FIELD--\n"

def compute_exact_digest(text: str) -> bytes:
    return xxhash.xxh3_64_digest(text.encode("utf-8"))

def migrate_op(op: str, lmdb_dir: str, jsonl_root: str):
    op_dir = os.path.join(lmdb_dir, f"op{op}.lmdb")
    if not os.path.isdir(op_dir):
        print(f"[skip] {op_dir} not found")
        return 0

    env = lmdb.open(op_dir, map_size=128<<30, max_dbs=4, subdir=True)
    db_digests = env.open_db(b"digests")

    op_path = os.path.join(jsonl_root, "zero_context", "medium", op)
    if not os.path.isdir(op_path):
        print(f"[warn] no directory {op_path}")
        env.close()
        return 0

    json_files = [
        os.path.join(op_path, fn)
        for fn in os.listdir(op_path) if fn.endswith(".jsonl")
    ]
    if not json_files:
        print(f"[warn] no jsonl files under {op_path}")
        env.close()
        return 0

    total = 0
    with env.begin(write=True, db=db_digests) as txn:
        for jf in tqdm(json_files, desc=f"op{op}"):
            fid = xxhash.xxh3_64_hexdigest(jf.encode("utf-8"))
            with open(jf, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if str(obj.get("op")) != str(op):
                        continue
                    text = f"{obj.get('problem','')}{FIELD_SEP}{obj.get('question','')}{FIELD_SEP}{obj.get('solution','')}"
                    digest = compute_exact_digest(text)
                    sid = f"{op}|{fid}|{idx}"
                    txn.put(digest, sid.encode("utf-8"), overwrite=False)
                    total += 1

    env.close()
    print(f"[done] op{op}: inserted {total} digests")
    return total

if __name__ == "__main__":
    import argparse, csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb_dir", default="data/raw/difficulty/zero_context_medium/hash_index_lsh",
                   help="path to hash_index_lsh containing op*.lmdb")
    ap.add_argument("--jsonl_root", default="data/raw/difficulty/zero_context_medium/",
                   help="root containing zero_context/medium/{op}/*.jsonl")
    ap.add_argument("--ops", type=str, default=None,
                   help="comma-separated op ids (default: auto detect)")
    args = ap.parse_args()

    if args.ops:
        ops = args.ops.split(",")
    else:
        # auto-detect ops from lmdb_dir
        ops = [
            fn.split("op")[-1].split(".lmdb")[0]
            for fn in os.listdir(args.lmdb_dir)
            if fn.startswith("op") and fn.endswith(".lmdb")
        ]
        ops.sort(key=lambda x: int(x) if x.isdigit() else x)

    summary = []
    for op in ops:
        count = migrate_op(op, args.lmdb_dir, args.jsonl_root)
        summary.append((op, count))

    # print global summary as CSV
    print("\n=== Migration Summary ===")
    print("op,total_digests")
    for op, cnt in summary:
        print(f"{op},{cnt}")
    total_all = sum(cnt for _, cnt in summary)
    print(f"ALL,{total_all}")
