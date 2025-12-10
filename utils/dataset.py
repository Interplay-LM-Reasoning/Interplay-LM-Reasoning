import json
from pathlib import Path
from typing import List, Dict, Any

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            yield obj

def load_split(split_dir: Path, source_index: Dict = None, repair_mode: str = "symlink") -> List[Dict[str, Any]]:
    """
    Loads all .jsonl files in a split directory and returns a list of dicts (rows).
    Optionally repairs missing files using source_index and repair_mode (not implemented here).
    """
    rows = []
    if not split_dir.exists():
        return rows
    for file in sorted(split_dir.glob("*.jsonl")):
        for obj in iter_jsonl(file):
            rows.append(obj)
    return rows
