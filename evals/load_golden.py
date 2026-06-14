"""Loader for the golden eval set (evals/golden/golden_medical.jsonl).

Lines starting with // are comments. Each remaining line is one JSON object:
{id, question, ground_truth, type: factual|synthesis|unanswerable, smoke: bool}
"""

import json
import os
from pathlib import Path
from typing import List, Optional

# default medical set; override with GOLDEN_PATH env
GOLDEN_PATH = Path(
    os.getenv("GOLDEN_PATH", Path(__file__).resolve().parent / "golden" / "golden_medical.jsonl")
)

# `company` is optional and unused by the medical set (rows omit it)
REQUIRED_KEYS = {"id", "question", "ground_truth", "type", "smoke"}
VALID_TYPES = {"factual", "synthesis", "unanswerable"}


def load_golden(path: Optional[Path] = None, smoke_only: bool = False) -> List[dict]:
    path = path or GOLDEN_PATH
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        row = json.loads(line)
        missing = REQUIRED_KEYS - row.keys()
        if missing:
            raise ValueError(f"{path.name}:{n} missing keys {sorted(missing)}")
        if row["type"] not in VALID_TYPES:
            raise ValueError(f"{path.name}:{n} bad type {row['type']!r}")
        rows.append(row)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in golden set")
    if smoke_only:
        rows = [r for r in rows if r["smoke"]]
    return rows


if __name__ == "__main__":
    rows = load_golden()
    print(f"{len(rows)} golden rows ({sum(r['smoke'] for r in rows)} smoke)")
    for r in rows:
        print(f"  {r['id']} [{r['type']:>12}] smoke={r['smoke']} {r['question'][:60]}")
