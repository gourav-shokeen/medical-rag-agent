"""MIRAGE benchmark loader.

MIRAGE (Medical Information Retrieval-Augmented Generation Evaluation,
Teddy-XiongGZ/MIRAGE) ships a single benchmark.json mapping task -> qid -> record
{question, options:{A..}, answer (letter)}. The five tasks:
  MMLU-Med, MedQA-US, MedMCQA  -> exam MCQs (covered by textbooks+StatPearls)
  PubMedQA*, BioASQ-Y/N        -> research/literature QA (need a PubMed corpus we
                                   do NOT have -> retrieval is weak for these)

    python evals/mirage/load_mirage.py            # download (if needed) + counts

Downloaded to evals/mirage/benchmark.json (cached).
"""

import json
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BENCH_URL = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"
BENCH_PATH = Path(__file__).resolve().parent / "benchmark.json"

# tasks this corpus (medical textbooks + StatPearls, no PubMed) can fairly answer
EXAM_TASKS = ("mmlu", "medqa", "medmcqa")
# canonical task-name aliases seen across MIRAGE versions
TASK_ALIASES = {
    "mmlu": ("mmlu", "mmlu-med", "mmlu_med"),
    "medqa": ("medqa", "medqa-us", "medqa_us"),
    "medmcqa": ("medmcqa",),
    "pubmedqa": ("pubmedqa", "pubmedqa*"),
    "bioasq": ("bioasq", "bioasq-y/n", "bioasq_yn"),
}


def ensure_benchmark():
    if not BENCH_PATH.exists():
        print(f"Downloading MIRAGE benchmark -> {BENCH_PATH}")
        urllib.request.urlretrieve(BENCH_URL, BENCH_PATH)
    return BENCH_PATH


def _norm_task(name: str) -> str:
    n = name.lower()
    for canon, aliases in TASK_ALIASES.items():
        if n in aliases or any(n.startswith(a) for a in aliases):
            return canon
    return n


def load_mirage(tasks=None):
    """Return {canonical_task: [ {id, task, question, options, answer_letter} ]}."""
    data = json.loads(ensure_benchmark().read_text(encoding="utf-8"))
    want = set(tasks) if tasks else None
    out = {}
    for raw_task, records in data.items():
        task = _norm_task(raw_task)
        if want and task not in want:
            continue
        items = []
        for qid, rec in records.items():
            opts = rec.get("options") or {}
            ans = rec.get("answer") or rec.get("answer_letter")
            items.append(
                {
                    "id": qid,
                    "task": task,
                    "question": rec["question"],
                    "options": {str(k): str(v) for k, v in opts.items()},
                    "answer_letter": str(ans).strip().upper() if ans else None,
                }
            )
        out.setdefault(task, []).extend(items)
    return out


if __name__ == "__main__":
    bench = load_mirage()
    print(f"MIRAGE tasks: {len(bench)}")
    total = 0
    for task, items in sorted(bench.items()):
        covered = "exam (covered)" if task in EXAM_TASKS else "research (no PubMed corpus)"
        n_opts = len(items[0]["options"]) if items else 0
        print(f"  {task:<10} {len(items):>6} questions | {n_opts}-way | {covered}")
        total += len(items)
    print(f"  {'TOTAL':<10} {total:>6}")
