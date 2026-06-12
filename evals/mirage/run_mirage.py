"""Run the MIRAGE MCQ benchmark through the agent (MCQ mode).

    python evals/mirage/run_mirage.py --sample-per-task 3 --retriever general   # smoke
    python evals/mirage/run_mirage.py --full --retriever general                # full (needs key)
    python evals/mirage/run_mirage.py --offline-smoke                           # no Groq

Scores agent.predicted_option vs the gold answer letter, per task, and prints a
table with published RAG baselines for reference. The retriever is swappable
(general nomic vs MedCPT) so the SAME questions can be run on either index.

CAVEATS (printed with results):
  - This corpus is medical textbooks + StatPearls only; it has NO PubMed corpus,
    so the research tasks (PubMedQA, BioASQ) retrieve poorly — default to the 3
    exam tasks (MMLU-Med, MedQA-US, MedMCQA).
  - Published baselines (GPT-4 ~79.97, GPT-3.5 ~71.56, Llama2-70B ~53.38, MedRAG
    paper avg over MMLU/MedQA/MedMCQA/PubMedQA/BioASQ) used the full MedCorp
    corpus and different generators, so this is NOT an identical setup — treat
    them as orientation, not a head-to-head.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evals.mirage.load_mirage import EXAM_TASKS, load_mirage  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASELINES = {"GPT-4": 79.97, "GPT-3.5": 71.56, "Llama2-70B": 53.38}  # MedRAG paper, ref only


def offline_smoke(tasks, n):
    """Prove retrieval + MCQ-letter parsing work WITHOUT calling Groq."""
    from agent.graph import _parse_option
    from agent.retriever import get_retriever

    print("OFFLINE SMOKE (no Groq): retrieval + option parsing only\n")
    retr = get_retriever()
    bench = load_mirage(tasks=tasks)
    for task, items in bench.items():
        for item in items[:n]:
            docs = retr.invoke(item["question"])
            letters = list(item["options"].keys())
            mock = f"The answer is {letters[0]} based on the passages."
            parsed = _parse_option(mock, letters)
            print(
                f"[{task}] retrieved {len(docs)} docs | parse('{mock[:24]}...')={parsed} "
                f"| top=[{docs[0].metadata.get('source')}: {docs[0].metadata.get('title','')[:40]}]"
            )
    print("\nOffline smoke OK. Full Groq run pending GROQ_API_KEY + --full.")


def run(tasks, sample_per_task, retriever):
    from agent.graph import run_agent

    bench = load_mirage(tasks=tasks)
    per_task, details = {}, []
    for task, items in bench.items():
        rows = items if sample_per_task <= 0 else items[:sample_per_task]
        correct = 0
        for i, item in enumerate(rows, 1):
            if not item["answer_letter"]:
                continue
            t0 = time.perf_counter()
            out = run_agent(item["question"], options=item["options"], choice_only=True)
            ok = out.get("predicted_option") == item["answer_letter"]
            correct += ok
            details.append(
                {
                    "task": task,
                    "id": item["id"],
                    "predicted": out.get("predicted_option"),
                    "gold": item["answer_letter"],
                    "correct": ok,
                    "latency_ms": round((time.perf_counter() - t0) * 1000),
                }
            )
            print(f"  [{task} {i}/{len(rows)}] pred={out.get('predicted_option')} "
                  f"gold={item['answer_letter']} {'OK' if ok else 'X'}", flush=True)
        n = sum(1 for d in details if d["task"] == task and d["gold"])
        per_task[task] = {"n": n, "correct": correct, "acc": round(100 * correct / n, 2) if n else None}
    return per_task, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(EXAM_TASKS))
    ap.add_argument("--sample-per-task", type=int, default=3)
    ap.add_argument("--full", action="store_true", help="all questions (ignores --sample-per-task)")
    ap.add_argument("--retriever", choices=("general", "medcpt"), default="general")
    ap.add_argument("--offline-smoke", action="store_true")
    args = ap.parse_args()

    os.environ["RETRIEVER"] = args.retriever  # config reads this at import
    os.environ.setdefault("CORPUS", "medical")

    if args.offline_smoke:
        offline_smoke(args.tasks, max(args.sample_per_task, 1))
        return

    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY not set -> running OFFLINE smoke instead.\n")
        offline_smoke(args.tasks, max(args.sample_per_task, 1))
        return

    os.environ.setdefault("LLM_PROVIDER", "groq")
    sample = 0 if args.full else args.sample_per_task
    print(f"MIRAGE | retriever={args.retriever} | tasks={args.tasks} | "
          f"{'FULL' if args.full else f'sample {sample}/task'}\n")

    per_task, details = run(args.tasks, sample, args.retriever)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"mirage_{args.retriever}_{stamp}.json"
    out_path.write_text(json.dumps({"retriever": args.retriever, "per_task": per_task,
                                    "details": details}, indent=2), encoding="utf-8")

    print("\n=== MIRAGE results (retriever=%s) ===" % args.retriever)
    print(f"{'task':<12}{'n':>6}{'correct':>9}{'acc%':>8}")
    accs = []
    for task, r in sorted(per_task.items()):
        print(f"{task:<12}{r['n']:>6}{r['correct']:>9}{str(r['acc']):>8}")
        if r["acc"] is not None:
            accs.append(r["acc"])
    if accs:
        print(f"{'MEAN':<12}{'':>6}{'':>9}{round(sum(accs)/len(accs),2):>8}")
    print("\nPublished baselines (reference only, different corpus+generator):")
    for k, v in BASELINES.items():
        print(f"  {k:<12}{v:>6}")
    print("\nCAVEAT: textbooks+StatPearls only (no PubMed) -> research tasks limited; "
          "baselines used full MedCorp, so not an identical setup.")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
