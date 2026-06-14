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

from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # pick up GROQ_API_KEY from .env

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


import re

QUOTA_MARKERS = ("rate limit", "rate_limit", "ratelimit", "429", "quota",
                 "tokens per day", "tpd", "too many requests")


def _is_quota_error(exc) -> bool:
    return any(m in str(exc).lower() for m in QUOTA_MARKERS)


def _retry_seconds(exc):
    """Parse Groq's 'try again in 1m2.3s' / 'try again in 4.5s' hint."""
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(exc).lower())
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + float(m.group(2))


def _is_terminal_quota(exc) -> bool:
    """Daily (TPD/RPD) limit = terminal until reset; per-minute = transient."""
    s = str(exc).lower()
    if "per day" in s or "tpd" in s or "requests per day" in s or "rpd" in s:
        return True
    wait = _retry_seconds(exc)
    return wait is not None and wait > 120  # long wait => daily reset, not a minute


def _score_with_backoff(run_agent, item, max_wait_rounds=6):
    """Run one MCQ question; on a transient per-minute 429, sleep + retry.
    Raises SystemExit on a terminal (daily) quota limit."""
    import time as _t

    for _ in range(max_wait_rounds):
        try:
            return run_agent(item["question"], options=item["options"], choice_only=True)
        except Exception as exc:
            if not _is_quota_error(exc):
                raise
            if _is_terminal_quota(exc):
                print(f"\nQUOTA EXHAUSTED (daily) — resumable, relaunch after reset. "
                      f"({str(exc)[:90]})", flush=True)
                raise SystemExit
            wait = min((_retry_seconds(exc) or 20) + 2, 75)
            print(f"    per-minute rate limit; sleeping {wait:.0f}s...", flush=True)
            _t.sleep(wait)
    # exhausted local retries without success -> treat as terminal for safety
    print("\nToo many per-minute rate limits — stopping (resumable).", flush=True)
    raise SystemExit


def progress_path(retriever):
    return RESULTS_DIR / f"mirage_{retriever}_progress.jsonl"


def load_done(retriever):
    """Return {(task, id): record} already scored (per-question resume)."""
    path = progress_path(retriever)
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["task"], r["id"])] = r
    return done


def run(tasks, sample_per_task, retriever):
    """Run MIRAGE, resuming from the per-question checkpoint. Returns
    (records, quota_exhausted). Each completed question is appended to the
    progress JSONL immediately, so a crash/quota-exit loses nothing."""
    from agent.graph import run_agent

    RESULTS_DIR.mkdir(exist_ok=True)
    bench = load_mirage(tasks=tasks)
    done = load_done(retriever)
    if done:
        print(f"Resuming: {len(done)} questions already scored, skipping them.\n")
    records = list(done.values())
    pfile = progress_path(retriever).open("a", encoding="utf-8")
    tokens = 0
    quota_exhausted = False

    try:
        for task, items in bench.items():
            rows = items if sample_per_task <= 0 else items[:sample_per_task]
            for i, item in enumerate(rows, 1):
                if not item["answer_letter"] or (task, item["id"]) in done:
                    continue
                t0 = time.perf_counter()
                try:
                    out = _score_with_backoff(run_agent, item)
                except SystemExit:
                    quota_exhausted = True
                    raise
                except Exception as exc:
                    print(f"  [{task} {i}] ERROR (skipped): {str(exc)[:80]}", flush=True)
                    continue
                tokens += out.get("usage", {}).get("total_tokens", 0)
                rec = {
                    "task": task, "id": item["id"],
                    "predicted": out.get("predicted_option"),
                    "gold": item["answer_letter"],
                    "correct": out.get("predicted_option") == item["answer_letter"],
                    "latency_ms": round((time.perf_counter() - t0) * 1000),
                }
                records.append(rec)
                pfile.write(json.dumps(rec) + "\n")
                pfile.flush()
                done[(task, item["id"])] = rec
                if len(records) % 10 == 0:
                    print(f"  [{task} {i}/{len(rows)}] scored={len(records)} "
                          f"tokens~{tokens:,}", flush=True)
    except SystemExit:
        pass
    finally:
        pfile.close()
    print(f"\nTotal tokens this run: ~{tokens:,}")
    return records, quota_exhausted


def summarize(records):
    per_task = {}
    for r in records:
        t = per_task.setdefault(r["task"], {"n": 0, "correct": 0})
        t["n"] += 1
        t["correct"] += bool(r["correct"])
    for t in per_task.values():
        t["acc"] = round(100 * t["correct"] / t["n"], 2) if t["n"] else None
    return per_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(EXAM_TASKS))
    ap.add_argument("--sample-per-task", type=int, default=3)
    ap.add_argument("--full", action="store_true", help="all questions (ignores --sample-per-task)")
    ap.add_argument("--retriever", choices=("general", "medcpt"), default="general")
    ap.add_argument("--offline-smoke", action="store_true")
    args = ap.parse_args()

    # task aliases: "exam" -> 3 covered exam tasks; "research" -> PubMed-dependent
    if args.tasks == ["exam"]:
        args.tasks = list(EXAM_TASKS)
    elif args.tasks == ["research"]:
        args.tasks = ["pubmedqa", "bioasq"]

    os.environ["RETRIEVER"] = args.retriever  # config reads this at import
    os.environ.setdefault("CORPUS", "medical")

    if args.offline_smoke:
        offline_smoke(args.tasks, max(args.sample_per_task, 1))
        return

    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY not set -> running OFFLINE smoke instead.\n")
        offline_smoke(args.tasks, max(args.sample_per_task, 1))
        return

    os.environ["LLM_PROVIDER"] = "groq"  # MIRAGE always runs under Groq (force over .env)
    # cheap, high-quota model for the agent unless the caller overrides GROQ_MODEL
    os.environ.setdefault("GROQ_MODEL", "llama-3.1-8b-instant")
    sample = 0 if args.full else args.sample_per_task
    print(f"MIRAGE | retriever={args.retriever} | tasks={args.tasks} | "
          f"model={os.environ['GROQ_MODEL']} | "
          f"{'FULL' if args.full else f'sample {sample}/task'}\n")

    records, quota_exhausted = run(args.tasks, sample, args.retriever)
    per_task = summarize(records)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"mirage_{args.retriever}_{stamp}.json"
    out_path.write_text(json.dumps(
        {"retriever": args.retriever, "model": os.environ["GROQ_MODEL"],
         "quota_exhausted": quota_exhausted, "n_scored": len(records),
         "per_task": per_task, "details": records}, indent=2), encoding="utf-8")

    print("\n=== MIRAGE results (retriever=%s, n=%d%s) ===" %
          (args.retriever, len(records), ", PARTIAL/quota" if quota_exhausted else ""))
    print(f"{'task':<12}{'n':>6}{'correct':>9}{'acc%':>8}")
    accs = []
    for task, r in sorted(per_task.items()):
        print(f"{task:<12}{r['n']:>6}{r['correct']:>9}{str(r['acc']):>8}")
        if r["acc"] is not None and task in EXAM_TASKS:
            accs.append(r["acc"])
    if accs:
        print(f"{'EXAM-MEAN':<12}{'':>6}{'':>9}{round(sum(accs)/len(accs),2):>8}")
    print("\nPublished baselines (full MedCorp incl. PubMed + different generator — "
          "reference, NOT identical setup):")
    for k, v in BASELINES.items():
        print(f"  {k:<12}{v:>6}")
    print("\nCAVEAT: corpus is textbooks+StatPearls only (no PubMed) -> research tasks "
          "(pubmedqa/bioasq) are expected low; report them separately, never in the "
          "exam headline.")
    print(f"\nWrote {out_path}")
    if quota_exhausted:
        print(f"\n>>> RESUME after quota reset:\n"
              f"    CORPUS=medical RETRIEVER={args.retriever} python -m evals.mirage.run_mirage "
              f"--tasks {' '.join(args.tasks)} {'--full' if args.full else f'--sample-per-task {sample}'} "
              f"--retriever {args.retriever}\n    (skips the {len(records)} already-scored questions)")


if __name__ == "__main__":
    main()
