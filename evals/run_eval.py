"""RAGAS evaluation over the golden set: python evals/run_eval.py [--smoke-only]

Runs the agent on every golden question, scores answers with the fixed judge
(see evals/judge.py), writes evals/results/eval_<timestamp>.json + .csv, and
prints a summary table of metric averages.

RESUMABLE (Groq-quota-aware), two independently checkpointed phases:
  1. agent answers  -> evals/results/eval_answers_progress.jsonl   (one line / id)
  2. RAGAS scores   -> evals/results/eval_scores_progress.jsonl     (one line / id)
Re-launching skips any id already present in BOTH caches, so a 429/quota kill
mid-run loses nothing. Set GROQ_MODEL=llama-3.1-8b-instant to answer cheaply;
the judge stays pinned to 70B via evals/judge.py (JUDGE_MODEL).

Installed ragas is 0.4.3: EvaluationDataset still takes the v0.2+ keys
(user_input, retrieved_contexts, response, reference) and evaluate() still
accepts llm=/embeddings= wrappers. The classic metric INSTANCES moved to
private modules — these are the exact imports evaluate() itself uses for its
defaults, so they are the supported spelling for this version:
    ragas.metrics._faithfulness.faithfulness
    ragas.metrics._answer_relevance.answer_relevancy  (class: ResponseRelevancy)
    ragas.metrics._context_precision.context_precision
"""

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# allow `python evals/run_eval.py` from the repo root (script dir != root)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

# reuse the MIRAGE harness's battle-tested quota classification/backoff
from evals.mirage.run_mirage import (  # noqa: E402
    _is_quota_error,
    _is_terminal_quota,
    _retry_seconds,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ANSWERS_PATH = RESULTS_DIR / "eval_answers_progress.jsonl"
SCORES_PATH = RESULTS_DIR / "eval_scores_progress.jsonl"
METRIC_KEYS = ("faithfulness", "answer_relevancy", "context_precision")


def _load_jsonl(path):
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def _call_with_backoff(fn, max_wait_rounds=15):
    """Run fn(); on transient per-minute 429 sleep + retry; on daily quota exit.

    8b's TPM ceiling is only 6,000 tokens/minute, and ONE open-ended agent answer
    (3-5 calls, each carrying ~1.2k-char passage context) is ~6-8k tokens — enough
    to blow a full minute's budget by itself. The short Groq-hinted retry (a few
    seconds) therefore keeps re-failing because the window never drains, so we
    ESCALATE: each successive throttle on the same call waits longer (up to ~65s),
    forcing a clean minute-window reset. 15 rounds rides out the worst bursts.
    """
    for attempt in range(max_wait_rounds):
        try:
            return fn()
        except Exception as exc:
            if not _is_quota_error(exc):
                raise
            if _is_terminal_quota(exc):
                print(f"\nQUOTA EXHAUSTED (daily) — resumable, relaunch after reset. "
                      f"({str(exc)[:90]})", flush=True)
                raise SystemExit
            hinted = _retry_seconds(exc) or 15
            wait = min(max(hinted + 2, 20 + attempt * 10), 65)
            print(f"    per-minute rate limit; sleeping {wait:.0f}s "
                  f"(attempt {attempt + 1})...", flush=True)
            time.sleep(wait)
    print("\nToo many per-minute rate limits — stopping (resumable).", flush=True)
    raise SystemExit


def collect_agent_runs(rows):
    """Answer each golden row (resumable). Returns {id: answer-record}."""
    from agent.graph import run_agent

    RESULTS_DIR.mkdir(exist_ok=True)
    done = _load_jsonl(ANSWERS_PATH)
    if done:
        print(f"Resuming answers: {len(done)} already cached, skipping them.")
    pending = [r for r in rows if r["id"] not in done]
    if pending:
        afile = ANSWERS_PATH.open("a", encoding="utf-8")
        try:
            for i, row in enumerate(pending, 1):
                print(f"[answer {i}/{len(pending)}] {row['id']}: {row['question'][:60]}")
                t0 = time.perf_counter()
                out = _call_with_backoff(lambda: run_agent(row["question"]))
                rec = {
                    "id": row["id"], "type": row["type"],
                    "company": row.get("company"),  # finance-only; medical omits it
                    "user_input": row["question"],
                    "retrieved_contexts": out["contexts"],
                    "response": out["answer"],
                    "reference": row["ground_truth"],
                    "citations": out["citations"], "retries": out["retries"],
                    "grounded": out["grounded"], "latency_ms": out["latency_ms"],
                }
                afile.write(json.dumps(rec) + "\n")
                afile.flush()
                done[row["id"]] = rec
                print(f"    answered in {out['latency_ms'] / 1000:.1f}s "
                      f"(retries={out['retries']}, grounded={out['grounded']})")
        finally:
            afile.close()
    # preserve golden order
    return [done[r["id"]] for r in rows if r["id"] in done]


def _ragas_record(ans):
    return {
        "user_input": ans["user_input"],
        "retrieved_contexts": ans["retrieved_contexts"],
        "response": ans["response"],
        "reference": ans["reference"],
    }


def score_with_ragas(answers):
    """Score each answered row with RAGAS (resumable). Returns {id: {metric:val}}."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._faithfulness import faithfulness
    from ragas.run_config import RunConfig

    from evals.judge import JUDGE_PROVIDER, get_ragas_embeddings, get_ragas_llm

    def _is_real(rec):
        # a row counts as scored only if at least one metric came back non-null;
        # an ALL-null row is a swallowed-429 (RAGAS turns the judge's rate-limit
        # into NaN instead of raising) and must be re-scored on the next pass.
        return any(rec.get(m) is not None for m in METRIC_KEYS)

    done = _load_jsonl(SCORES_PATH)
    real = {k: v for k, v in done.items() if _is_real(v)}
    if real:
        print(f"Resuming scores: {len(real)} already scored, skipping them.")
    pending = [a for a in answers if not _is_real(done.get(a["id"], {}))]
    if not pending:
        return real

    # Serialize judge calls (max_workers=1) so a single row's 3 metrics don't
    # spike the 70B judge's 12k TPM all at once; the ollama path was already serial.
    run_config = RunConfig(timeout=900 if JUDGE_PROVIDER == "ollama" else 300,
                           max_workers=1)

    llm, embeddings = get_ragas_llm(), get_ragas_embeddings()
    metrics = [faithfulness, answer_relevancy, context_precision]
    sfile = SCORES_PATH.open("a", encoding="utf-8")
    try:
        # score one row at a time so each success is checkpointed immediately.
        for i, ans in enumerate(pending, 1):
            print(f"[score {i}/{len(pending)}] {ans['id']}")

            def _do():
                ds = EvaluationDataset.from_list([_ragas_record(ans)])
                res = evaluate(dataset=ds, metrics=metrics, llm=llm,
                               embeddings=embeddings, run_config=run_config)
                return res.to_pandas()

            # RAGAS swallows the judge's 429 into NaN, so _call_with_backoff never
            # sees it. Detect an all-null result and retry with a TPM-draining
            # sleep; only persist null after the judge window clearly won't yield.
            rec = None
            for attempt in range(5):
                df = _call_with_backoff(_do)
                scored = df.iloc[0]
                cand = {"id": ans["id"]}
                for m in METRIC_KEYS:
                    v = scored[m] if m in df.columns else None
                    cand[m] = (None if v is None or (isinstance(v, float) and math.isnan(v))
                               else round(float(v), 4))
                if _is_real(cand):
                    rec = cand
                    break
                wait = min(35 + attempt * 15, 75)
                print(f"    all-null (judge throttled); draining {wait}s "
                      f"(attempt {attempt + 1})...", flush=True)
                time.sleep(wait)
            if rec is None:
                rec = cand  # persist the null after exhausting retries (resumable)
            sfile.write(json.dumps(rec) + "\n")
            sfile.flush()
            done[ans["id"]] = rec
            print("    " + "  ".join(f"{m}={rec[m]}" for m in METRIC_KEYS))
            if JUDGE_PROVIDER != "ollama":
                time.sleep(8)  # pace rows to respect the 70B judge's TPM
    finally:
        sfile.close()
    return {a["id"]: done[a["id"]] for a in answers
            if a["id"] in done and _is_real(done[a["id"]])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    from evals.judge import JUDGE_MODEL, JUDGE_PROVIDER
    from evals.load_golden import load_golden

    rows = load_golden(smoke_only=args.smoke_only)
    print(f"Golden rows: {len(rows)} | judge: {JUDGE_PROVIDER}/{JUDGE_MODEL}\n")

    answers = collect_agent_runs(rows)
    print("\nScoring with RAGAS (fixed judge, temperature 0)...")
    scores = score_with_ragas(answers)

    # aggregate from the score cache (only rows we actually have)
    averages = {}
    for m in METRIC_KEYS:
        vals = [scores[a["id"]][m] for a in answers
                if a["id"] in scores and scores[a["id"]].get(m) is not None]
        averages[m] = round(sum(vals) / len(vals), 4) if vals else None

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = RESULTS_DIR / f"eval_{stamp}.json"
    csv_path = RESULTS_DIR / f"eval_{stamp}.csv"

    per_question = []
    for a in answers:
        entry = {k: a.get(k) for k in (
            "id", "type", "company", "user_input", "response", "reference",
            "citations", "retries", "grounded", "latency_ms")}
        sc = scores.get(a["id"], {})
        for m in METRIC_KEYS:
            entry[m] = sc.get(m)
        per_question.append(entry)

    json_path.write_text(json.dumps({
        "timestamp_utc": stamp, "judge_provider": JUDGE_PROVIDER,
        "judge_model": JUDGE_MODEL, "n_questions": len(rows),
        "n_scored": sum(1 for a in answers if a["id"] in scores),
        "averages": averages, "per_question": per_question,
    }, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_question[0].keys()))
        writer.writeheader()
        for entry in per_question:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v
                             for k, v in entry.items()})

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}\n")
    print(f"{'metric':<28}{'average':>10}")
    print("-" * 38)
    for k, v in averages.items():
        print(f"{k:<28}{str(v):>10}")


if __name__ == "__main__":
    main()
