"""RAGAS evaluation over the golden set: python evals/run_eval.py [--smoke-only]

Runs the agent on every golden question, scores answers with the fixed judge
(see evals/judge.py), writes evals/results/eval_<timestamp>.json + .csv, and
prints a summary table of metric averages.

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

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def collect_agent_runs(rows):
    """Call run_agent per golden row; return ragas records + raw run info."""
    from agent.graph import run_agent

    records, runs = [], []
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['id']}: {row['question'][:70]}")
        t0 = time.perf_counter()
        out = run_agent(row["question"])
        print(
            f"    answered in {out['latency_ms'] / 1000:.1f}s "
            f"(retries={out['retries']}, grounded={out['grounded']})"
        )
        records.append(
            {
                "user_input": row["question"],
                "retrieved_contexts": out["contexts"],
                "response": out["answer"],
                "reference": row["ground_truth"],
            }
        )
        runs.append(
            {
                "id": row["id"],
                "type": row["type"],
                "company": row["company"],
                "question": row["question"],
                "answer": out["answer"],
                "citations": out["citations"],
                "retries": out["retries"],
                "grounded": out["grounded"],
                "latency_ms": out["latency_ms"],
                "agent_wall_s": round(time.perf_counter() - t0, 1),
            }
        )
    return records, runs


def score_with_ragas(records):
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._faithfulness import faithfulness
    from ragas.run_config import RunConfig

    from evals.judge import JUDGE_PROVIDER, get_ragas_embeddings, get_ragas_llm

    # A local single-stream judge (ollama) serializes requests, so ragas's
    # default concurrency makes queued jobs blow the 180s timeout. Groq is fast
    # and parallel; keep its defaults snappier.
    if JUDGE_PROVIDER == "ollama":
        run_config = RunConfig(timeout=900, max_workers=1)
    else:
        run_config = RunConfig(timeout=300, max_workers=4)

    dataset = EvaluationDataset.from_list(records)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
        run_config=run_config,
    )
    return result.to_pandas()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    from evals.judge import JUDGE_PROVIDER
    from evals.load_golden import load_golden

    rows = load_golden(smoke_only=args.smoke_only)
    print(f"Golden rows: {len(rows)} | judge provider: {JUDGE_PROVIDER}\n")

    records, runs = collect_agent_runs(rows)
    print("\nScoring with RAGAS (fixed judge, temperature 0)...")
    df = score_with_ragas(records)

    metric_cols = [
        c
        for c in df.columns
        if c not in ("user_input", "retrieved_contexts", "response", "reference")
    ]
    averages = {
        c: round(float(df[c].mean(skipna=True)), 4)
        for c in metric_cols
        if df[c].dtype.kind in "fi"
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = RESULTS_DIR / f"eval_{stamp}.json"
    csv_path = RESULTS_DIR / f"eval_{stamp}.csv"

    per_question = []
    for run, (_, scored) in zip(runs, df.iterrows()):
        entry = dict(run)
        for c in metric_cols:
            v = scored[c]
            entry[c] = None if isinstance(v, float) and math.isnan(v) else (
                round(float(v), 4) if isinstance(v, (int, float)) else v
            )
        per_question.append(entry)

    json_path.write_text(
        json.dumps(
            {
                "timestamp_utc": stamp,
                "judge_provider": JUDGE_PROVIDER,
                "n_questions": len(rows),
                "averages": averages,
                "per_question": per_question,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_question[0].keys()))
        writer.writeheader()
        for entry in per_question:
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, list) else v for k, v in entry.items()}
            )

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}\n")
    print(f"{'metric':<28}{'average':>10}")
    print("-" * 38)
    for k, v in averages.items():
        print(f"{k:<28}{v:>10}")


if __name__ == "__main__":
    main()
