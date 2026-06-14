"""Build evals/results/SUMMARY_gemini.md from the combined results_ragas.jsonl.

Per-row table (id | judge | faithfulness | answer_relevancy | context_precision),
plus means split by judge (Gemini-judged n, Groq-judged n) and a blended 14-row
"mixed judge" mean. Pure reporting — never re-scores.
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
R = RESULTS_DIR / "results_ragas.jsonl"
S = RESULTS_DIR / "SUMMARY_gemini.md"
METRICS = ("faithfulness", "answer_relevancy", "context_precision")


def mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def clean_judge(tag):
    # the writer tagged gemini rows "gemini-<model>" where <model> already starts
    # with "gemini-" -> collapse the redundant prefix for display
    return tag.replace("gemini-gemini-", "gemini-")


def main():
    rows = [json.loads(l) for l in R.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["id"])
    gem_model = clean_judge(
        next((r["judge"] for r in rows if r["judge"].startswith("gemini")), "gemini"))

    lines = ["# RAGAS results — mixed judge (Gemini + Groq)", ""]
    lines.append(f"Gemini judge model: **{gem_model}** | Groq judge: **llama-3.3-70b-versatile**")
    lines.append("")
    lines.append("Answer-generating agent: Groq **llama-3.1-8b-instant**, temperature 0 "
                 "(answers cached in `eval_answers_progress.jsonl`).")
    lines.append("")
    lines.append("| id | judge | faithfulness | answer_relevancy | context_precision |")
    lines.append("|----|-------|-------------:|-----------------:|------------------:|")
    for r in rows:
        lines.append(f"| {r['id']} | {clean_judge(r['judge'])} | {r['faithfulness']} | "
                     f"{r['answer_relevancy']} | {r['context_precision']} |")
    lines.append("")

    gem = [r for r in rows if r["judge"].startswith("gemini")]
    groq = [r for r in rows if r["judge"].startswith("groq")]
    lines.append("## Means")
    lines.append("")
    lines.append("| group | n | faithfulness | answer_relevancy | context_precision |")
    lines.append("|-------|--:|-------------:|-----------------:|------------------:|")
    for label, grp in (("Gemini-judged", gem), ("Groq-judged", groq),
                       ("Blended (mixed judge)", rows)):
        lines.append(f"| {label} | {len(grp)} | "
                     f"{mean([r['faithfulness'] for r in grp])} | "
                     f"{mean([r['answer_relevancy'] for r in grp])} | "
                     f"{mean([r['context_precision'] for r in grp])} |")
    lines.append("")
    lines.append(f"Total rows judged: **{len(rows)}/14**. Means are computed over "
                 "non-null metric values; the blended row mixes two judges and is "
                 "reported as a coverage figure, not a single-judge benchmark.")
    lines.append("")
    S.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {S}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
