"""Retrieval-only ablation general vs MedCPT — NO LLM/Groq needed.

For a fixed sample of golden + MIRAGE-exam questions, compares the two bi-encoders
on two LLM-free signals over the reranked top-5:
  - mean ms-marco cross-encoder rerank score (higher = passages more relevant)
  - StatPearls hit-rate in top-5 (clinical-source coverage)

    python evals/retrieval_ablation.py
"""

import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_retriever_vs(kind):
    from langchain_chroma import Chroma

    from agent.config import medical_index
    from agent.embeddings import get_medical_embeddings

    d, c = medical_index(kind)
    return Chroma(collection_name=c, persist_directory=str(d),
                  embedding_function=get_medical_embeddings(kind))


def sample_questions(n_per_task=8):
    import os

    os.environ["GOLDEN_PATH"] = "evals/golden/golden_medical.jsonl"
    from evals.load_golden import load_golden
    from evals.mirage.load_mirage import EXAM_TASKS, load_mirage

    qs = [r["question"] for r in load_golden() if r["type"] != "unanswerable"]
    bench = load_mirage(tasks=list(EXAM_TASKS))
    for task in EXAM_TASKS:
        qs += [it["question"] for it in bench[task][:n_per_task]]
    return qs


def main():
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    ce = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=ce, top_n=5)

    questions = sample_questions()
    print(f"Sample: {len(questions)} questions (golden answerable + MIRAGE exam)\n")

    results = {}
    for kind in ("general", "medcpt"):
        vs = build_retriever_vs(kind)
        rerank_means, sp_rates = [], []
        for q in questions:
            cand = vs.as_retriever(search_kwargs={"k": 20}).invoke(q)
            top5 = reranker.compress_documents(cand, q)
            scores = ce.score([(q, d.page_content) for d in top5])
            rerank_means.append(float(np.mean(scores)))
            sp_rates.append(sum(d.metadata.get("source") == "statpearls" for d in top5) / len(top5))
        results[kind] = {
            "mean_rerank_score": round(float(np.mean(rerank_means)), 4),
            "statpearls_hit_rate_top5": round(float(np.mean(sp_rates)), 4),
        }
        print(f"{kind:<9} mean_rerank={results[kind]['mean_rerank_score']:.4f} "
              f"statpearls_hit_rate={results[kind]['statpearls_hit_rate_top5']:.3f}")

    print(f"\n{'metric':<28}{'general':>10}{'medcpt':>10}{'delta':>10}")
    for m in ("mean_rerank_score", "statpearls_hit_rate_top5"):
        g, c = results["general"][m], results["medcpt"][m]
        print(f"{m:<28}{g:>10.4f}{c:>10.4f}{c - g:>+10.4f}")

    import json

    out = Path(__file__).resolve().parent / "results" / "retrieval_ablation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"n": len(questions), "results": results}, indent=2),
                   encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
