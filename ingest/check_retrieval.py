"""Print top-5 source+title for the Step-3 clinical queries against chroma_med/.

    python ingest/check_retrieval.py            # via the agent MedicalRetriever (rerank)
    python ingest/check_retrieval.py --raw      # raw vector top-k (no rerank)

Use before vs after the StatPearls add to show the retrieval fix.
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUERIES = [
    "first-line treatment for community-acquired pneumonia in adults",
    "management of diabetic ketoacidosis",
    "first-line antihypertensive in a patient with diabetes",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="raw vector top-k, skip reranker")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    from ingest.build_medical_index import get_vectorstore

    from agent.config import RETRIEVER, medical_index

    med_dir, _ = medical_index(RETRIEVER)
    vs = get_vectorstore()
    total = vs._collection.count()
    n_sp = len(vs._collection.get(where={"source": "statpearls"}, include=[])["ids"])
    n_tb = len(vs._collection.get(where={"source": "textbook"}, include=[])["ids"])
    print(f"{med_dir.name}/ (retriever={RETRIEVER}) total vectors: {total:,}")
    print(f"  by source: statpearls={n_sp:,}  textbook={n_tb:,}  other={total - n_sp - n_tb:,}\n")

    rerank = None
    if not args.raw:
        from agent.retriever import _build_reranker

        rerank = _build_reranker()

    for q in QUERIES:
        print(f"QUERY: {q}")
        docs = vs.as_retriever(search_kwargs={"k": 20 if rerank else args.k}).invoke(q)
        if rerank:
            docs = rerank.compress_documents(docs, q)
        n_sp = sum(1 for d in docs[: args.k] if d.metadata.get("source") == "statpearls")
        for d in docs[: args.k]:
            print(f"  [{d.metadata.get('source')}: {d.metadata.get('title', '')[:80]}]")
        print(f"  -> statpearls in top-{args.k}: {n_sp}\n")


if __name__ == "__main__":
    main()
