"""Corpus selection: CORPUS env = "medical" (default) | "finance".

The finance Chroma index (chroma_db/) is untouched; the medical index lives in
its own directory + collection so the two coexist.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent

VALID_CORPORA = ("medical", "finance")

CORPUS = os.getenv("CORPUS", "medical").strip().lower()
if CORPUS not in VALID_CORPORA:
    raise ValueError(f"CORPUS must be one of {VALID_CORPORA}, got {CORPUS!r}")

FINANCE_CHROMA_DIR = REPO_ROOT / "chroma_db"
# medical dir/collection are env-overridable so a throwaway index can be used
# (e.g. for smoke tests) without disturbing the main build
MEDICAL_CHROMA_DIR = Path(os.getenv("MEDICAL_CHROMA_DIR", REPO_ROOT / "chroma_med"))
MEDICAL_COLLECTION = os.getenv("MEDICAL_COLLECTION", "medical")

# finance index query embedder (Ollama)
EMBEDDING_MODEL = "nomic-embed-text"

# RETRIEVER selects the medical bi-encoder + its index, for the MedCPT ablation.
# "general" -> nomic-embed-text-v1.5 in chroma_med/ ; "medcpt" -> MedCPT in
# chroma_med_medcpt/. Everything else (snippets, reranker, k) is held identical
# so the ablation isolates only the embedding model.
VALID_RETRIEVERS = ("general", "medcpt")
RETRIEVER = os.getenv("RETRIEVER", "general").strip().lower()
if RETRIEVER not in VALID_RETRIEVERS:
    raise ValueError(f"RETRIEVER must be one of {VALID_RETRIEVERS}, got {RETRIEVER!r}")

MEDCPT_CHROMA_DIR = Path(os.getenv("MEDCPT_CHROMA_DIR", REPO_ROOT / "chroma_med_medcpt"))
MEDCPT_COLLECTION = os.getenv("MEDCPT_COLLECTION", "medical_medcpt")


def medical_index(retriever=None):
    """(persist_dir, collection) for the active (or given) medical retriever."""
    r = (retriever or RETRIEVER).strip().lower()
    if r == "medcpt":
        return MEDCPT_CHROMA_DIR, MEDCPT_COLLECTION
    return MEDICAL_CHROMA_DIR, MEDICAL_COLLECTION
