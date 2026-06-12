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

# both indexes were/are built with this model — query embedder must match
EMBEDDING_MODEL = "nomic-embed-text"
