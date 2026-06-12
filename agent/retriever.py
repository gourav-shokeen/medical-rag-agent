"""SmartRetriever over the active corpus (CORPUS env), extracted from rag.ipynb.

Reuses a persisted Chroma index (nomic-embed-text embeddings) + the same ms-marco
cross-encoder reranker. No re-chunking/re-indexing here.

- finance: company detection -> per-company metadata filter -> top-20 -> rerank top-5
- medical: NO company logic; vector top-20 -> rerank top-5, optional `source` filter

Return type (both): list[langchain_core.documents.Document] with .page_content +
.metadata (finance: company/source/file; medical: source/title/snippet_id).
"""

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_ollama import OllamaEmbeddings

from agent.config import (
    CORPUS,
    EMBEDDING_MODEL,
    FINANCE_CHROMA_DIR,
    MEDICAL_CHROMA_DIR,
    MEDICAL_COLLECTION,
)

# kept module-level for backwards-compat (finance domain + tests import this)
COMPANY_ALIASES = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft", "azure"],
    "NVDA": ["nvidia", "nvda"],
}


def detect_companies(q):
    ql = q.lower()
    return [t for t, aliases in COMPANY_ALIASES.items() if any(a in ql for a in aliases)]


class SmartRetriever:
    """Finance corpus: company-aware retrieval (original behavior, unchanged)."""

    def __init__(self, vectorstore, reranker, k=20):
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.k = k

    def invoke(self, query, source=None):  # source ignored (finance has no source filter)
        companies = detect_companies(query)
        if len(companies) == 0:
            candidates = self.vectorstore.as_retriever(
                search_kwargs={"k": self.k}
            ).invoke(query)
        else:
            candidates = []
            per_company_k = max(self.k // len(companies), 8)
            for c in companies:
                hits = self.vectorstore.as_retriever(
                    search_kwargs={"k": per_company_k, "filter": {"company": c}}
                ).invoke(query)
                candidates.extend(hits)
        return self.reranker.compress_documents(candidates, query)


class MedicalRetriever:
    """Medical corpus: plain vector top-k -> rerank, optional `source` filter."""

    def __init__(self, vectorstore, reranker, k=20):
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.k = k

    def invoke(self, query, source=None):
        kwargs = {"k": self.k}
        if source:
            kwargs["filter"] = {"source": source}
        candidates = self.vectorstore.as_retriever(search_kwargs=kwargs).invoke(query)
        return self.reranker.compress_documents(candidates, query)


_retriever = None


def _build_reranker():
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoderReranker(model=cross_encoder, top_n=5)


def get_retriever():
    """Lazy singleton for the active corpus (loads cross-encoder + Chroma once)."""
    global _retriever
    if _retriever is not None:
        return _retriever

    if CORPUS == "medical":
        # RETRIEVER selects the bi-encoder + its index (general nomic vs MedCPT);
        # queries MUST use the same embedder the index was built with, or vectors
        # mismatch. Everything downstream (reranker, k) is identical -> clean ablation.
        from agent.config import RETRIEVER, medical_index
        from agent.embeddings import get_medical_embeddings

        med_dir, med_collection = medical_index(RETRIEVER)
        if not med_dir.exists():
            raise FileNotFoundError(
                f"Medical index not found at {med_dir} (RETRIEVER={RETRIEVER}). "
                "Run: python ingest/build_index_fast.py"
                + (" --embedder medcpt" if RETRIEVER == "medcpt" else "")
            )
        vectorstore = Chroma(
            collection_name=med_collection,
            persist_directory=str(med_dir),
            embedding_function=get_medical_embeddings(RETRIEVER),
        )
        _retriever = MedicalRetriever(vectorstore, _build_reranker(), k=20)
    else:
        if not FINANCE_CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"Finance index not found at {FINANCE_CHROMA_DIR}. "
                "Run the indexing cells in rag.ipynb first."
            )
        vectorstore = Chroma(
            persist_directory=str(FINANCE_CHROMA_DIR),
            embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        )
        _retriever = SmartRetriever(vectorstore, _build_reranker(), k=20)
    return _retriever


# backwards-compatible alias (finance code/tests referenced this name)
def get_smart_retriever():
    return get_retriever()
