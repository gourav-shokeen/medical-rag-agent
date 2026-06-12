"""Existing SmartRetriever, extracted verbatim from rag.ipynb so the agent can import it.

Reuses the persisted Chroma index in chroma_db/ (nomic-embed-text embeddings) and the
same ms-marco cross-encoder reranker. No re-chunking or re-indexing happens here.
Pipeline: company detection -> metadata filter -> vector top-k 20 -> rerank top-5.
"""

from pathlib import Path

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_ollama import OllamaEmbeddings

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

COMPANY_ALIASES = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft", "azure"],
    "NVDA": ["nvidia", "nvda"],
}


def detect_companies(q):
    ql = q.lower()
    return [t for t, aliases in COMPANY_ALIASES.items() if any(a in ql for a in aliases)]


class SmartRetriever:
    def __init__(self, vectorstore, reranker, k=20):
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.k = k

    def invoke(self, query):
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


_smart_retriever = None


def get_smart_retriever() -> SmartRetriever:
    """Lazy singleton: loading the cross-encoder and Chroma once per process."""
    global _smart_retriever
    if _smart_retriever is None:
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"Persisted Chroma index not found at {CHROMA_DIR}. "
                "Run the indexing cells in rag.ipynb first."
            )
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        cross_encoder = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)
        _smart_retriever = SmartRetriever(vectorstore, reranker, k=20)
    return _smart_retriever
