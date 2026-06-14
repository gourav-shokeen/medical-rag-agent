"""Retriever over the medical corpus (CORPUS env).

Reuses a persisted Chroma index (nomic-embed-text / MedCPT embeddings) + an
ms-marco cross-encoder reranker. No re-chunking/re-indexing here.

- medical: vector top-20 -> rerank top-5, with an optional `source` filter.

Return type: list[langchain_core.documents.Document] with .page_content +
.metadata (source/title/snippet_id).
"""

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from agent.config import CORPUS


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
        raise ValueError(f"Unsupported CORPUS={CORPUS!r}; expected 'medical'.")
    return _retriever
