"""GPU-batched medical embedders (documents AND queries), selectable for ablation.

Two embedders, chosen by `kind`:
  - "general": sentence-transformers nomic-embed-text-v1.5 (symmetric, task prefixes
    "search_document: " / "search_query: ", normalized). Index: chroma_med/.
  - "medcpt": ncbi/MedCPT-Article-Encoder for documents + MedCPT-Query-Encoder for
    queries (asymmetric, CLS pooling, normalized). Index: chroma_med_medcpt/.

The retriever MUST embed queries with the SAME embedder the index was built with,
or query and document vectors land in different spaces. Finance keeps its separate
Ollama embedder.

No fp16/.half() — this GTX 1660 has a weak fp16 path; fp32 throughout.
"""

import threading

import numpy as np
from langchain_core.embeddings import Embeddings

MEDICAL_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"


def bucketed_encode(texts, encode_batch, *, budget, max_bs):
    """Length-bucketed encoding (returns numpy in INPUT order).

    Attention memory is ~O(batch * seq^2); a single batch size either starves the
    GPU on short snippets or spills VRAM on long ones (on Windows the driver
    silently pages to system RAM -> ~5x slowdown rather than a clean OOM). So sort
    by length and size each sub-batch from its longest member, keeping
    batch*maxlen^2 ~ constant: big batches for short snippets, small for long.
    `encode_batch(list[str]) -> np.ndarray` does the actual model forward.
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    i = 0
    while i < len(order):
        head = len(texts[order[i]])
        bs = max(8, min(max_bs, budget // max(head * head, 1)))
        idx = order[i : i + bs]
        longest = max(len(texts[j]) for j in idx)
        bs = max(8, min(bs, budget // max(longest * longest, 1)))
        idx = order[i : i + bs]
        vecs = encode_batch([texts[j] for j in idx])
        for j, v in zip(idx, vecs):
            out[j] = v
        i += len(idx)
    return np.asarray(out)


class NomicSTEmbeddings(Embeddings):
    """sentence-transformers nomic-embed-text-v1.5 on CUDA, OOM-resilient."""

    def __init__(self, device=None, batch_size=256):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            MEDICAL_EMBED_MODEL, trust_remote_code=True, device=self.device
        )
        self.model.max_seq_length = 512  # snippets <=1000 chars; cap long-pad outliers
        self.batch_size = batch_size

    def encode(self, texts, batch_size=None):
        import torch

        bs = batch_size or self.batch_size
        while True:
            try:
                return self.model.encode(
                    texts,
                    batch_size=bs,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs <= 64:
                    raise
                bs //= 2

    def embed_documents(self, texts):
        return self.encode([DOC_PREFIX + t for t in texts]).tolist()

    def embed_query(self, text):
        return self.encode([QUERY_PREFIX + text])[0].tolist()

    def encode_bucketed(self, texts, *, budget=50_000_000, max_bs=512):
        return bucketed_encode(
            texts,
            lambda batch: self.encode([DOC_PREFIX + t for t in batch], batch_size=len(batch)),
            budget=budget,
            max_bs=max_bs,
        )


class MedCPTEmbeddings(Embeddings):
    """ncbi MedCPT asymmetric bi-encoder (Article for docs, Query for queries).

    CLS-pooled, L2-normalized 768-d vectors. Article max_len 512, Query max_len 64
    (per the model cards). fp32.
    """

    def __init__(self, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.art_tok = AutoTokenizer.from_pretrained(MEDCPT_ARTICLE_MODEL)
        self.art_model = AutoModel.from_pretrained(MEDCPT_ARTICLE_MODEL).to(self.device).eval()
        self.qry_tok = AutoTokenizer.from_pretrained(MEDCPT_QUERY_MODEL)
        self.qry_model = AutoModel.from_pretrained(MEDCPT_QUERY_MODEL).to(self.device).eval()

    def _encode(self, texts, model, tok, max_len):
        import torch

        with torch.no_grad():
            enc = tok(
                texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt"
            ).to(self.device)
            cls = model(**enc).last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, dim=1)
        return cls.cpu().numpy()

    def _encode_docs(self, texts):
        import torch

        bs = len(texts)
        while True:
            try:
                return self._encode(texts, self.art_model, self.art_tok, 512)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs <= 16:
                    raise
                bs //= 2  # bucketed_encode already sizes batches; this is a backstop
                # re-encode the (already small) list in halves
                mid = len(texts) // 2
                return np.concatenate(
                    [self._encode_docs(texts[:mid]), self._encode_docs(texts[mid:])]
                )

    def embed_documents(self, texts):
        return self._encode(texts, self.art_model, self.art_tok, 512).tolist()

    def embed_query(self, text):
        return self._encode([text], self.qry_model, self.qry_tok, 64)[0].tolist()

    def encode_bucketed(self, texts, *, budget=30_000_000, max_bs=256):
        return bucketed_encode(texts, self._encode_docs, budget=budget, max_bs=max_bs)


_cache = {}
_lock = threading.Lock()


def get_medical_embeddings(kind="general") -> Embeddings:
    """Process-wide singleton per embedder kind (model loads are expensive)."""
    kind = (kind or "general").strip().lower()
    if kind not in _cache:
        with _lock:
            if kind not in _cache:
                _cache[kind] = (
                    MedCPTEmbeddings() if kind == "medcpt" else NomicSTEmbeddings()
                )
    return _cache[kind]
