# Engineering decisions

This document records *why* the system is built the way it is — the tradeoffs, the things that were tried and rejected, and the evidence behind each call. The README says what it does; this says why.

## 1. Vector retrieval + reranking — not hybrid search

**Decision:** The pipeline uses dense vector retrieval followed by a cross-encoder reranker. Hybrid search (vector + BM25) was implemented, measured, and **dropped**.

**Why:** The obvious move for financial filings is hybrid search — BM25 to catch exact tokens (line items, dollar figures, "Item 7A") that semantic search fumbles. I built it and measured it against the baseline. On these three filings it *lowered* company precision (0.45 → 0.40), because BM25 surfaced boilerplate — tables of contents, SEC-website notices, regulation references — where query words appear incidentally but the chunk isn't actually relevant. Tuning the fusion weights (`[0.6, 0.4]` → `[0.7, 0.3]`) didn't recover it.

So hybrid was the wrong tool *for this corpus and test set*. The reranker — which re-reads each (question, chunk) pair jointly — fixes the same precision problem far better, because it judges actual relevance rather than term overlap.

**Tradeoff / caveat:** This is corpus- and query-dependent. Hybrid likely earns its place on a test set that stresses exact-token lookups (specific figures, exact section codes). On this semantically-phrased test set it didn't, so it was cut. The honest result is kept rather than the assumed one.

## 2. Company-aware query routing

**Decision:** Before retrieval, detect which company a question names and filter the search to that company's chunks.

**Why:** The core failure of naive retrieval here was **cross-company bleed** — asking about Apple returned Microsoft and NVIDIA chunks, because every 10-K's risk and finance language reads almost identically, so the embedding model can't distinguish them by meaning. Filtering to the named company makes wrong-company results structurally impossible, taking company precision from 0.63 (reranking alone) to 0.97.

**Tradeoff:** Routing only helps when the question names a company. For company-ambiguous questions ("which segment had the strongest cloud growth?") there's no filter to apply, and the reranker carries the load alone — these remain the hardest case, and the 0.97 (not 1.0) precision reflects exactly those.

## 3. Cross-encoder reranking as the precision driver

**Decision:** Retrieve a wide candidate pool (top 20), then a cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores and keeps the best 5.

**Why:** Bi-encoder retrieval (the vector search) embeds the question and chunks *separately* — fast, but it can rank the right chunk 8th. A cross-encoder reads question and chunk *together*, so it scores true relevance and promotes the right chunk to the top. Wide-net-then-filter is the standard production pattern: cheap recall first, expensive precision second.

**Tradeoff:** Latency. The cross-encoder scores every candidate, so query time rose from ~45 ms to ~120 ms. That's the real speed/accuracy decision in production retrieval, and it's measured in the benchmark rather than hidden. Running it on GPU keeps the cost small.

## 4. The reranker is fed vector candidates, not hybrid

**Decision:** The candidate pool feeding the reranker comes from vector search, not the hybrid retriever.

**Why:** Since hybrid added boilerplate noise (decision #1), feeding the reranker cleaner vector candidates gives it better raw material. A reranker can only pick good results from what it's given — garbage candidates cap its ceiling.

## 5. Local models over hosted APIs

**Decision:** Embeddings (`nomic-embed-text`), the reranker, and the LLM (`llama3.2`) all run locally via Ollama / sentence-transformers. No API keys.

**Why:** The corpus is public (SEC EDGAR), the models are small enough to run on a 6 GB GPU, and the embedding step is call-heavy (1,162 chunks). Local means free, no rate limits, full privacy, and a project anyone can reproduce without billing setup.

**Tradeoff:** `llama3.2` (3B) was chosen over `llama3.1:8b` specifically for VRAM headroom on a GTX 1660 — the 8B model fits but is tight and sluggish. The chat-model interface (`ChatOllama`) was used so the LLM backend is swappable: a one-line change points it at Groq or any API with no other code changes.

## 6. LangChain pinned to 0.3.x

**Decision:** `langchain>=0.3,<1.0` and the related packages pinned below 1.0.

**Why:** LangChain 1.0 (GA Oct 2025) moved the "classic" retrievers — `EnsembleRetriever`, `ContextualCompressionRetriever`, `CrossEncoderReranker`, `MultiQueryRetriever` — out of the core package into `langchain-classic`. The retrieval techniques this project depends on live in that legacy surface either way, so 1.0 buys nothing here and only adds import churn. Pinning to 0.3.x keeps the code working as written and the build reproducible.

## 7. Chunking: 1,000 characters, 200 overlap

**Decision:** `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`.

**Why:** ~1,000 characters is large enough to keep a figure with its label and surrounding context, small enough that retrieval stays precise and the LLM's context isn't wasted. The 200-char overlap is insurance against boundary loss — in filings, a label and its dollar figure often straddle a line break, and overlap ensures the pair survives in at least one chunk. The recursive splitter prefers natural boundaries (paragraphs, then sentences) over blind character cuts, keeping chunks coherent.

## 8. Source document: primary HTML, never the full submission

**Decision:** Load each filing's `primary-document.html`, explicitly *not* `full-submission.txt`.

**Why:** EDGAR's `full-submission.txt` is the entire SGML submission — the 10-K *plus every exhibit and XBRL block* — which inflated the corpus ~15x (34,610 chunks of mostly noise) and embedded SGML headers as if they were filing prose. Pointing firmly at the primary document dropped it to a clean 1,162 chunks of actual annual-report text. Garbage in, garbage out: this was the difference between a working retriever and a broken one.

## 9. Honesty guard via distance threshold

**Decision:** If the closest retrieved chunk is farther than a distance threshold (calibrated empirically at 0.8), refuse to answer without calling the LLM.

**Why:** In finance, a confident wrong answer is worse than "I don't know." The threshold was set by observing actual distances — on-topic questions scored ~0.6, off-topic ~0.94 — and placing the cutoff in the gap. The prompt also instructs the model to refuse when context is insufficient, so there are two layers of refusal.

**Tradeoff:** `nomic-embed-text` produces a compressed distance range, so the on/off-topic gap is narrow — the threshold is a pragmatic guard, not a precise classifier. The prompt-level refusal is the more reliable safety net.

## 10. Tooling: uv

**Decision:** `uv` for environment and dependency management, with a committed `uv.lock`.

**Why:** Reproducibility is the point of a portfolio repo — `git clone` → `uv sync` → it runs, with exact pinned versions. uv is also dramatically faster than pip for the heavy ML dependency tree.

**Note on CUDA torch:** `sentence-transformers` pulls a CPU build of torch by default. Getting the GPU build required declaring `torch` as a *direct* dependency and pointing it at PyTorch's CUDA index in `pyproject.toml` — `[tool.uv.sources]` only redirects packages the project explicitly declares, so a transitive torch never picks up the override.

## 11. Evaluation honesty

**Decision:** Report the small-test-set numbers with explicit caveats rather than presenting them as a general benchmark.

**Why:** The test set is ~23 hand-built questions over three filings. Company precision of 0.97 is high partly *by construction* (filtering guarantees correct-company retrieval for questions that name a company). A benchmark is only as good as its test set, and overselling it would be the opposite of the rigor this project is meant to demonstrate. Two test questions were also found mid-evaluation to be measuring the wrong thing (an arbitrary keyword; an under-specified company) and were corrected, with all configurations re-measured against the fixed set.
