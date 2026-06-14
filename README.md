# SEC 10-K Q&A — RAG with Cross-Encoder Reranking

Ask natural-language questions about real **SEC 10-K annual filings** (Apple, Microsoft, NVIDIA) and get **cited, source-grounded answers**. The system runs entirely locally — embeddings, retrieval, reranking, and the LLM — on a single consumer GPU.

The point of this project isn't "chat with a PDF." It's that retrieval quality was **measured**, a failure mode was found (the naive retriever returned the wrong company's text more than half the time), and it was **fixed with real information-retrieval technique** — taking company-attribution precision from **0.45 → 0.97** on a hand-built test set.

> **Headline result:** retrieval precision improved from **0.45 to 0.97** by adding a cross-encoder reranker and company-aware query routing — measured before and after, on the same test set.

<!-- TODO: add a screenshot of a cited answer here — it's the single most convincing thing a reader sees -->
<!-- ![Cited answer example](docs/cited-answer.png) -->

## Two corpora, one engine (`CORPUS` env)

The same agentic engine runs over two interchangeable corpora, selected by the `CORPUS` env var:

- **`CORPUS=finance`** — SEC 10-K filings (AAPL, MSFT, NVDA), with company-aware routing. Index: `chroma_db/`.
- **`CORPUS=medical`** (default) — **491,497** pre-chunked clinical snippets: **365,650** from StatPearls + **125,847** from [MedRAG/textbooks](https://huggingface.co/datasets/MedRAG/textbooks) (USMLE medical textbooks). No company logic; optional `source` filter. Index: `chroma_med/` (and a parallel `chroma_med_medcpt/` for the embedder ablation below).

Everything else is shared: the LangGraph self-correcting loop, the cross-encoder reranker, the structured-output graders, the FastAPI/Next.js/Gradio surfaces. Only the corpus, retrieval routing, and node prompts change — the finance path is preserved exactly.

**Build the medical index** (one-time, ~hours on a consumer GPU; resumable, idempotent):
```bash
python ingest/build_medical_index.py            # MedRAG/textbooks
# StatPearls is not redistributable via the HF Hub. To add it, build the chunks
# locally with the MedRAG toolkit and pass --statpearls-dir <chunk_dir>.
```

**Medical safety frame.** Every open-ended medical answer ends with a standing not-advice frame, and the generator refuses to give personalized diagnosis/treatment ("I have chest pain, what should I do?" → defers to a clinician / emergency care). Answers cite `[source: title]` from the snippet metadata, and unanswerable questions return exactly *"This is not covered in the available medical references."*

**MCQ mode.** `run_agent(question, options={"A": ...}, choice_only=True)` returns a single grounded `predicted_option` letter — the interface the MIRAGE benchmark calls.

## Architecture

```mermaid
flowchart TD
    A["SEC 10-K filings<br/>AAPL · MSFT · NVDA"] --> B["Load, clean & chunk<br/>1,000-char chunks + company metadata"]
    B --> C["Embed & store<br/>nomic-embed-text → Chroma (1,162 chunks)"]
    C --> D["Company router<br/>filter to the named filing"]
    D --> E["Vector search<br/>retrieve top 20 candidates"]
    E --> F["Cross-encoder rerank<br/>ms-marco-MiniLM, keep best 5 (GPU)"]
    F --> G["Grounded LLM + honesty guard<br/>llama3.2 · refuses if unsupported"]
    G --> H["Cited answer + sources<br/>claims tagged by company"]
```

The first two stages run **once** to build the index. The rest runs **per question**.

## The agentic layer (LangGraph)

On top of the retriever sits a **self-correcting agent** (`agent/`). Instead of one retrieve→generate pass, the agent grades its own evidence and rewrites its query when retrieval comes back weak:

```mermaid
flowchart TD
    S([START]) --> R["route<br/>detect company / intent"]
    R --> T["retrieve<br/>SmartRetriever: filter → top-20 → rerank top-5"]
    T --> G["grade_documents<br/>judge: is the evidence sufficient?"]
    G -- "insufficient · retries < 2" --> W["rewrite_query<br/>expand abbreviations, add 10-K terms"]
    W --> T
    G -- "sufficient · or retry cap hit" --> A["generate<br/>grounded answer, inline citations"]
    A --> GA["grade_answer<br/>grounded in the passages? (informational)"]
    GA --> E([END])
```

The rewrite loop is **hard-capped at 2 retries** — no unbounded loops. Every node appends to a `reasoning_steps` log that the API and UI surface verbatim.

**What makes it agentic** — the system inspects its own intermediate results and changes strategy. A real trace (local llama3.1:8b):

```
Q: "What did Tim Cook's company report for revenue?"
1. Routed: detected company=none (searching all filings), retrieval needed
2. Retrieved 5 candidate passages
3. Doc grade: no — The passages provided do not mention Tim Cook's company...
4. Rewrote query (attempt 1): What are the net sales reported by Apple in its
   Management's Discussion and Analysis section of the 10-K filing?
5. Retrieved 5 candidate passages
6. Doc grade: yes — The passages provide detailed financial information...
7. Generated answer            → "$416,161 million in 2025. [AAPL, 48]"
8. Answer grounded: yes
```

A vanilla RAG pipeline returns "I don't know" at step 3. The agent notices *why* it failed (vague query), repairs the query, and lands a cited, grounded answer — with the whole decision path visible.

Unanswerable questions ("What is Apple's CEO's home address?") exhaust the retry cap and return exactly: *"This information is not available in the filings."* — refusal, not hallucination.

**LLM providers:** every node gets its model from `agent/llm_provider.py` — `LLM_PROVIDER=ollama` (local llama3.1:8b, free) or `LLM_PROVIDER=groq` (llama-3.3-70b-versatile, fast). Query embeddings always stay local (the index was built with nomic-embed-text).

## Evaluation harness

Two complementary layers, both scored by **one fixed judge** (Groq llama-3.3-70b-versatile, temperature 0 — never changed mid-project, so scores stay comparable). RAGAS answer-relevancy embeddings are local MiniLM, so no embeddings API key is needed.

- **`evals/run_eval.py`** — RAGAS (faithfulness, answer relevancy, context precision) over the golden set in `evals/golden/golden.jsonl`; writes timestamped JSON + CSV to `evals/results/`.
- **`evals/test_quality.py`** — DeepEval regression gate on the `smoke: true` subset; runs in CI on every PR (`.github/workflows/eval.yml`). Thresholds (faithfulness ≥ 0.80, answer relevancy ≥ 0.75, hallucination ≤ 0.15) are deliberately below expected performance: the gate catches *regressions*, not perfection.

| Metric (RAGAS, finance golden set) | Average |
|---|---|
| Faithfulness | _finance set not re-scored this round — see the medical results below_ |
| Answer relevancy | _—_ |
| Context precision | _—_ |

## Medical corpus — MIRAGE benchmark & MedCPT ablation

The medical corpus (491,497 snippets, [composition above](#two-corpora-one-engine-corpus-env)) is evaluated end-to-end through the same agent. Two indexes hold the **identical** snippet set and metadata; only the embedding model differs, so `RETRIEVER=general|medcpt` isolates the embedder for a clean ablation.

| Index | Embedder | Vectors |
|---|---|---|
| `chroma_med/` | `nomic-embed-text-v1.5` (general) | 491,497 (StatPearls 365,650 · Textbooks 125,847) |
| `chroma_med_medcpt/` | `ncbi/MedCPT-Article-Encoder` (domain) | 491,497 (same snippets + metadata) |

### MIRAGE exam accuracy (the headline)

[MIRAGE](https://github.com/Teddy-XiongGZ/MIRAGE) MCQ accuracy on the three tasks this corpus actually covers — **MMLU-Med, MedQA-US, MedMCQA** — run through the agent in MCQ mode (`run_agent(..., options=..., choice_only=True)`), 30 questions/task, agent = Groq `llama-3.1-8b-instant`, identical settings across the two retrievers:

| Task | general (nomic) | MedCPT | Δ (MedCPT − general) |
|---|---|---|---|
| MMLU-Med | **70.00** | 63.33 | −6.67 |
| MedQA-US | **60.00** | 56.67 | −3.33 |
| MedMCQA | **56.67** | 53.33 | −3.34 |
| **Exam mean** | **62.22** | 57.78 | **−4.44** |

Published RAG baselines for orientation (from the MedRAG paper — **full MedCorp incl. PubMed + a different/larger generator, so NOT a head-to-head**, treat as a yardstick only): GPT-4 ≈ 79.97, GPT-3.5 ≈ 71.56, Llama2-70B ≈ 53.38. A 7-8B agent on a textbooks+StatPearls subset landing at **62.2** exam-mean sits sensibly between the Llama2-70B and GPT-3.5 reference points.

### The MedCPT ablation finding (honest, and not what I expected)

The domain-specific MedCPT encoder **lost** to the general-purpose nomic encoder by **4.44 exam-mean points** — even though MedCPT was built for biomedical retrieval. A retrieval-only probe (`evals/retrieval_ablation.py`, no LLM, 34 questions) explains why:

| Signal (top-5, reranked) | general | MedCPT | Δ |
|---|---|---|---|
| StatPearls hit-rate | 0.665 | **0.771** | +0.106 |
| ms-marco rerank score (mean) | **0.817** | −1.023 | −1.840 |

MedCPT surfaces **more** StatPearls clinical content (+0.106 hit-rate), but the general-domain cross-encoder reranker — and the downstream MCQ accuracy — both strongly prefer nomic's passages. The lesson: a domain encoder upstream doesn't help if the reranker and generator downstream are general-domain; the pipeline rewards *alignment* across stages more than per-stage domain specialization. A genuinely domain-matched stack would pair MedCPT with a biomedical reranker — out of scope here, but the measured negative result is kept honestly.

**Engineering note (6 GB GTX 1660).** Building both 491k-vector indexes on a 6 GB consumer GPU drove the embedder choices. Observed operating points with length-bucketed dynamic batching:

| Embedder | emb/sec | Peak VRAM |
|---|---|---|
| `nomic-embed-text-v1.5` | ~28–37 | 3.7–4.3 GB |
| `MedCPT-Article-Encoder` | ~46–70 | 1.9–2.5 GB |

MedCPT is the lighter/faster encoder (BERT CLS pooling, ~2× throughput at ~half the VRAM) — so the ablation's accuracy loss is *despite* a cheaper encoder, not because of a starved one. Pushing either toward the 6 GB ceiling was **slower**, not faster (the Windows driver silently pages over-allocations to system RAM); both settled at a lower-VRAM, higher-throughput point. Full build notes in `MORNING_REPORT.md`.

### Research tasks (PubMedQA, BioASQ) — reported separately, expected low

These two MIRAGE tasks need a **PubMed** corpus, which is **not ingested** here. They are **never folded into the exam headline** — shown only to be honest about the corpus boundary (general retriever, 30 q/task):

| Task | Accuracy | Why |
|---|---|---|
| BioASQ-Y/N | 63.33 | yes/no format is partly guessable even without the source corpus |
| PubMedQA | 46.67 | pure PubMed abstracts — the corpus genuinely lacks the evidence |

PubMedQA dropping to ~47% (vs the 62.2 exam mean) is the expected signature of a missing corpus, not a pipeline regression.

### RAGAS on the medical golden set (open-ended answers)

RAGAS over the 14-question medical golden set (`evals/golden/golden_medical.jsonl`), fixed judge = Groq `llama-3.3-70b-versatile` @ temperature 0, agent answers from `llama-3.1-8b-instant`:

| Metric (RAGAS, medical golden) | Average | n scored |
|---|---|---|
| Faithfulness | 0.793 | 6 |
| Answer relevancy | 0.924 | 5 |
| Context precision | 0.775 | 6 |

> **Quota caveat (honest, not a bug).** The 70B judge is token-hungry (~16k tokens/row across the three metrics), and the **Groq free tier caps it at 100k tokens/day** — which is reached after ~6 rows. Rows 7–14 are checkpointed `null` and are **resumable after the daily reset** (the harness skips already-scored rows; the agent answers are fully cached, so no answers re-run). `answer_relevancy` is `null` on 2 rows because RAGAS requests `n=3` generations while Groq caps `n=1`. **DeepEval (`evals/test_quality.py`) uses the same 70B judge and is therefore blocked on the same daily limit** — it runs to completion once the judge quota resets. Resume both with the commands in `MORNING_REPORT.md`.

Everything above is fully **resumable and quota-aware**: every Groq-dependent eval checkpoints per question, distinguishes a transient per-minute rate-limit (back off and retry) from a daily token-limit (checkpoint and exit cleanly), and prints the exact relaunch command on exit.

## Serving it

- **API** — `uvicorn app.main:app --port 8000` → `POST /ask {"question": ...}` returns the answer **plus the full reasoning path**, citations, groundedness, latency, and estimated cost (token usage × configurable Groq rates; `null` on local Ollama, which has no metered cost).
- **Web UI** — `frontend/` is a single-page Next.js 14 app: ask a question, see the answer, the agent's numbered reasoning steps (the hero of the page), citation chips, and latency/cost labels. `app/gradio_app.py` is a one-file fallback UI.
- **Deploy** — Dockerfile targets a Hugging Face Docker Space (embedding model baked in, Groq for generation); frontend goes to Vercel with one env var. Exact steps: [deploy/DEPLOY.md](deploy/DEPLOY.md).

## Observability (Langfuse)

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` in `.env` and every run produces a full trace (each node's LLM calls, latencies, token counts) plus a `trace_url` in the API response. Without keys the agent runs untraced — one warning, no crash.

**Cost tracking note:** Langfuse only computes dollar cost for models whose pricing it knows. For the Groq judge/agent model, add a custom model entry in *Langfuse → Settings → Models* (match `llama-3.3-70b-versatile`, input $0.59/1M, output $0.79/1M as of mid-2026) — otherwise traces show token counts but no cost.

<!-- TODO: screenshot of a Langfuse trace here -->
<!-- ![Langfuse trace](docs/langfuse-trace.png) -->

## Benchmark

Measured on a hand-built test set of graded questions (easy, hard-numeric, cross-company, and unanswerable) against the *same* questions for every configuration.

| Configuration | Hit-rate | Company precision | Avg latency |
|---|---|---|---|
| Baseline (vector top-k) | 0.93 | 0.45 | ~45 ms |
| + Cross-encoder reranking | 1.00 | 0.63 | ~112 ms |
| + Company-aware routing | 1.00 | **0.97** | ~121 ms |

**Reading the table:** company precision is the headline — naive vector search pulled the wrong company's chunks 55% of the time, because every 10-K's risk and finance language reads almost identically. Reranking removes boilerplate; routing filters retrieval to the named company, which is why precision approaches (but doesn't hit) 1.0. Latency rises modestly — the honest cost of the accuracy gain, and small because reranking runs on the GPU.

## How it works

**Indexing.** Each filing's primary HTML document is cleaned (stripped of markup), split into 1,000-character chunks with 200-character overlap, and tagged with `company` metadata. Chunks are embedded with a local `nomic-embed-text` model and stored in a persistent Chroma vector database.

**Company router.** When a question names a company ("What are *Apple's* risk factors?"), retrieval is filtered to that company's chunks before searching — making wrong-company results structurally impossible. Questions naming multiple companies retrieve from each; ambiguous questions fall back to searching everything.

**Cross-encoder reranking.** A wide candidate pool (top 20) is retrieved cheaply, then a cross-encoder re-reads each *(question, chunk)* pair jointly and keeps only the best 5. This is the single highest-impact upgrade — it discards chunks that merely *mention* a keyword in favor of chunks that actually *answer* the question.

**Grounded generation + honesty guard.** Retrieved chunks are passed to a local `llama3.2` model with a prompt that answers *only* from the provided context and cites the company per claim. If no chunk is close enough to the question (distance threshold), the system refuses rather than hallucinating — important for finance.

## Stack

- **Orchestration:** LangChain (pinned to 0.3.x — see `DECISIONS.md`)
- **Embeddings:** `nomic-embed-text` via Ollama (local)
- **Vector store:** Chroma (persistent, on disk)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers, GPU)
- **LLM:** `llama3.2` via Ollama (local)
- **Data:** SEC EDGAR 10-K filings via `sec-edgar-downloader`
- **Tooling:** `uv` for environment + lockfile

Everything runs locally and free — no API keys, no rate limits.

## Setup

**Prerequisites:**
- [uv](https://docs.astral.sh/uv/) for Python + dependency management
- [Ollama](https://ollama.com/) running, with the two models pulled:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.2
  ```
- (Optional) An NVIDIA GPU for the reranker. CPU works but is slower. For GPU, torch is pinned to a CUDA build in `pyproject.toml`.

**Install:**
```bash
git clone https://github.com/gouravshokeen/rag-finance.git
cd rag-finance
uv sync
```

**Download the filings** (SEC requires a real name + email in the user-agent):
```bash
uv run get_filings.py
```

**Run the notebook:**
```bash
uv run jupyter lab
```
Open `rag.ipynb` and run the cells top to bottom. The index builds once (a few minutes on GPU); later runs load it from disk.

**Run the agent** (after the index exists):
```bash
cp .env.example .env          # set LLM_PROVIDER (+ GROQ_API_KEY if groq)
uv run python -m agent.run "What was Apple's total net sales in fiscal 2023?"
```

**Run the evals:**
```bash
uv run python evals/run_eval.py        # RAGAS, full golden set
uv run pytest evals/test_quality.py    # DeepEval smoke gate
```

**Run the API + frontend:**
```bash
uv run uvicorn app.main:app --port 8000
cd frontend && cp .env.local.example .env.local && npm install && npm run dev
# open http://localhost:3000
```

## Evaluation & limitations

These numbers are honest about their scope:

- The test set is **small and hand-built** (~23 graded questions across three filings). The results show the approach works *on these filings*, not a generalized benchmark.
- **Company precision of 0.97 is high partly by construction** — metadata filtering guarantees correct-company retrieval for questions that name a company. The genuine retrieval work shows up on questions that *don't* name a company, where the reranker still surfaces the right content.
- **Hybrid (vector + BM25) search was tried and dropped.** On these filings, BM25 introduced boilerplate noise that *lowered* precision, so the pipeline uses vector retrieval + reranking instead. The reasoning is documented in `DECISIONS.md` — a measured negative result, kept honestly.

And for the agentic layer specifically:

- **LLM-judge variance is real.** RAGAS/DeepEval scores come from a judge LLM; the judge is fixed (one model, temperature 0) to keep runs comparable, but absolute numbers still carry judge bias — treat them as a regression signal, not ground truth.
- **The golden set is ~12–15 questions.** Big enough to catch breakage, far too small for statistical claims.
- **Cost is estimated**, not billed: token usage × published Groq per-token rates (configurable). Local Ollama runs report no cost because there is none to meter.
- **The self-correction loop helps vague queries, not missing data.** If the filings genuinely lack the answer, the agent burns its 2 retries and refuses — that's the designed behavior, but it means retries aren't free for hopeless questions.

And for the medical corpus / MIRAGE results specifically:

- **MIRAGE here is a 30-question/task sample, not the full benchmark.** 30×3 = 90 exam questions per retriever is enough to rank the two embedders consistently (general beats MedCPT on all three tasks), but task-level accuracies carry a few points of sampling noise — read the *direction* of the ablation, not the third decimal.
- **The baseline comparison is orientation, not a head-to-head.** Published baselines used the full MedCorp corpus (incl. PubMed) and larger generators; this runs a 7-8B agent over a textbooks+StatPearls subset. The numbers are positioned as a yardstick and labelled as such everywhere they appear.
- **The agent model was chosen for quota, not peak accuracy.** MIRAGE answers come from `llama-3.1-8b-instant` (≈5× the free-tier daily token budget of 70B), so the exam-mean reflects an 8B generator — a 70B agent would likely score higher. The point of the run is the *embedder ablation* (general vs MedCPT, everything else held fixed), which the model choice doesn't bias.
- **RAGAS medical scores are over 6 rows, not 14** — the Groq free-tier 70B daily token cap stopped scoring partway. They're real (fixed 70B judge, temperature 0) but thin; resumable after reset. DeepEval is blocked on the same cap.
- **MedCPT lost the ablation, and that's reported as-is.** It would have been easy to bury a negative result for the fancier domain encoder; instead it's the centerpiece, with a retrieval-level explanation (stage misalignment) rather than a hand-wave.

## Repo layout

```
rag-finance/
├── get_filings.py        # one-time: download 10-Ks from SEC EDGAR
├── rag.ipynb             # phase 1: the retrieval pipeline, cell by cell
├── agent/                # phase 2: LangGraph agent
│   ├── graph.py          #   the state machine + run_agent() entry point
│   ├── retriever.py      #   SmartRetriever (from rag.ipynb) over the persisted index
│   ├── llm_provider.py   #   LLM_PROVIDER switch: ollama | groq
│   ├── tracing.py        #   Langfuse (version-aware, graceful without keys)
│   └── run.py            #   CLI: python -m agent.run "question"
├── evals/                # phase 3: evaluation
│   ├── golden/golden.jsonl   # golden Q&A set (edit me!)
│   ├── judge.py          #   THE fixed judge (RAGAS + DeepEval wrappers)
│   ├── run_eval.py       #   RAGAS runner → evals/results/
│   └── test_quality.py   #   DeepEval smoke gate (runs in CI)
├── app/                  # serving
│   ├── main.py           #   FastAPI: POST /ask, GET /health
│   └── gradio_app.py     #   one-file fallback UI
├── frontend/             # Next.js 14 single-page UI
├── deploy/DEPLOY.md      # HF Spaces + Vercel, step by step
├── Dockerfile            # backend image (HF Docker Space compatible)
├── pyproject.toml        # dependencies (LangChain pinned 0.3.x, CUDA torch)
├── uv.lock               # reproducible environment
├── README.md
└── DECISIONS.md          # why each architectural choice was made
```

---

Built by Gourav Shokeen.
