# Medical Reference Agent

A self-correcting, agentic RAG system over a 491k-snippet medical corpus (StatPearls + 18 USMLE textbooks). Ask a clinical question and the agent routes it, retrieves passages, grades whether they actually answer the question, generates a citation-grounded answer, and verifies that answer against its sources — repairing the query and retrying when retrieval comes up short, and refusing when the corpus can't support a claim.

```
route → retrieve → grade docs → generate → check grounding → (repair & retry | grounded answer)
```

A vanilla pipeline stops at step 3 with a weak answer. The agent notices *why* retrieval failed, repairs the query, and lands a grounded one — with the whole decision path visible.

## Results

### MIRAGE benchmark — embedding ablation (the headline)

Accuracy on the three [MIRAGE](https://github.com/Teddy-XiongGZ/MIRAGE) exam tasks this corpus covers, run through the agent in MCQ mode (30 q/task, agent = Groq `llama-3.1-8b-instant`, everything held fixed except the embedder):

| Task | general (nomic) | MedCPT (domain) | Δ |
|---|---|---|---|
| MMLU-Med | **70.00** | 63.33 | −6.67 |
| MedQA-US | **60.00** | 56.67 | −3.33 |
| MedMCQA | **56.67** | 53.33 | −3.34 |
| **Exam mean** | **62.22** | 57.78 | **−4.44** |

### The finding I didn't expect

The **domain-specific** medical encoder (MedCPT) **lost** to the general one. A retrieval-only probe shows why:

| Signal (top-5, reranked) | general | MedCPT | Δ |
|---|---|---|---|
| StatPearls hit-rate | 0.665 | **0.771** | +0.106 |
| reranker score (mean) | **0.817** | −1.023 | −1.840 |

MedCPT surfaces **more** clinical content (+0.106) — but the general-domain cross-encoder reranker downstream scores its passages far lower. **The pipeline rewards alignment across stages more than per-stage domain specialization.** A real, non-obvious systems result, kept honestly rather than buried.

*For orientation only (not head-to-head — they used a larger corpus + generator): published baselines are GPT-3.5 ≈ 71.6, Llama2-70B ≈ 53.4. A 7-8B agent on a textbooks+StatPearls subset landing at 62.2 sits sensibly between them.*

### Answer quality (RAGAS)

| Metric | Score | n |
|---|---|---|
| Faithfulness | 0.793 | 6 |
| Answer relevancy | 0.924 | 5 |
| Context precision | 0.775 | 6 |

Scored by a fixed judge (Groq 70B, temp 0). Partial — the free-tier daily token cap stopped scoring at ~6 rows; the harness is resumable. Treat as a regression signal, not ground truth.

## The corpus

| Source | Snippets | What it is |
|---|---|---|
| StatPearls | 365,650 | Point-of-care clinical decision support |
| Textbooks | 125,847 | 18 USMLE medical textbooks |
| **Total** | **491,497** | embedded with `nomic-embed-text-v1.5` |

PubMed is **not** ingested (its 23.9M snippets are infeasible on a 6 GB GPU), so PubMed-specific MIRAGE tasks are reported separately and expected low — an honest corpus boundary, not a regression.

## Run it

```bash
git clone https://github.com/gouravshokeen/medical-rag-agent.git
cd medical-rag-agent && uv sync

# build the index once (resumable, ~hours on a consumer GPU)
python ingest/build_medical_index.py

# ask a question (set LLM_PROVIDER=groq + GROQ_API_KEY in .env for fast answers)
cp .env.example .env
uv run python -m agent.run "What are common adverse effects of ACE inhibitors?"

# API + web UI
uv run uvicorn app.main:app --port 8000
cd frontend && npm install && npm run dev   # → localhost:3000
```

**LLM providers:** `LLM_PROVIDER=groq` (fast, ~7s/answer) or `ollama` (local, free, slow). Embeddings always run locally.

## Stack

LangGraph · LangChain 0.3.x · Chroma · `nomic-embed-text` + `MedCPT` (ablation) · `ms-marco-MiniLM` reranker · Groq / Ollama · RAGAS + DeepEval · Langfuse tracing · FastAPI + Next.js · `uv`

## Honest limitations

- **MIRAGE is a 30 q/task sample**, not the full benchmark — read the *direction* of the ablation, not the third decimal.
- **The agent is 7-8B**, chosen for free-tier quota; a larger model would likely score higher. The ablation holds everything fixed except the embedder, so model size doesn't bias the finding.
- **RAGAS scores are over ~6 rows** — real but thin.
- **Self-correction fixes vague queries, not missing data** — if the corpus lacks the answer, the agent exhausts its retries and refuses.

---

Built by Gourav Shokeen.