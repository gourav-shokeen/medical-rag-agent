# Morning Report — Medical Corpus Overnight Run

Branch: `feat/medical-corpus`. Everything below is committed; run logs are copied
into `logs/`.

---

## UPDATE (2026-06-14) — full evals run, real numbers in

The Groq quota reset, so the full evals below were actually run (not just smoked).
All numbers are now in `README.md` (section "Medical corpus — MIRAGE benchmark &
MedCPT ablation"). Per-eval captures are in `logs/step3_mirage_ablation.txt`,
`logs/step3b_research_summary.txt`, `logs/step4_ragas_summary.txt`.

**MIRAGE exam ablation (30 q/task, 8b-instant agent, the headline):**

| Task | general (nomic) | MedCPT | Δ |
|---|---|---|---|
| MMLU-Med | 70.00 | 63.33 | −6.67 |
| MedQA-US | 60.00 | 56.67 | −3.33 |
| MedMCQA | 56.67 | 53.33 | −3.34 |
| **Exam mean** | **62.22** | **57.78** | **−4.44** |

The general nomic encoder **beat** the domain-specific MedCPT encoder. Retrieval
probe (`evals/retrieval_ablation.py`, no LLM) shows why: MedCPT surfaces more
StatPearls (hit-rate 0.771 vs 0.665, **+0.106**) but the general ms-marco reranker
scores its passages far lower (−1.023 vs 0.817, **−1.840**). Stage misalignment —
a domain encoder doesn't pay off with a general reranker + generator downstream.

**Research tasks (general, separate — no PubMed corpus):** PubMedQA 46.67, BioASQ
63.33. Never folded into the exam headline.

**RAGAS (medical golden, fixed 70B judge @ temp 0):** faithfulness 0.793 (n=6),
answer_relevancy 0.924 (n=5), context_precision 0.775 (n=6). **PARTIAL** — the 70B
free-tier daily token cap (100k) is reached after ~6 rows (the judge uses ~16k
tokens/row). Rows 7–14 are checkpointed null and resumable.

**Still pending the next 70B daily reset (resume commands):**
```bash
# RAGAS rows 7-14 (skips the 6 done; agent answers are cached, nothing re-runs):
CORPUS=medical LLM_PROVIDER=groq JUDGE_PROVIDER=groq GROQ_MODEL=llama-3.1-8b-instant \
  GOLDEN_PATH=evals/golden/golden_medical.jsonl python evals/run_eval.py

# DeepEval smoke gate (same 70B judge — blocked on the same cap until reset):
CORPUS=medical LLM_PROVIDER=groq JUDGE_PROVIDER=groq GROQ_MODEL=llama-3.1-8b-instant \
  GOLDEN_PATH=evals/golden/golden_medical.jsonl python -m pytest evals/test_quality.py
```
Everything is per-question resumable: transient per-minute rate-limits back off and
retry; the daily token-limit checkpoints and exits cleanly with the relaunch line.

---

## TL;DR

- **StatPearls is in.** `chroma_med/` was fully re-embedded on GPU with
  nomic-embed-text-v1.5 and now holds **491,497 vectors** (StatPearls 365,650 +
  Textbooks 125,847). Clinical retrieval is **fixed** — all three test queries now
  return 5/5 StatPearls clinical snippets instead of tangential textbook noise.
- **MedCPT ablation index** (`chroma_med_medcpt/`) built with the same ~491k
  snippets using ncbi/MedCPT-Article-Encoder, swappable via `RETRIEVER=medcpt`.
- **Harnesses proven**: MIRAGE MCQ (Groq) and RAGAS (Groq judge) both execute on
  the medical corpus; DeepEval collects. Full evals are **pending your Groq key
  quota** (see caveat) — I ran only smokes.

## What completed vs skipped

| Step | Status | Notes |
|------|--------|-------|
| 1. Build full `chroma_med/` (general embedder) | ✅ DONE | 491,497 vectors, GPU nomic ST |
| 2. Verify clinical retrieval fixed | ✅ DONE | 5/5 StatPearls in top-5, all 3 queries |
| 3. Build `chroma_med_medcpt/` (MedCPT) | ✅ DONE | 491,497 vectors; retrieval 5/5,5/5,4/5 StatPearls |
| 4. MIRAGE harness + smoke | ✅ DONE (smoke) | 3/task general = 66.67%; full run pending key |
| 5. RAGAS/DeepEval on medical | ✅ DONE (smoke) | RAGAS ran; DeepEval collects; **Groq daily quota hit** |
| 6. This report + logs + commit | ✅ DONE | — |

Nothing was skipped. The only thing NOT run (by design + quota) is the *full*
thousands-question MIRAGE eval and the full RAGAS/DeepEval scoring — those need
your Groq key with quota and your sign-off on cost.

## Final corpus composition

| Index | Embedder | Vectors | Per source |
|-------|----------|---------|------------|
| `chroma_med/` | nomic-embed-text-v1.5 (general) | 491,497 | statpearls 365,650 · textbook 125,847 |
| `chroma_med_medcpt/` | ncbi/MedCPT-Article-Encoder | 491,497 | statpearls 365,650 · textbook 125,847 |

Both indexes hold the identical snippet set + metadata `{source, title,
snippet_id}`; only the embedding model differs, so `RETRIEVER=general|medcpt`
isolates the embedder for a clean ablation.

## Step 2 — clinical retrieval before/after (the point of the night)

Query (a) "first-line treatment for community-acquired pneumonia in adults",
top-5 by source (full captures in `logs/step2_retrieval_*`):

**BEFORE** (textbooks-only index):
```
[textbook: First_Aid_Step2]
[textbook: Gynecology_Novak]
[textbook: InternalMed_Harrison]
[textbook: Pharmacology_Katzung]
[textbook: Gynecology_Novak]      -> statpearls in top-5: 0
```
**AFTER** (textbooks + StatPearls, GPU nomic):
```
[statpearls: Aspiration Pneumonia -- Treatment / Management]
[statpearls: Community-Acquired Pneumonia -- Treatment / Management]
[statpearls: Community-Acquired Pneumonia -- Treatment / Management]
[statpearls: Cough: Evaluation and Management -- ... -- Pneumonia]
[statpearls: Community-Acquired Pneumonia -- Treatment / Management]   -> 5/5
```
(b) diabetic ketoacidosis → 5/5 StatPearls; (c) first-line antihypertensive in
diabetes → 5/5 StatPearls (incl. "Lisinopril -- Diabetes and hypertension").
**Verdict: StatPearls fixed clinical retrieval.**

The **MedCPT** index retrieves comparably well on the same 3 queries (pneumonia
5/5, DKA 5/5, antihypertensive 4/5 StatPearls) — both embedders surface the right
clinical content, so the ablation is a fair like-for-like (full accuracy numbers
come from the MIRAGE run you'll trigger). Capture: `logs/step3_medcpt_retrieval.txt`.

### One real bug found & fixed during MedCPT verification
`get_vectorstore(None)` resolved the index *directory* from the `RETRIEVER` env
(→ `chroma_med_medcpt`) but the *query embedder* fell back to `"general"` (nomic),
so the verification script queried the MedCPT index with nomic vectors → garbage
("manage DKA" returned cholangiocarcinoma lab snippets). Fixed: `get_vectorstore`
now resolves the retriever once and uses it for BOTH dir and embedder. NOTE: the
**agent path was never affected** — `agent/retriever.get_retriever()` always passed
`RETRIEVER` explicitly to both. Only the standalone check helper had the mismatch.

## Embedder operating points actually observed (6 GB GTX 1660)

| Embedder | Budget | emb/sec | Peak VRAM | Notes |
|----------|--------|---------|-----------|-------|
| nomic-embed-text-v1.5 | 50M | ~28–37 | 3.7–4.3 GB | 100M overshot to 5.6 GB → hard OOM on the real (longer) clinical corpus; 50M holds the target |
| MedCPT-Article-Encoder | 50M | ~46–70 | 1.9–2.5 GB | lighter (BERT CLS); 100M was *slower* (61/s) — 50M is its optimum |

Key engineering notes:
- **Length-bucketed dynamic batching** (`agent/embeddings.encode_bucketed`): sort
  each buffer by length, size sub-batches from their longest member so
  `batch×maxlen²` stays ~constant. Without it, a fixed batch either starves the GPU
  on short snippets or spills VRAM on long ones. On Windows the driver silently
  pages over-allocations to system RAM (no clean OOM) → ~5× slowdown, which is what
  made the first naive run crawl at 4–8 emb/sec.
- **Filling VRAM is counter-productive here**: pushing either embedder toward 5–6 GB
  was *slower*, not faster (driver paging on the 6 GB ceiling). Both settled at the
  lower-VRAM, higher-throughput operating point.
- **expandable_segments** is unsupported on Windows (no-op) — the lower budget is
  the real fix.

## Errors hit overnight & how they were resolved

1. **CUDA OOM crash** at budget 100M (real clinical snippets are longer/denser than
   the benchmark sample, peaked 5.6 GB) → lowered to 50M (peak ~4 GB). Resolved.
2. **ChromaDB add cap** — a single `collection.add()` is capped at 5461 rows; the
   8192 buffer crashed it → adds are chunked to ≤5000. Resolved.
3. **Windows cp1252 console** crashed on `≤` (U+2264) in medical text → every
   corpus-printing script sets `sys.stdout.reconfigure(encoding="utf-8")`. Resolved.
4. **Repeated background-process kills** (~every 30–130 min, tied to session
   activity) → buffer reduced to 2048 so checkpoints are frequent; the build is
   resumable by id (marker + skip), so each relaunch lost ~0 work.
5. **MIRAGE used Ollama** (from `.env` `LLM_PROVIDER=ollama`, Ollama not serving) →
   `run_mirage.py` now `load_dotenv()` + forces `LLM_PROVIDER=groq`. Resolved.
6. **`KeyError: 'company'`** in run_eval on medical golden → `company` made optional.
   Resolved.
7. **Groq free-tier daily token limit (100k/day) exhausted** by the MIRAGE + RAGAS
   smokes → a few RAGAS jobs 429'd (it still produced averages). NOT a code bug; see
   caveat. This blocks the *full* evals until reset/upgrade.

## Smoke results (NOT statistically meaningful — harness proofs only)

- **MIRAGE** (general retriever, Groq, 3 questions/task): MMLU 66.67, MedQA 66.67,
  MedMCQA 66.67 → mean **66.67%** (n=9). Reference baselines (different corpus +
  generator, NOT a head-to-head): GPT-4 79.97, GPT-3.5 71.56, Llama2-70B 53.38.
- **RAGAS** (3 medical golden rows, Groq judge): faithfulness **1.0**,
  answer_relevancy **0.89**, context_precision **0.88**.

## Exactly what to run next (needs YOUR Groq key with quota)

The repo `.env` already has a `GROQ_API_KEY`, but its **free-tier daily quota was
used up** tonight. Wait for the daily reset (or upgrade to Groq Dev tier), then:

```bash
# Full MIRAGE on the GENERAL embedder (all exam questions: MMLU+MedQA+MedMCQA)
CORPUS=medical python -m evals.mirage.run_mirage --full --retriever general

# Full MIRAGE on the MedCPT embedder (the ablation — same questions, swap encoder)
CORPUS=medical RETRIEVER=medcpt python -m evals.mirage.run_mirage --full --retriever medcpt

# Compare the two result JSONs in evals/mirage/results/ -> this is the embedder ablation.

# Full open-ended evals on chroma_med (after you add real golden questions, below)
CORPUS=medical LLM_PROVIDER=groq JUDGE_PROVIDER=groq \
  GOLDEN_PATH=evals/golden/golden_medical.jsonl python evals/run_eval.py
CORPUS=medical LLM_PROVIDER=groq JUDGE_PROVIDER=groq \
  GOLDEN_PATH=evals/golden/golden_medical.jsonl python -m pytest evals/test_quality.py
```
Paste MIRAGE accuracy numbers into the table in this report / the README ablation
section; paste RAGAS/DeepEval averages into the eval table in README.

## TODO — only you can do these

1. **Groq quota**: wait for daily reset or upgrade to Dev tier, then run the full
   MIRAGE (general + medcpt) and full RAGAS/DeepEval (commands above).
2. **Golden questions**: replace the 5 placeholders in
   `evals/golden/golden_medical.jsonl` with 12–15 real medical Q/A (3–4
   unanswerable, mark 6 `smoke:true`).
3. **README numbers**: once the full evals run, paste the MIRAGE ablation table
   (general vs MedCPT) and the RAGAS/DeepEval averages into the README.

## Logs in this commit (`logs/`)

- `step2_retrieval_before_textbooks_only.txt` / `step2_retrieval_after_statpearls.txt`
- `step4_mirage_smoke_general.txt`
- `step5_ragas_smoke_medical.txt`
- `step1_chroma_med_build.txt` (nomic rebuild) / `step3_medcpt_build.txt` (MedCPT build)
