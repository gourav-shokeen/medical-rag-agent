---
title: rag-finance API
emoji: 📊
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# rag-finance agent API

FastAPI backend for the agentic SEC 10-K RAG. See the main repo for docs.

Endpoints: `POST /ask {"question": "..."}` · `GET /health`

Required Space secrets: `GROQ_API_KEY` (and optionally `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`). Required Space variables:
`LLM_PROVIDER=groq`, `FRONTEND_ORIGIN=<your Vercel URL>`.
