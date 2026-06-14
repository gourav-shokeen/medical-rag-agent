"""FastAPI wrapper around the agent: uvicorn app.main:app --port 8000

POST /ask {question} -> {answer, reasoning_steps, citations, grounded,
                         latency_ms, cost_usd, retries, trace_url?}
GET  /health         -> {"status": "ok"}
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

# Per-token pricing for cost estimation. Defaults are Groq's published rates for
# llama-3.3-70b-versatile ($/1M tokens). Override via env if the price changes.
# cost_usd is None when usage metadata is absent or the model is local Ollama
# (a local model has no metered per-token cost to estimate).
PRICE_IN_PER_M = float(os.getenv("PRICE_INPUT_PER_MTOK", "0.59"))
PRICE_OUT_PER_M = float(os.getenv("PRICE_OUTPUT_PER_MTOK", "0.79"))

app = FastAPI(title="medical-rag agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    reasoning_steps: list[str]
    citations: list[str]
    grounded: bool | None
    retries: int
    latency_ms: float
    cost_usd: float | None
    trace_url: str | None = None


def _estimate_cost(usage: dict) -> float | None:
    if os.getenv("LLM_PROVIDER", "").lower() != "groq":
        return None  # local Ollama: no metered cost
    if not usage or not usage.get("total_tokens"):
        return None  # model returned no usage metadata
    return round(
        usage["input_tokens"] / 1e6 * PRICE_IN_PER_M
        + usage["output_tokens"] / 1e6 * PRICE_OUT_PER_M,
        6,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    from agent.graph import run_agent  # lazy: keeps /health instant on cold start

    try:
        out = run_agent(req.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        answer=out["answer"],
        reasoning_steps=out["reasoning_steps"],
        citations=out["citations"],
        grounded=out["grounded"],
        retries=out["retries"],
        latency_ms=out["latency_ms"],
        cost_usd=_estimate_cost(out.get("usage", {})),
        trace_url=out.get("trace_url"),
    )
