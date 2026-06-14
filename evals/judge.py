"""Single source of truth for the eval judge LLM — used by RAGAS and DeepEval.

THE JUDGE IS FIXED: Groq llama-3.3-70b-versatile at temperature 0. Scores from
different judges are NOT comparable; never change it mid-project.

JUDGE_PROVIDER=ollama exists ONLY as a keyless local smoke path (e.g. on a dev
box without a GROQ_API_KEY). Numbers produced that way are for plumbing checks,
not for the README results table or for comparing runs.

Embeddings for RAGAS answer-relevancy are LOCAL (MiniLM via sentence-transformers)
so no embeddings API key is needed — Groq has no embeddings endpoint.
"""

import os

from dotenv import load_dotenv

from agent.llm_provider import get_llm

load_dotenv()

JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "groq")
# The judge is PINNED to 70B regardless of GROQ_MODEL (which the agent may set to
# 8b to stretch quota). JUDGE_MODEL overrides only if you deliberately want a
# different fixed judge. Ollama judge ignores this (uses OLLAMA_MODEL).
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Gemini judge: an alternative fixed judge used to complete rows the Groq 70B
# judge couldn't (its daily TPD ran out mid-set). Resolved once, lazily, trying
# the newest free-tier flash model first and falling back. Build the LangChain
# object DIRECTLY here (not via agent/llm_provider.py) to keep the agent's own
# LLM wiring untouched.
_GEMINI_FALLBACKS = ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash")
_GEMINI_MODEL = None  # cached resolved model id


def _gemini_key():
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def resolve_gemini_model():
    """Pick the first Gemini flash model whose key+model actually answer.

    Returns the model id (and caches it). Raises RuntimeError with the exact
    underlying error if no model loads, so the caller can STOP rather than spin.
    """
    global _GEMINI_MODEL
    if _GEMINI_MODEL:
        return _GEMINI_MODEL
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = _gemini_key()
    if not key:
        raise RuntimeError("No GOOGLE_API_KEY / GEMINI_API_KEY in environment")
    override = os.getenv("JUDGE_GEMINI_MODEL")
    candidates = (override,) + _GEMINI_FALLBACKS if override else _GEMINI_FALLBACKS
    last = None
    for m in candidates:
        try:
            llm = ChatGoogleGenerativeAI(model=m, temperature=0, google_api_key=key)
            llm.invoke("Reply with the single word OK")
            _GEMINI_MODEL = m
            print(f"[judge] Gemini model loaded: {m}")
            return m
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[judge] Gemini model {m} unavailable: {str(exc)[:120]}")
    raise RuntimeError(f"Gemini key failed on all models; last error: {last}")


def get_judge_llm():
    """The one judge chat model (LangChain object, temperature 0)."""
    if JUDGE_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=resolve_gemini_model(), temperature=0,
            google_api_key=_gemini_key(),
        )
    model = JUDGE_MODEL if JUDGE_PROVIDER == "groq" else None
    return get_llm(temperature=0, provider=JUDGE_PROVIDER, model=model)


def get_ragas_llm():
    """Judge wrapped for ragas (installed 0.4.x keeps LangchainLLMWrapper)."""
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(get_judge_llm())


def get_ragas_embeddings():
    """Local MiniLM embeddings wrapped for ragas (no API key required)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=_EMBED_MODEL))


def get_deepeval_model():
    """DeepEval judge wrapper around the same fixed Groq model.

    DeepEval defaults to OpenAI; passing model=this to every metric forces our
    judge instead, so no OPENAI_API_KEY is needed. deepeval 4.x calls
    a_generate_with_schema(prompt, schema=...), whose base implementation tries
    a_generate(prompt, schema=...) first — so we accept the schema kwarg and use
    LangChain structured output, returning a schema instance directly.
    """
    from deepeval.models.base_model import DeepEvalBaseLLM

    class GroqJudge(DeepEvalBaseLLM):
        def __init__(self):
            super().__init__(model="medical-rag-judge")

        def load_model(self):
            return get_judge_llm()

        def generate(self, prompt: str, schema=None):
            if schema is not None:
                try:
                    return self.model.with_structured_output(schema).invoke(prompt)
                except Exception:
                    pass  # fall through: deepeval parses JSON out of plain text
            return self.model.invoke(prompt).content

        async def a_generate(self, prompt: str, schema=None):
            if schema is not None:
                try:
                    return await self.model.with_structured_output(schema).ainvoke(
                        prompt
                    )
                except Exception:
                    pass
            return (await self.model.ainvoke(prompt)).content

        def get_model_name(self) -> str:
            return f"{JUDGE_PROVIDER}-judge (fixed, temperature 0)"

    return GroqJudge()
