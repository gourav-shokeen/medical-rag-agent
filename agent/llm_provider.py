"""Single source of truth for the agent's chat LLM.

Every graph node gets its model from get_llm(); the provider is selected by the
LLM_PROVIDER env var ("ollama" | "groq").
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Current Groq Llama-3.x 70B model. If Groq retires it, swap here and note in README.
GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

VALID_PROVIDERS = ("ollama", "groq")


def get_llm(temperature: float = 0, provider: str | None = None):
    """Return the chat model selected by LLM_PROVIDER (or the explicit override).

    The `provider` argument exists so callers with a FIXED provider requirement
    (e.g. the eval judge, which must always be Groq) don't silently follow the
    agent's LLM_PROVIDER setting.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "")).strip().lower()
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER must be one of {VALID_PROVIDERS}, got {provider!r}. "
            "Set it in your environment or .env file (see .env.example)."
        )

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is not set. "
                "Add it to your .env file (see .env.example)."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(model=GROQ_MODEL, temperature=temperature)

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        temperature=temperature,
    )
