"""Langfuse tracing helper — graceful: the agent runs fully even without keys.

Installed langfuse is 4.x, which keeps the v3 import path (langfuse.langchain) and
reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from env. The v2
fallback import takes keys as constructor args instead.

NOTE on cost tracking: Langfuse only computes per-token COST for models whose
pricing it knows. For the Groq model (llama-3.3-70b-versatile) you must define a
custom model + pricing in the Langfuse project settings (Settings -> Models),
otherwise traces show token counts but no dollar cost.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def get_langfuse():
    """Return (callback_handler, trace_url, client) or (None, None, None).

    trace_url is pre-computed from a client-generated trace id so run_agent can
    report it without waiting for ingestion. client is returned so the caller can
    flush() before process exit (the v3+/v4 SDK batches spans via OTEL).
    """
    missing = [k for k in _REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        logger.warning(
            "Langfuse tracing disabled (missing env: %s) — agent runs untraced.",
            ", ".join(missing),
        )
        return None, None, None

    try:
        try:
            # v3+ (incl. installed 4.x): handler reads keys from env
            from langfuse.langchain import CallbackHandler

            from langfuse import Langfuse, get_client

            trace_id = Langfuse.create_trace_id()
            handler = CallbackHandler(trace_context={"trace_id": trace_id})
            client = get_client()
            trace_url = client.get_trace_url(trace_id=trace_id)
            return handler, trace_url, client
        except ImportError:
            # v2: keys passed as constructor args
            from langfuse.callback import CallbackHandler

            handler = CallbackHandler(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
            return handler, None, None
    except Exception as exc:  # tracing must never take the agent down
        logger.warning("Langfuse init failed (%s) — agent runs untraced.", exc)
        return None, None, None
