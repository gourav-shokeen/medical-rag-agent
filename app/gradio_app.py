"""Single-file Gradio fallback UI: python app/gradio_app.py

Talks to the same FastAPI backend (API_BASE_URL env, default localhost:8000).
Install the optional extra first: uv sync --extra ui

SHIP DECISION (1-day budget): ship THIS, not the Next.js app. It is one file,
has no build step, no Node toolchain, no CORS setup (same-origin if mounted
next to the API), and HF Spaces can host it natively. The Next.js front end
looks better and is what I'd grow into a product, but on a 1-day budget the
agent demo lives or dies on the backend, and Gradio gets a working public UI
in ~15 minutes.
"""

import json
import os

import gradio as gr
import requests

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def ask(question: str):
    if not question.strip():
        return "", "", "", ""
    try:
        res = requests.post(
            f"{API_BASE}/ask", json={"question": question}, timeout=600
        )
        res.raise_for_status()
        out = res.json()
    except Exception as exc:
        return f"**Error:** {exc}", "", "", ""

    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(out["reasoning_steps"], 1))
    citations = ", ".join(out["citations"]) or "(none)"
    cost = f"${out['cost_usd']:.6f}" if out.get("cost_usd") is not None else "n/a (local model)"
    meta = (
        f"latency: {out['latency_ms'] / 1000:.1f}s · cost: {cost} · "
        f"retries: {out['retries']} · grounded: {out['grounded']}"
    )
    return out["answer"], steps, citations, meta


with gr.Blocks(title="10-K Analyst Agent") as demo:
    gr.Markdown(
        "# 10-K Analyst Agent\n"
        "Self-correcting RAG over SEC 10-K filings — Apple, Microsoft, NVIDIA."
    )
    question = gr.Textbox(
        label="Question",
        placeholder="What was Apple's total net sales in fiscal 2023?",
    )
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Markdown(label="Answer")
    steps = gr.Textbox(label="Agent reasoning (the agentic part)", lines=8)
    citations = gr.Textbox(label="Citations")
    meta = gr.Textbox(label="Run info")
    btn.click(ask, inputs=question, outputs=[answer, steps, citations, meta])
    question.submit(ask, inputs=question, outputs=[answer, steps, citations, meta])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("UI_PORT", "7861")))
