"use client";

import { useState } from "react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function ask(e) {
    e.preventDefault();
    if (!question.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) {
        const body = await res.text();
        throw new Error(`API ${res.status}: ${body.slice(0, 300)}`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>Medical Reference Agent</h1>
      <p className="subtitle">
        Self-correcting RAG over published medical textbooks &amp; clinical references.
      </p>

      <div className="disclaimer">
        Educational summaries from published medical references — not medical advice.
        For personal symptoms, consult a clinician (or seek emergency care if urgent).
      </div>

      <form onSubmit={ask}>
        <input
          type="text"
          value={question}
          placeholder="e.g. What is the first-line management of community-acquired pneumonia?"
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? "Thinking…" : "Ask"}
        </button>
      </form>

      {loading && (
        <p className="hint">
          The agent retrieves, grades its own evidence, and retries with a
          rewritten query if needed — a local run can take a minute or two.
        </p>
      )}

      {error && <p className="error">{error}</p>}

      {result && (
        <>
          <section>
            <div className="label">Answer</div>
            <div className="answer">{result.answer}</div>
          </section>

          <section>
            <div className="label">Agent reasoning</div>
            <ol className="steps">
              {result.reasoning_steps.map((step, i) => (
                <li key={i}>{step}</li>
              ))}
            </ol>
          </section>

          {result.citations.length > 0 && (
            <section>
              <div className="label">Citations</div>
              <ul className="citations">
                {result.citations.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </section>
          )}

          <section className="meta">
            <span>{(result.latency_ms / 1000).toFixed(1)}s</span>
            <span>
              {result.cost_usd != null
                ? `$${result.cost_usd.toFixed(6)}`
                : "cost n/a (local model)"}
            </span>
            <span>retries: {result.retries}</span>
            <span>grounded: {String(result.grounded)}</span>
          </section>
        </>
      )}
    </main>
  );
}
