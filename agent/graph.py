"""Agentic RAG over SEC 10-K filings, built with LangGraph.

Flow: route -> retrieve -> grade_documents -> (rewrite_query -> retrieve loop, max 2
rewrites) -> generate -> grade_answer -> END. Retrieval is the existing SmartRetriever
(company filter + vector top-20 + cross-encoder top-5); this module only orchestrates.
"""

import logging
import operator
import re
import time
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from agent.llm_provider import get_llm
from agent.retriever import detect_companies, get_smart_retriever
from agent.tracing import get_langfuse

logger = logging.getLogger(__name__)

REFUSAL = "This information is not available in the filings."

# small models paraphrase the refusal ("...in the passages/documents"); normalize
# near-misses so downstream exact-match logic (citations, grading) stays correct
_REFUSAL_PATTERN = re.compile(
    r"information is not available in the (filings|passages|context|documents|provided)",
    re.IGNORECASE,
)

# Passage text per doc handed to the graders (keeps grading prompts small).
_GRADE_DOC_CHARS = 1200


class AgentState(TypedDict):
    question: str  # current retrieval query; rewrite_query may replace it
    original_question: str  # what the user actually asked; generate answers THIS
    company: Optional[str]
    documents: List[Document]
    generation: str
    citations: List[str]
    retries: int
    # reducer: each node returns only its NEW step(s); LangGraph appends them
    reasoning_steps: Annotated[List[str], operator.add]
    grounded: Optional[bool]
    # verdict of grade_documents, read by the conditional edge
    docs_sufficient: Optional[bool]


class Grade(BaseModel):
    binary_score: Literal["yes", "no"]
    reason: str


def _grade(prompt: str, default: str = "yes") -> Grade:
    """Run a yes/no grading prompt; structured output first, lenient text fallback.

    Defaults to `default` on any failure so a flaky grader can never loop the graph.
    """
    llm = get_llm()
    try:
        return llm.with_structured_output(Grade).invoke(prompt)
    except Exception as exc:
        logger.debug("Structured output failed (%s); falling back to text parse.", exc)
    try:
        text = llm.invoke(
            prompt + "\n\nRespond with ONLY the single word 'yes' or 'no'."
        ).content
        match = re.search(r"\b(yes|no)\b", text.lower())
        score = match.group(1) if match else default
        return Grade(binary_score=score, reason=text.strip()[:200] or "text fallback")
    except Exception as exc:
        return Grade(binary_score=default, reason=f"grader failed ({exc}); defaulted")


def _format_docs(documents: List[Document], max_chars: Optional[int] = None) -> str:
    parts = []
    for d in documents:
        company = d.metadata.get("company", "UNKNOWN")
        source = d.metadata.get("source", "10-K")
        text = d.page_content[:max_chars] if max_chars else d.page_content
        parts.append(f"[{company}, {source}] {text}")
    return "\n\n".join(parts)


# --- nodes -------------------------------------------------------------------


def route(state: AgentState) -> dict:
    companies = detect_companies(state["question"])
    company = ",".join(companies) if companies else None
    label = company if company else "none (searching all filings)"
    return {
        "company": company,
        "reasoning_steps": [f"Routed: detected company={label}, retrieval needed"],
    }


def retrieve(state: AgentState) -> dict:
    # SmartRetriever does its own company detection + filtering from the question
    docs = list(get_smart_retriever().invoke(state["question"]))
    return {
        "documents": docs,
        "reasoning_steps": [f"Retrieved {len(docs)} candidate passages"],
    }


def grade_documents(state: AgentState) -> dict:
    if not state["documents"]:
        return {
            "docs_sufficient": False,
            "reasoning_steps": ["Doc grade: no — no passages retrieved"],
        }
    grade = _grade(
        "You are grading whether the retrieved passages contain enough information "
        f"to answer the user's question. Question: {state['question']}. "
        f"Passages: {_format_docs(state['documents'], _GRADE_DOC_CHARS)}. "
        "Answer 'yes' if they are sufficient, 'no' if key information is missing.",
        default="yes",  # never loop forever on a flaky grader
    )
    sufficient = grade.binary_score == "yes"
    return {
        "docs_sufficient": sufficient,
        "reasoning_steps": [f"Doc grade: {grade.binary_score} — {grade.reason}"],
    }


def rewrite_query(state: AgentState) -> dict:
    retries = state["retries"] + 1
    response = get_llm().invoke(
        "Rewrite this question to retrieve better passages from SEC 10-K filings. "
        "The corpus contains ONLY the 10-K filings of Apple (AAPL), Microsoft (MSFT) "
        "and NVIDIA (NVDA) — if the question hints at one of them, name it explicitly. "
        "Expand abbreviations and add likely 10-K section terms (e.g. 'Risk Factors', "
        "\"Management's Discussion and Analysis\", 'net sales', 'segment'). "
        "Return ONLY the rewritten question, nothing else.\n\n"
        f"Question: {state['question']}"
    )
    # models sometimes add preamble or refuse despite "return ONLY the question";
    # fall back to the unchanged question — the retries cap still ends the loop
    lines = [l.strip().strip('"') for l in response.content.strip().splitlines() if l.strip()]
    questions = [l for l in lines if l.endswith("?")]
    new_q = questions[-1] if questions else (lines[-1] if lines else state["question"])
    if re.search(r"\b(can't|cannot|unable to)\b", new_q.lower()):
        new_q = state["question"]
    return {
        "question": new_q,  # next retrieve uses the rewritten question
        "retries": retries,
        "reasoning_steps": [f"Rewrote query (attempt {retries}): {new_q}"],
    }


def generate(state: AgentState) -> dict:
    context = _format_docs(state["documents"])
    system = (
        "You are a financial analyst assistant answering from SEC 10-K filings. "
        "Answer using ONLY the provided passages. Cite each claim as "
        "[Company, Section]. If the answer is not in the passages, respond exactly: "
        f"'{REFUSAL}' Do not use outside knowledge or guess. "
        "Numbers inside tables in the passages count as available information — "
        "when the passages contain the requested figures or facts, answer concisely "
        "with them instead of refusing. Passages are excerpts and may come from any "
        "section of a filing; judge them by their content, not by whether they name "
        "a particular section. If the passages cover the question's topic but do not "
        "mention a specific name or term used in the question, do not refuse — "
        "answer with what the passages do state about that topic."
    )
    # answer the user's original question; the rewritten one only steered retrieval
    question = state.get("original_question") or state["question"]
    human = f"Passages:\n{context}\n\nQuestion: {question}"
    if state["question"] != question:
        human += f"\n(Clarified form used for retrieval: {state['question']})"
    response = get_llm().invoke([("system", system), ("human", human)])
    generation = response.content.strip()
    if _REFUSAL_PATTERN.search(generation) and len(generation) < 200:
        generation = REFUSAL  # length guard: don't clobber real answers
    return {
        "generation": generation,
        "citations": _extract_citations(generation, state["documents"]),
        "reasoning_steps": ["Generated answer"],
    }


def _extract_citations(generation: str, documents: List[Document]) -> List[str]:
    if REFUSAL in generation:
        return []
    cites = [f"[{c.strip()}]" for c in re.findall(r"\[([^\[\]]+)\]", generation)]
    cites = list(dict.fromkeys(cites))  # dedup, keep order
    if cites:
        return cites
    # model answered but skipped inline brackets: fall back to the source docs
    return sorted(
        {
            f"[{d.metadata.get('company', 'UNKNOWN')}, {d.metadata.get('source', '10-K')}]"
            for d in documents
        }
    )


def grade_answer(state: AgentState) -> dict:
    # informational only — always proceeds to END, never loops
    if REFUSAL in state["generation"]:
        return {
            "grounded": True,
            "reasoning_steps": ["Answer grounded: yes (refused — no claims made)"],
        }
    grade = _grade(
        "You are grading whether the answer is grounded in the passages, i.e. every "
        f"factual claim is supported by them. Question: {state['question']}. "
        f"Passages: {_format_docs(state['documents'], _GRADE_DOC_CHARS)}. "
        f"Answer: {state['generation']}. "
        "Answer 'yes' if fully grounded, 'no' if it contains unsupported claims.",
        default="yes",
    )
    grounded = grade.binary_score == "yes"
    return {
        "grounded": grounded,
        "reasoning_steps": [f"Answer grounded: {grade.binary_score} — {grade.reason}"],
    }


# --- wiring ------------------------------------------------------------------


def _decide_after_grade(state: AgentState) -> str:
    # HARD CAP: at most 2 rewrites (retries goes 0 -> 1 -> 2, then always generate)
    if not state["docs_sufficient"] and state["retries"] < 2:
        return "rewrite_query"
    return "generate"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate", generate)
    g.add_node("grade_answer", grade_answer)

    g.add_edge(START, "route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        _decide_after_grade,
        {"rewrite_query": "rewrite_query", "generate": "generate"},
    )
    g.add_edge("rewrite_query", "retrieve")  # self-correction loop
    g.add_edge("generate", "grade_answer")
    g.add_edge("grade_answer", END)
    return g.compile()


app = build_graph()


def run_agent(question: str) -> dict:
    from langchain_core.callbacks import UsageMetadataCallbackHandler

    handler, trace_url, client = get_langfuse()
    usage_handler = UsageMetadataCallbackHandler()
    callbacks = [usage_handler] + ([handler] if handler else [])
    config = {"callbacks": callbacks}

    initial: AgentState = {
        "question": question,
        "original_question": question,
        "company": None,
        "documents": [],
        "generation": "",
        "citations": [],
        "retries": 0,
        "reasoning_steps": [],
        "grounded": None,
        "docs_sufficient": None,
    }

    start = time.perf_counter()
    try:
        final = app.invoke(initial, config=config)
    finally:
        if client is not None:
            client.flush()  # OTEL batching: ship spans before process exit
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    # sum usage across every LLM call in the run (route/grade/rewrite/generate)
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for per_model in usage_handler.usage_metadata.values():
        for k in usage:
            usage[k] += per_model.get(k, 0)

    result = {
        "answer": final["generation"],
        "citations": final["citations"],
        "reasoning_steps": final["reasoning_steps"],
        "retries": final["retries"],
        "grounded": final["grounded"],
        "latency_ms": latency_ms,
        # final top-5 passages the answer was generated from (for RAGAS/DeepEval)
        "contexts": [d.page_content for d in final["documents"]],
        "usage": usage,
    }
    if trace_url:
        result["trace_url"] = trace_url
    return result
