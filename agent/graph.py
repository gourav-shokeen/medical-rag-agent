"""Agentic RAG built with LangGraph — corpus-agnostic orchestration.

Flow: route -> retrieve -> grade_documents -> (rewrite_query -> retrieve loop, max 2
rewrites) -> generate -> grade_answer -> END. Domain-specific text (prompts, citation
format, refusal string) comes from agent/domains.py, selected by the CORPUS env; the
graph structure, reducer, and hard caps are identical across corpora.

generate has two modes: open-ended (default) and MCQ (when `options` is supplied) —
the latter picks a single option letter grounded in the retrieved passages, which is
what the MIRAGE benchmark calls via run_agent(..., options=..., choice_only=True).
"""

import logging
import operator
import re
import time
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from agent.domains import DOMAIN
from agent.llm_provider import get_llm
from agent.retriever import get_retriever
from agent.tracing import get_langfuse

logger = logging.getLogger(__name__)

REFUSAL = DOMAIN.refusal

# Passage text per doc handed to the graders (keeps grading prompts small).
_GRADE_DOC_CHARS = 1200


class AgentState(TypedDict):
    question: str  # current retrieval query; rewrite_query may replace it
    original_question: str  # what the user actually asked; generate answers THIS
    company: Optional[str]
    source_filter: Optional[str]  # optional metadata filter (medical: source)
    documents: List[Document]
    generation: str
    citations: List[str]
    retries: int
    # reducer: each node returns only its NEW step(s); LangGraph appends them
    reasoning_steps: Annotated[List[str], operator.add]
    grounded: Optional[bool]
    # verdict of grade_documents, read by the conditional edge
    docs_sufficient: Optional[bool]
    # MCQ mode (open-ended when options is None)
    options: Optional[dict]
    choice_only: bool
    predicted_option: Optional[str]


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
        text = d.page_content[:max_chars] if max_chars else d.page_content
        parts.append(f"{DOMAIN.doc_tag(d.metadata)} {text}")
    return "\n\n".join(parts)


# --- nodes -------------------------------------------------------------------


def route(state: AgentState) -> dict:
    routed = DOMAIN.route(state["question"])
    return {
        "company": routed["company"],
        "source_filter": routed["source_filter"],
        "reasoning_steps": [routed["step"]],
    }


def retrieve(state: AgentState) -> dict:
    docs = list(
        get_retriever().invoke(state["question"], source=state.get("source_filter"))
    )
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
    response = get_llm().invoke(DOMAIN.rewrite_prompt(state["question"]))
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


def _parse_option(text: str, letters: List[str]) -> str:
    """Pull the chosen option letter out of an LLM response (lenient)."""
    upper = text.upper()
    m = re.search(
        r"\b(?:ANSWER|OPTION|BEST)\s*(?:IS|:)?\s*\(?([" + "".join(letters) + r"])\)?\b",
        upper,
    )
    if m:
        return m.group(1)
    m = re.search(r"\b([" + "".join(letters) + r"])\b", upper)  # first standalone letter
    return m.group(1) if m else letters[0]  # default: never crash the benchmark


def generate(state: AgentState) -> dict:
    context = _format_docs(state["documents"])
    question = state.get("original_question") or state["question"]

    if state.get("options"):
        letters = list(state["options"].keys())
        opts_text = "\n".join(f"{k}. {v}" for k, v in state["options"].items())
        response = get_llm().invoke(
            [
                ("system", DOMAIN.mcq_system()),
                (
                    "human",
                    f"Passages:\n{context}\n\nQuestion: {question}\n\n"
                    f"Options:\n{opts_text}\n\n"
                    "Choose the single best option. Start your reply with the letter.",
                ),
            ]
        )
        rationale = response.content.strip()
        letter = _parse_option(rationale, letters)
        generation = letter if state.get("choice_only") else rationale
        citations = [] if state.get("choice_only") else _extract_citations(
            rationale, state["documents"]
        )
        return {
            "generation": generation,
            "predicted_option": letter,
            "citations": citations,
            "reasoning_steps": [f"Selected option {letter} (grounded in passages)"],
        }

    # open-ended mode
    human = f"Passages:\n{context}\n\nQuestion: {question}"
    if state["question"] != question:
        human += f"\n(Clarified form used for retrieval: {state['question']})"
    response = get_llm().invoke([("system", DOMAIN.generate_system()), ("human", human)])
    generation = response.content.strip()
    if DOMAIN.refusal_pattern.search(generation) and len(generation) < 200:
        generation = REFUSAL  # length guard: don't clobber real answers
    generation = DOMAIN.finalize(generation)  # e.g. append the medical not-advice frame
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
    return sorted({DOMAIN.doc_tag(d.metadata) for d in documents})


def grade_answer(state: AgentState) -> dict:
    # informational only — always proceeds to END, never loops
    if state.get("options"):
        return {
            "grounded": True,
            "reasoning_steps": [
                f"Answer grounded: n/a (MCQ — selected {state.get('predicted_option')})"
            ],
        }
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


def run_agent(
    question: str, options: Optional[dict] = None, choice_only: bool = False
) -> dict:
    """Run the agent. Open-ended when options is None; MCQ when options is a
    {letter: text} dict — then the result includes `predicted_option`.
    """
    from langchain_core.callbacks import UsageMetadataCallbackHandler

    handler, trace_url, client = get_langfuse()
    usage_handler = UsageMetadataCallbackHandler()
    callbacks = [usage_handler] + ([handler] if handler else [])
    config = {"callbacks": callbacks}

    initial: AgentState = {
        "question": question,
        "original_question": question,
        "company": None,
        "source_filter": None,
        "documents": [],
        "generation": "",
        "citations": [],
        "retries": 0,
        "reasoning_steps": [],
        "grounded": None,
        "docs_sufficient": None,
        "options": options,
        "choice_only": choice_only,
        "predicted_option": None,
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
    if options is not None:
        result["predicted_option"] = final["predicted_option"]
    if trace_url:
        result["trace_url"] = trace_url
    return result
