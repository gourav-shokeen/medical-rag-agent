"""Per-corpus prompts, citations, and refusal strings for the agent graph.

The graph STRUCTURE (nodes, reducer, hard caps, structured-output grading) is
held in agent/graph.py; only the domain-specific text lives here. CORPUS selects
the domain.
"""

import re

from agent.config import CORPUS

# Standing frame appended to every open-ended medical answer (not refusals/MCQ).
MED_FRAME = (
    "This is an educational summary of published references, not medical advice; "
    "it does not diagnose or recommend treatment for an individual — consult a clinician."
)


class MedicalDomain:
    name = "medical"
    refusal = "This is not covered in the available medical references."
    refusal_pattern = re.compile(
        r"not covered in the available medical references"
        r"|information is not available in the (passages|context|references|provided)",
        re.IGNORECASE,
    )

    def doc_tag(self, meta: dict) -> str:
        return f"[{meta.get('source', 'reference')}: {meta.get('title', 'untitled')}]"

    def route(self, question: str) -> dict:
        # no company logic; optional source hint, default no filter
        ql = question.lower()
        source = "statpearls" if "statpearls" in ql else (
            "textbook" if "textbook" in ql else None
        )
        return {
            "company": None,
            "source_filter": source,
            "step": (
                f"Routed: medical question ({source or 'no source filter'}), "
                "retrieval needed"
            ),
        }

    def rewrite_prompt(self, question: str) -> str:
        return (
            "Rewrite this question to retrieve better passages from a corpus of "
            "medical textbooks and clinical reference articles. Expand abbreviations "
            "(e.g. CAP -> community-acquired pneumonia, MI -> myocardial infarction), "
            "and add likely clinical terms (diagnosis, first-line treatment, "
            "pathophysiology, management, presentation). "
            "Return ONLY the rewritten question, nothing else.\n\n"
            f"Question: {question}"
        )

    def generate_system(self) -> str:
        return (
            "You are a medical reference assistant answering from textbook and clinical "
            "reference passages. Answer using ONLY the provided passages. Cite each claim "
            "as [source: title] using the passage labels. If the passages do not cover "
            f"the question, respond exactly: '{self.refusal}' Do not use outside knowledge "
            "or guess.\n"
            "SAFETY: This is an educational summary of published references, not medical "
            "advice; it does not diagnose or recommend treatment for any individual. If the "
            "user asks for personalized diagnosis or treatment for themselves or someone "
            "else (e.g. 'I have X, what should I do'), do NOT provide it: say you cannot give "
            "personal medical advice, urge them to consult a clinician (or seek emergency "
            "care if symptoms may be urgent), and you may add only general educational "
            "information from the passages."
        )

    def mcq_system(self) -> str:
        return (
            "You are a medical exam assistant. Using ONLY the provided reference passages, "
            "choose the single best answer option for the question. Respond with the option "
            "letter, then a one-line rationale grounded in the passages and cited as "
            "[source: title]. This is educational, not medical advice."
        )

    def finalize(self, generation: str) -> str:
        # guarantee the not-advice frame on real answers (small models forget it)
        if self.refusal in generation or MED_FRAME[:40] in generation:
            return generation
        return f"{generation}\n\n{MED_FRAME}"


_DOMAINS = {"medical": MedicalDomain}
DOMAIN = _DOMAINS[CORPUS]()
