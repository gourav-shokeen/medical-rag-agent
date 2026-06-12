"""Per-corpus prompts, citations, and refusal strings for the agent graph.

The graph STRUCTURE (nodes, reducer, hard caps, structured-output grading) is
identical across corpora — only the domain-specific text differs. CORPUS selects
the domain; the finance strings are kept byte-identical to the original graph so
the finance path is an exact regression.
"""

import re

from agent.config import CORPUS

# Standing frame appended to every open-ended medical answer (not refusals/MCQ).
MED_FRAME = (
    "This is an educational summary of published references, not medical advice; "
    "it does not diagnose or recommend treatment for an individual — consult a clinician."
)


class FinanceDomain:
    name = "finance"
    refusal = "This information is not available in the filings."
    refusal_pattern = re.compile(
        r"information is not available in the (filings|passages|context|documents|provided)",
        re.IGNORECASE,
    )

    def doc_tag(self, meta: dict) -> str:
        return f"[{meta.get('company', 'UNKNOWN')}, {meta.get('source', '10-K')}]"

    def route(self, question: str) -> dict:
        from agent.retriever import detect_companies

        companies = detect_companies(question)
        company = ",".join(companies) if companies else None
        label = company if company else "none (searching all filings)"
        return {
            "company": company,
            "source_filter": None,
            "step": f"Routed: detected company={label}, retrieval needed",
        }

    def rewrite_prompt(self, question: str) -> str:
        return (
            "Rewrite this question to retrieve better passages from SEC 10-K filings. "
            "The corpus contains ONLY the 10-K filings of Apple (AAPL), Microsoft (MSFT) "
            "and NVIDIA (NVDA) — if the question hints at one of them, name it explicitly. "
            "Expand abbreviations and add likely 10-K section terms (e.g. 'Risk Factors', "
            "\"Management's Discussion and Analysis\", 'net sales', 'segment'). "
            "Return ONLY the rewritten question, nothing else.\n\n"
            f"Question: {question}"
        )

    def generate_system(self) -> str:
        return (
            "You are a financial analyst assistant answering from SEC 10-K filings. "
            "Answer using ONLY the provided passages. Cite each claim as "
            "[Company, Section]. If the answer is not in the passages, respond exactly: "
            f"'{self.refusal}' Do not use outside knowledge or guess. "
            "Numbers inside tables in the passages count as available information — "
            "when the passages contain the requested figures or facts, answer concisely "
            "with them instead of refusing. Passages are excerpts and may come from any "
            "section of a filing; judge them by their content, not by whether they name "
            "a particular section. If the passages cover the question's topic but do not "
            "mention a specific name or term used in the question, do not refuse — "
            "answer with what the passages do state about that topic."
        )

    def mcq_system(self) -> str:
        return (
            "You are a financial analyst assistant. Using ONLY the provided passages, "
            "choose the single best answer option. State the letter, then a one-line "
            "rationale citing [Company, Section]."
        )

    def finalize(self, generation: str) -> str:
        return generation


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


_DOMAINS = {"finance": FinanceDomain, "medical": MedicalDomain}
DOMAIN = _DOMAINS[CORPUS]()
