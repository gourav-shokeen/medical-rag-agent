"""DeepEval regression gate over the smoke subset: pytest evals/test_quality.py

Thresholds are intentionally BELOW expected performance — this gate catches
regressions, not perfection:
    Faithfulness     >= 0.80
    AnswerRelevancy  >= 0.75
    Hallucination    <= 0.15  (lower is better; deepeval inverts the comparison)

Every metric gets model=get_deepeval_model() (the fixed judge) so no
OPENAI_API_KEY is needed.
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
# A serial local judge (ollama) needs a far larger per-task budget than
# deepeval's default, or every metric dies in asyncio.wait_for. Groq (CI)
# keeps the defaults. Must be set before deepeval is imported anywhere.
if os.getenv("JUDGE_PROVIDER", "groq") == "ollama":
    os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "1800")

from evals.load_golden import load_golden  # noqa: E402

SMOKE_ROWS = load_golden(smoke_only=True)


@pytest.fixture(scope="session")
def judge():
    from evals.judge import get_deepeval_model

    return get_deepeval_model()


@pytest.fixture(scope="session")
def agent_outputs():
    """One agent output per smoke row, shared by all metric tests.

    Reuses answers already cached by run_eval.py (eval_answers_progress.jsonl) so
    DeepEval doesn't re-spend Groq quota re-answering rows RAGAS already ran; any
    row not in the cache is answered live.
    """
    import json
    from pathlib import Path

    cache_path = (Path(__file__).resolve().parent / "results"
                  / "eval_answers_progress.jsonl")
    cached = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cached[r["id"]] = {"answer": r["response"],
                                   "contexts": r["retrieved_contexts"]}

    from agent.graph import run_agent

    outputs = {}
    for row in SMOKE_ROWS:
        outputs[row["id"]] = cached.get(row["id"]) or run_agent(row["question"])
    return outputs


def _test_case(row, out):
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=row["question"],
        actual_output=out["answer"],
        expected_output=row["ground_truth"],
        # retrieval_context feeds Faithfulness; context feeds Hallucination
        retrieval_context=out["contexts"],
        context=out["contexts"],
    )


@pytest.mark.parametrize("row", SMOKE_ROWS, ids=[r["id"] for r in SMOKE_ROWS])
def test_smoke_quality(row, agent_outputs, judge):
    from deepeval import assert_test
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        HallucinationMetric,
    )

    case = _test_case(row, agent_outputs[row["id"]])
    assert_test(
        case,
        [
            FaithfulnessMetric(threshold=0.80, model=judge, include_reason=True),
            AnswerRelevancyMetric(threshold=0.75, model=judge, include_reason=True),
            HallucinationMetric(threshold=0.15, model=judge, include_reason=True),
        ],
    )
