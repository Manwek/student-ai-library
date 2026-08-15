"""Deterministic retrieval and guardrail checks for the evaluation question set."""

import json
from pathlib import Path

import pytest

from src.agents.retriever import RetrieverAgent
from src.agents.verifier import REFUSAL_MESSAGE, VerifierAgent


EVALUATION_FILE = Path(__file__).with_name("evaluation_questions.json")
EVALUATION_QUESTIONS = json.loads(EVALUATION_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pipeline() -> tuple[RetrieverAgent, VerifierAgent]:
    """Build only the local retrieval and verification stages; never construct GeminiClient."""
    return RetrieverAgent(), VerifierAgent()


@pytest.mark.parametrize("case", EVALUATION_QUESTIONS, ids=lambda case: case["id"])
def test_evaluation_question_matches_expected_guardrail(case, pipeline) -> None:
    retriever, verifier = pipeline
    evidence = retriever.retrieve(case["question"])
    verification = verifier.verify(case["question"], evidence)

    if case["answerable"]:
        assert verification.allowed, verification.message
        selected_documents = {item.document_filename for item in verification.selected_evidence}
        assert selected_documents.intersection(case["expected_documents"])
        assert case["expected_behavior"] == "answer"
    else:
        assert not verification.allowed
        assert verification.message == REFUSAL_MESSAGE
        assert case["expected_behavior"] == "refuse"
