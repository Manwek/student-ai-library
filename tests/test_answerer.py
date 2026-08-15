"""Unit tests for Answer/Citation Agent behaviour without a real Gemini call."""

import unittest

from src.agents.answerer import AnswerCitationAgent
from src.agents.retriever import RetrievedEvidence, RetrieverAgent
from src.agents.verifier import REFUSAL_MESSAGE, VerificationResult, VerifierAgent


class FakeGeminiClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate_from_verified_evidence(self, question: str, evidence: str) -> str:
        self.calls.append((question, evidence))
        return "Answer:\nA grounded response.\n\nSources:\n- Example Paper, p. 3"


def verified_chunk(page_number: int = 3) -> RetrievedEvidence:
    return RetrievedEvidence(
        text="Verified source text about AI management.",
        document_filename="example.pdf",
        document_title="Example Paper",
        page_number=page_number,
        chunk_id=f"example-p{page_number}-c0",
        distance=0.5,
    )


def allowed_verification() -> VerificationResult:
    return VerificationResult(
        allowed=True,
        best_distance=0.5,
        threshold=1.0,
        reason="passed_similarity_and_metadata_checks",
        message="Evidence passed deterministic verification checks.",
        selected_evidence=[verified_chunk(), verified_chunk()],
    )


class AnswerCitationAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_client = FakeGeminiClient()
        self.answerer = AnswerCitationAgent(gemini_client=self.fake_client)

    def test_guardrail_refusal_does_not_call_gemini(self) -> None:
        refusal = VerificationResult(False, None, 1.0, "failed_similarity_threshold", REFUSAL_MESSAGE, [])
        answer = self.answerer.answer("What is the capital city of France?", refusal)
        self.assertEqual(answer, REFUSAL_MESSAGE)
        self.assertEqual(self.fake_client.calls, [])

    def test_unanswerable_question_is_refused_without_calling_gemini(self) -> None:
        question = "What is the capital city of France?"
        evidence = RetrieverAgent().retrieve(question)
        verification = VerifierAgent().verify(question, evidence)

        answer = self.answerer.answer(question, verification)

        self.assertFalse(verification.allowed)
        self.assertEqual(answer, REFUSAL_MESSAGE)
        self.assertEqual(self.fake_client.calls, [])

    def test_verified_evidence_is_passed_to_gemini(self) -> None:
        self.answerer.answer("What does the paper say?", allowed_verification())
        self.assertEqual(len(self.fake_client.calls), 1)
        self.assertIn("Verified source text about AI management.", self.fake_client.calls[0][1])

    def test_instruction_prohibits_outside_knowledge(self) -> None:
        self.answerer.answer("What does the paper say?", allowed_verification())
        instruction = self.fake_client.calls[0][0]
        self.assertIn("only from the supplied verified evidence", instruction)
        self.assertIn("Do not use outside knowledge", instruction)
        self.assertIn("Do not guess or infer unsupported facts", instruction)

    def test_citation_metadata_is_included_in_evidence(self) -> None:
        self.answerer.answer("What does the paper say?", allowed_verification())
        evidence = self.fake_client.calls[0][1]
        self.assertIn("Document title: Example Paper", evidence)
        self.assertIn("Page number: 3", evidence)
        self.assertIn("Chunk ID: example-p3-c0", evidence)

    def test_duplicate_document_page_citations_are_deduplicated(self) -> None:
        self.answerer.answer("What does the paper say?", allowed_verification())
        instruction = self.fake_client.calls[0][0]
        self.assertEqual(instruction.count("- Example Paper, p. 3"), 1)


if __name__ == "__main__":
    unittest.main()
