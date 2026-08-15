"""Deterministic tests for the Guardrail/Verifier Agent."""

import unittest

from src.agents.retriever import RetrieverAgent, RetrievedEvidence
from src.agents.verifier import REFUSAL_MESSAGE, VerifierAgent
from src.config import MAX_EVIDENCE_DISTANCE


def evidence(distance: float, text: str = "Relevant source text.") -> RetrievedEvidence:
    return RetrievedEvidence(
        text=text,
        document_filename="example.pdf",
        document_title="Example Document",
        page_number=1,
        chunk_id="example-p1-c0",
        distance=distance,
    )


class VerifierAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.verifier = VerifierAgent()
        cls.retriever = RetrieverAgent()

    def test_relevant_question_is_allowed(self) -> None:
        question = "How can AI support strategic decision-making for entrepreneurs?"
        result = self.verifier.verify(question, self.retriever.retrieve(question))
        self.assertTrue(result.allowed)
        self.assertLessEqual(result.best_distance, MAX_EVIDENCE_DISTANCE)
        self.assertGreater(len(result.selected_evidence), 0)
        self.assertEqual(result.reason, "passed_similarity_and_metadata_checks")

    def test_irrelevant_question_is_refused(self) -> None:
        question = "What is the capital city of France?"
        result = self.verifier.verify(question, self.retriever.retrieve(question))
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "failed_similarity_threshold")
        self.assertEqual(result.message, REFUSAL_MESSAGE)
        self.assertGreater(result.best_distance, MAX_EVIDENCE_DISTANCE)

    def test_distance_at_threshold_is_allowed_and_above_is_refused(self) -> None:
        allowed = self.verifier.verify("Near threshold", [evidence(MAX_EVIDENCE_DISTANCE)])
        refused = self.verifier.verify("Above threshold", [evidence(MAX_EVIDENCE_DISTANCE + 0.001)])
        self.assertTrue(allowed.allowed)
        self.assertFalse(refused.allowed)
        self.assertEqual(refused.reason, "failed_similarity_threshold")
        self.assertEqual(refused.message, REFUSAL_MESSAGE)

    def test_empty_retrieval_is_refused(self) -> None:
        result = self.verifier.verify("Any question", [])
        self.assertFalse(result.allowed)
        self.assertIsNone(result.best_distance)
        self.assertEqual(result.reason, "failed_empty_evidence")
        self.assertEqual(result.message, REFUSAL_MESSAGE)

    def test_empty_text_is_refused(self) -> None:
        result = self.verifier.verify("Relevant question", [evidence(0.5, text="   ")])
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "failed_empty_evidence")
        self.assertEqual(result.message, REFUSAL_MESSAGE)

    def test_missing_document_title_is_refused(self) -> None:
        invalid = evidence(0.5)
        invalid = RetrievedEvidence(**{**invalid.__dict__, "document_title": ""})
        result = self.verifier.verify("Relevant question", [invalid])
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "failed_missing_citation_metadata")

    def test_missing_page_number_is_refused(self) -> None:
        invalid = evidence(0.5)
        invalid = RetrievedEvidence(**{**invalid.__dict__, "page_number": 0})
        result = self.verifier.verify("Relevant question", [invalid])
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "failed_missing_citation_metadata")

    def test_missing_chunk_id_is_refused(self) -> None:
        invalid = evidence(0.5)
        invalid = RetrievedEvidence(**{**invalid.__dict__, "chunk_id": ""})
        result = self.verifier.verify("Relevant question", [invalid])
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "failed_missing_citation_metadata")


if __name__ == "__main__":
    unittest.main()
