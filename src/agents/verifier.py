"""Deterministically verify retrieval evidence; do not answer the user's question."""

import math
from dataclasses import dataclass
from typing import Sequence

from src.agents.retriever import RetrievedEvidence
from src.config import MAX_EVIDENCE_DISTANCE

REFUSAL_MESSAGE = "Insufficient knowledge — this is not covered in the provided documents."


@dataclass(frozen=True)
class VerificationResult:
    """The verifier's decision and the evidence eligible for a later answer step."""

    allowed: bool
    best_distance: float | None
    threshold: float
    reason: str
    message: str
    selected_evidence: list[RetrievedEvidence]


class VerifierAgent:
    """Allow only relevant chunks with meaningful text and complete citation metadata."""

    def __init__(self, max_distance: float = MAX_EVIDENCE_DISTANCE) -> None:
        if max_distance < 0:
            raise ValueError("max_distance must be zero or greater")
        self.max_distance = max_distance

    def verify(
        self, question: str, retrieved_chunks: Sequence[RetrievedEvidence]
    ) -> VerificationResult:
        """Assess retrieval evidence with deterministic, explainable rules only."""
        if not question or not question.strip():
            return self._refuse("failed_empty_evidence", None)

        finite_distance_chunks = [
            chunk
            for chunk in retrieved_chunks
            if math.isfinite(chunk.distance)
        ]
        if not finite_distance_chunks:
            return self._refuse("failed_empty_evidence", None)

        best_distance = min(chunk.distance for chunk in finite_distance_chunks)
        similarity_matches = [
            chunk for chunk in finite_distance_chunks if chunk.distance <= self.max_distance
        ]
        if not similarity_matches:
            return self._refuse("failed_similarity_threshold", best_distance)

        if any(not chunk.text or not chunk.text.strip() for chunk in similarity_matches):
            return self._refuse("failed_empty_evidence", best_distance)

        if any(not self._has_citation_metadata(chunk) for chunk in similarity_matches):
            return self._refuse("failed_missing_citation_metadata", best_distance)

        return VerificationResult(
            allowed=True,
            best_distance=best_distance,
            threshold=self.max_distance,
            reason="passed_similarity_and_metadata_checks",
            message="Evidence passed deterministic verification checks.",
            selected_evidence=similarity_matches,
        )

    def _refuse(self, reason: str, best_distance: float | None) -> VerificationResult:
        """Return one consistent, safe refusal result for every failed check."""
        return VerificationResult(
            allowed=False,
            best_distance=best_distance,
            threshold=self.max_distance,
            reason=reason,
            message=REFUSAL_MESSAGE,
            selected_evidence=[],
        )

    @staticmethod
    def _has_citation_metadata(chunk: RetrievedEvidence) -> bool:
        """Require the citation fields needed by a future Answer/Citation Agent."""
        return bool(
            chunk.document_title
            and chunk.document_title.strip()
            and chunk.chunk_id
            and chunk.chunk_id.strip()
            and isinstance(chunk.page_number, int)
            and chunk.page_number > 0
        )
