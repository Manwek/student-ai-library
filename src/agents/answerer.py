"""Generate cited wording only from evidence already approved by the Verifier Agent."""

from collections.abc import Sequence

from src.agents.retriever import RetrievedEvidence
from src.agents.verifier import REFUSAL_MESSAGE, VerificationResult
from src.services.gemini_client import GeminiClient


class AnswerCitationAgent:
    """Prepare verified evidence for Gemini; it never retrieves or checks ChromaDB."""

    def __init__(self, gemini_client: GeminiClient | None = None) -> None:
        # Delayed creation means a verifier refusal never reads an API key or calls Gemini.
        self._gemini_client = gemini_client

    def answer(self, question: str, verification: VerificationResult) -> str:
        """Return a direct refusal or generate wording from the verifier's evidence only."""
        if not verification.allowed or not verification.selected_evidence:
            return REFUSAL_MESSAGE
        if not question or not question.strip():
            return REFUSAL_MESSAGE

        verified_evidence = self.format_verified_evidence(verification.selected_evidence)
        instruction_question = self._build_instruction(question, verification.selected_evidence)
        client = self._get_gemini_client()
        return client.generate_from_verified_evidence(instruction_question, verified_evidence)

    def _get_gemini_client(self) -> GeminiClient:
        if self._gemini_client is None:
            self._gemini_client = GeminiClient()
        return self._gemini_client

    @staticmethod
    def format_verified_evidence(evidence_chunks: Sequence[RetrievedEvidence]) -> str:
        """Label supplied evidence separately from Gemini's future generated wording."""
        sections = []
        for number, chunk in enumerate(evidence_chunks, start=1):
            sections.append(
                f"[Verified evidence {number}]\n"
                f"Document title: {chunk.document_title}\n"
                f"Page number: {chunk.page_number}\n"
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Text: {chunk.text}"
            )
        return "\n\n".join(sections)

    @staticmethod
    def _build_instruction(question: str, evidence_chunks: Sequence[RetrievedEvidence]) -> str:
        """Create the grounded-answer instruction and a de-duplicated citation allowlist."""
        citations = []
        seen = set()
        for chunk in evidence_chunks:
            citation = f"{chunk.document_title}, p. {chunk.page_number}"
            if citation not in seen:
                citations.append(citation)
                seen.add(citation)

        allowed_citations = "\n".join(f"- {citation}" for citation in citations)
        return (
            "Answer the user's question only from the supplied verified evidence. "
            "Do not use outside knowledge. Do not guess or infer unsupported facts. "
            f"If the evidence does not support the question, return exactly: {REFUSAL_MESSAGE}\n"
            "Do not invent citations. Cite claims only with the supplied document title and "
            "page number, using the allowed citations below.\n\n"
            "Use this exact readable format:\n"
            "Answer:\n[grounded answer]\n\n"
            "Sources:\n"
            f"{allowed_citations}\n\n"
            f"User question: {question.strip()}"
        )
