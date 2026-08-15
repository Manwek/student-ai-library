"""One controlled end-to-end RAG validation with at most one Gemini API request."""

from src.agents.answerer import AnswerCitationAgent
from src.agents.retriever import RetrieverAgent
from src.agents.verifier import VerifierAgent
from src.config import GEMINI_MODEL
from src.services.gemini_client import is_gemini_api_key_configured

QUESTION = "How can AI support strategic decision-making for entrepreneurs?"


def citation_labels(evidence) -> list[str]:
    """Return concise, de-duplicated citation labels without printing source text."""
    return list(dict.fromkeys(f"{item.document_title}, p. {item.page_number}" for item in evidence))


def main() -> None:
    if not is_gemini_api_key_configured():
        print("Configuration error: GEMINI_API_KEY is not configured.")
        return

    retrieved_evidence = RetrieverAgent().retrieve(QUESTION)
    verification = VerifierAgent().verify(QUESTION, retrieved_evidence)
    if not verification.allowed:
        print(verification.message)
        return

    answer = AnswerCitationAgent().answer(QUESTION, verification)
    print(f"Question: {QUESTION}")
    print(
        "Verification: "
        f"allowed={verification.allowed}; "
        f"best_distance={verification.best_distance:.4f}; "
        f"reason={verification.reason}"
    )
    print("Final Gemini answer:")
    print(answer)
    print("Verified evidence citations supplied to Gemini:")
    for citation in citation_labels(verification.selected_evidence):
        print(f"- {citation}")
    print(f"Configured Gemini model: {GEMINI_MODEL}")


if __name__ == "__main__":
    main()
