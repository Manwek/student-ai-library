"""Compact local validation for the Retriever Agent; no answers are generated."""

from src.agents.retriever import RetrieverAgent

QUESTIONS = [
    "How can AI support strategic decision-making for entrepreneurs and investors?",
    "What are the corporate governance consequences of autonomous AI management?",
    "Which AI-enabled value propositions are described in the digital density framework?",
    "What is the capital city of France?",
]


def preview(text: str, limit: int = 160) -> str:
    """Keep validation output compact and avoid printing source passages in full."""
    return text[:limit].rsplit(" ", 1)[0] + "…" if len(text) > limit else text


if __name__ == "__main__":
    retriever = RetrieverAgent()
    for question in QUESTIONS:
        evidence = retriever.retrieve(question)
        print(f"Question: {question}")
        print(f"Results returned: {len(evidence)}")
        for item in evidence:
            print(
                f"- {item.document_title} | p. {item.page_number} | "
                f"distance: {item.distance:.4f} | {preview(item.text)}"
            )
