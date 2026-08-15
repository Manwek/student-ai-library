"""Retrieve evidence from the local ChromaDB collection; do not judge or answer."""

from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from src.config import DEFAULT_RETRIEVAL_COUNT
from src.ingestion.indexer import load_embedding_model
from src.services.vector_store import LocalVectorStore


@dataclass(frozen=True)
class RetrievedEvidence:
    """One retrieved source chunk and its citation-ready metadata."""

    text: str
    document_filename: str
    document_title: str
    page_number: int
    chunk_id: str
    distance: float


class RetrieverAgent:
    """Embed a question locally and return the closest stored evidence chunks."""

    def __init__(
        self,
        retrieval_count: int = DEFAULT_RETRIEVAL_COUNT,
        embedding_model: SentenceTransformer | None = None,
        vector_store: LocalVectorStore | None = None,
    ) -> None:
        if retrieval_count <= 0:
            raise ValueError("retrieval_count must be greater than zero")
        self.retrieval_count = retrieval_count
        self.embedding_model = embedding_model or load_embedding_model()
        self.vector_store = vector_store or LocalVectorStore()

    def retrieve(
        self, question: str, retrieval_count: int | None = None
    ) -> list[RetrievedEvidence]:
        """Return nearest evidence only; it does not assess relevance or create an answer."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        result_count = retrieval_count if retrieval_count is not None else self.retrieval_count
        if result_count <= 0:
            raise ValueError("retrieval_count must be greater than zero")

        question_embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        query_result = self.vector_store.query(question_embedding, n_results=result_count)

        documents = query_result.get("documents", [[]])[0] or []
        metadatas = query_result.get("metadatas", [[]])[0] or []
        distances = query_result.get("distances", [[]])[0] or []
        chunk_ids = query_result.get("ids", [[]])[0] or []

        return [
            RetrievedEvidence(
                text=text,
                document_filename=metadata["document_filename"],
                document_title=metadata["document_title"],
                page_number=int(metadata["page_number"]),
                chunk_id=metadata.get("chunk_id", chunk_id),
                distance=float(distance),
            )
            for text, metadata, distance, chunk_id in zip(documents, metadatas, distances, chunk_ids)
        ]
