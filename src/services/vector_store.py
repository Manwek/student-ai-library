"""Small wrapper around the project's persistent local ChromaDB collection."""

from collections.abc import Sequence

import chromadb

from src.config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIRECTORY
from src.models import TextChunk


class LocalVectorStore:
    """Persist and query chunk embeddings without exposing ChromaDB to other layers."""

    def __init__(
        self,
        database_path=CHROMA_DB_DIRECTORY,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ) -> None:
        database_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(database_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Student AI Library PDF chunks"},
        )

    def upsert_chunks(
        self, chunks: Sequence[TextChunk], embeddings: Sequence[Sequence[float]]
    ) -> int:
        """Insert or replace chunks by their stable IDs, preventing duplicates on reruns."""
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")
        if not chunks:
            return 0

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata() for chunk in chunks],
            embeddings=[list(embedding) for embedding in embeddings],
        )
        return len(chunks)

    def count(self) -> int:
        """Return the number of records currently in the collection."""
        return self.collection.count()

    def get_by_ids(self, chunk_ids: Sequence[str]) -> dict:
        """Fetch known chunks for a small verification check."""
        return self.collection.get(ids=list(chunk_ids), include=["documents", "metadatas"])

    def query(self, query_embedding: Sequence[float], n_results: int = 5) -> dict:
        """Return the closest stored evidence for a future Retriever Agent."""
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero")
        record_count = self.count()
        if record_count == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        return self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(n_results, record_count),
            include=["documents", "metadatas", "distances"],
        )
