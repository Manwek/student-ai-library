"""Embed extracted chunks locally and store them in the local ChromaDB database."""

from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from src.config import (
    DOCUMENTS_DIRECTORY,
    EMBEDDING_LOCAL_FILES_ONLY,
    EMBEDDING_MODEL_NAME,
)
from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_extractor import extract_documents
from src.services.vector_store import LocalVectorStore


@dataclass(frozen=True)
class IndexingResult:
    """A compact result report for local indexing runs."""

    chunks_prepared: int
    records_indexed: int
    collection_records: int
    embedding_model: str


def load_embedding_model() -> SentenceTransformer:
    """Load the configured model locally, without an external embedding API."""
    try:
        return SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            local_files_only=EMBEDDING_LOCAL_FILES_ONLY,
        )
    except OSError as error:
        if EMBEDDING_LOCAL_FILES_ONLY:
            raise RuntimeError(
                f"The local embedding model '{EMBEDDING_MODEL_NAME}' is not available. "
                "No model files were downloaded because EMBEDDING_LOCAL_FILES_ONLY is True."
            ) from error
        raise


def index_documents() -> IndexingResult:
    """Extract, chunk, locally embed, and upsert all project PDFs into ChromaDB."""
    pages = extract_documents(DOCUMENTS_DIRECTORY)
    chunks = chunk_pages(pages)
    model = load_embedding_model()
    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    vector_store = LocalVectorStore()
    records_indexed = vector_store.upsert_chunks(chunks, embeddings)
    return IndexingResult(
        chunks_prepared=len(chunks),
        records_indexed=records_indexed,
        collection_records=vector_store.count(),
        embedding_model=EMBEDDING_MODEL_NAME,
    )


if __name__ == "__main__":
    result = index_documents()
    print(f"Embedding model: {result.embedding_model}")
    print(f"Chunks indexed: {result.records_indexed}")
    print(f"Collection records: {result.collection_records}")
