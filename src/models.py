"""Shared, serialisable data structures for the ingestion pipeline."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExtractedPage:
    """Readable text extracted from one PDF page."""

    document_filename: str
    document_title: str
    page_number: int
    text: str

    def metadata(self) -> dict[str, str | int]:
        """Return citation-friendly metadata without the page text."""
        return {
            "document_filename": self.document_filename,
            "document_title": self.document_title,
            "page_number": self.page_number,
        }


@dataclass(frozen=True)
class TextChunk:
    """A searchable text segment that never crosses a document page boundary."""

    chunk_id: str
    document_filename: str
    document_title: str
    page_number: int
    chunk_index_on_page: int
    text: str
    word_count: int

    def metadata(self) -> dict[str, str | int]:
        """Return the fields needed later for retrieval and citations."""
        return {
            "chunk_id": self.chunk_id,
            "document_filename": self.document_filename,
            "document_title": self.document_title,
            "page_number": self.page_number,
            "chunk_index_on_page": self.chunk_index_on_page,
            "word_count": self.word_count,
        }
