"""Small local validation for PDF extraction and chunking; no external services are used."""

from collections import Counter
from pathlib import Path

from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_extractor import extract_documents


def validate_ingestion(documents_directory: Path | str = "data/documents") -> dict:
    """Return a compact report suitable for a terminal check or future automated test."""
    directory = Path(documents_directory)
    pdf_files = sorted(directory.glob("*.pdf"))
    pages = extract_documents(directory)
    chunks = chunk_pages(pages)

    return {
        "pdfs_found": len(pdf_files),
        "pages_extracted": len(pages),
        "pages_by_document": dict(sorted(Counter(page.document_filename for page in pages).items())),
        "chunks_created": len(chunks),
        "chunks_by_document": dict(sorted(Counter(chunk.document_filename for chunk in chunks).items())),
        "sample_chunks": [chunk.metadata() for chunk in chunks[:2]],
    }


if __name__ == "__main__":
    report = validate_ingestion()
    print(f"PDFs found: {report['pdfs_found']}")
    print(f"Pages extracted: {report['pages_extracted']}")
    print(f"Chunks created: {report['chunks_created']}")
    print(f"Pages by document: {report['pages_by_document']}")
    print(f"Chunks by document: {report['chunks_by_document']}")
    print("Sample chunk metadata:")
    for chunk in report["sample_chunks"]:
        print(chunk)
