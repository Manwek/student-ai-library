"""Extract readable, page-level text from the project's local PDF documents."""

from pathlib import Path

import pymupdf

from src.models import ExtractedPage


def document_title_from_filename(filename: str) -> str:
    """Create a readable display title while preserving the original filename separately."""
    return Path(filename).stem.replace("_", " ").title()


def extract_pdf(pdf_path: Path) -> list[ExtractedPage]:
    """Extract non-empty page text from one PDF using one-based PDF page numbers."""
    pages: list[ExtractedPage] = []
    document_filename = pdf_path.name
    document_title = document_title_from_filename(document_filename)

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            # Normalising whitespace makes word chunking predictable without changing the PDF.
            text = " ".join(page.get_text("text").split())
            if not text:
                continue

            pages.append(
                ExtractedPage(
                    document_filename=document_filename,
                    document_title=document_title,
                    page_number=page_number,
                    text=text,
                )
            )

    return pages


def extract_documents(documents_directory: Path | str = "data/documents") -> list[ExtractedPage]:
    """Extract every PDF in a directory, in filename order for repeatable output."""
    directory = Path(documents_directory)
    if not directory.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {directory}")

    extracted_pages: list[ExtractedPage] = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        extracted_pages.extend(extract_pdf(pdf_path))
    return extracted_pages
