"""Create stable, citation-ready chunks from extracted PDF pages."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from src.models import ExtractedPage, TextChunk

DEFAULT_CHUNK_SIZE_WORDS = 500
DEFAULT_OVERLAP_WORDS = 75


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _chunk_id(page: ExtractedPage, chunk_index: int, chunk_text: str) -> str:
    """Make a deterministic ID that changes if the source text changes."""
    source = f"{page.document_filename}|{page.page_number}|{chunk_index}|{chunk_text}"
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"{page.document_filename}-p{page.page_number}-c{chunk_index}-{content_hash}"


def chunk_pages(
    pages: Iterable[ExtractedPage],
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[TextChunk]:
    """Split each page independently into overlapping word chunks.

    Keeping chunks within a page intentionally preserves one exact page citation on every
    result. Therefore, text is never mixed between PDFs or between pages.
    """
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than zero")
    if not 0 <= overlap_words < chunk_size_words:
        raise ValueError("overlap_words must be at least zero and smaller than chunk_size_words")

    chunks: list[TextChunk] = []
    step_size = chunk_size_words - overlap_words

    for page in pages:
        words = _words(page.text)
        start = 0
        chunk_index = 0
        while start < len(words):
            chunk_words = words[start : start + chunk_size_words]
            chunk_text = " ".join(chunk_words)
            chunks.append(
                TextChunk(
                    chunk_id=_chunk_id(page, chunk_index, chunk_text),
                    document_filename=page.document_filename,
                    document_title=page.document_title,
                    page_number=page.page_number,
                    chunk_index_on_page=chunk_index,
                    text=chunk_text,
                    word_count=len(chunk_words),
                )
            )
            if start + chunk_size_words >= len(words):
                break
            start += step_size
            chunk_index += 1

    return chunks
