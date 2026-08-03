"""PDF text extraction, producing per-page text in reading order for chunking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pypdf


@dataclass(frozen=True)
class LoadedDocument:
    """Extracted PDF text, one string per page, 0-indexed in reading order.

    A plain internal dataclass, not a `domain` model: it never leaves the
    loader -> chunker handoff within a single ingestion run, so it doesn't
    need Pydantic's validation or (de)serialization.
    """

    filename: str
    pages: list[str]


def load_pdf(path: Path) -> LoadedDocument:
    """Extract text from every page of the PDF at `path`.

    A page with no extractable text (e.g. a scanned image) yields an empty
    string rather than raising -- OCR is out of scope for this pipeline.
    """
    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return LoadedDocument(filename=path.name, pages=pages)
