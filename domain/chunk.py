"""Canonical Chunk model: one token-bounded slice of a Document with provenance."""

from __future__ import annotations

from pydantic import BaseModel


class Chunk(BaseModel):
    """One chunk of a document, carrying the provenance later needed for citations."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    token_count: int
    page_number: int | None = None
    section: str
    source_filename: str
