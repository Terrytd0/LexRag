"""Canonical Citation model: evidence returned to the user alongside a generated answer."""

from __future__ import annotations

from pydantic import BaseModel


class Citation(BaseModel):
    """A pointer from a generated answer back to the exact chunk it's grounded in."""

    doc_id: str
    filename: str
    page_number: int | None
    section: str | None
    chunk_id: str
    snippet: str
