"""Canonical Document model: a source PDF and its ingestion lifecycle state."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from domain.enums import DocumentStatus


class Document(BaseModel):
    """A source document and its ingestion state.

    `status` becomes READY only once every downstream store write has
    succeeded (see `docs/architecture.md` §3, dual-write consistency) --
    that invariant is enforced by `ingestion.pipeline`, not this model.

    `content_hash` (SHA-256 of the raw uploaded bytes) is what
    `IngestionPipeline.find_existing_document` checks before starting
    ingestion, so a byte-identical re-upload short-circuits into a
    near-instant response instead of re-embedding/re-indexing. `page_count`,
    `chunk_count`, and `indexed_at` are populated once ingestion completes
    (READY or FAILED) -- `None` while a document is still PROCESSING.
    """

    doc_id: str
    filename: str
    upload_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: DocumentStatus = DocumentStatus.PENDING
    content_hash: str | None = None
    file_size: int | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    indexed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
