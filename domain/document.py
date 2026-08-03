"""Canonical Document model: a source PDF and its ingestion lifecycle state."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    """Lifecycle status of a document through the ingestion pipeline."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(BaseModel):
    """A source document and its ingestion state.

    `status` becomes READY only once every downstream store write has
    succeeded (see `docs/architecture.md` §3, dual-write consistency) --
    that invariant is enforced by `ingestion.pipeline`, not this model.
    """

    doc_id: str
    filename: str
    upload_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: dict[str, Any] = Field(default_factory=dict)
