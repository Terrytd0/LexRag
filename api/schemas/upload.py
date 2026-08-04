"""Pydantic contracts for `POST /upload`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from domain.enums import DocumentStatus


class UploadResponse(BaseModel):
    """Metadata returned once a newly-uploaded PDF has been ingested (FR-1)."""

    status: Literal["success"] = "success"
    doc_id: str
    filename: str
    document_status: DocumentStatus


class DuplicateUploadResponse(BaseModel):
    """Returned instead of `UploadResponse` when the uploaded bytes match an
    already-ingested document's `content_hash` -- ingestion is skipped entirely.
    """

    status: Literal["already_exists"] = "already_exists"
    doc_id: str
    filename: str
    message: str = Field(default="Document has already been ingested.")
