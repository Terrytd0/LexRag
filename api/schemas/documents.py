"""Pydantic contracts for `GET /documents`."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from domain.enums import DocumentStatus


class DocumentSummary(BaseModel):
    """One row in the document browser: ingestion status and stats for one document."""

    doc_id: str
    filename: str
    upload_timestamp: datetime
    status: DocumentStatus
    page_count: int | None
    file_size: int | None
    chunk_count: int | None
    embedding_status: DocumentStatus
    indexed_at: datetime | None
    retrieval_ready: bool


class DocumentListResponse(BaseModel):
    """All ingested documents, most recently uploaded first."""

    documents: list[DocumentSummary]
    total: int


class DeleteDocumentResponse(BaseModel):
    """Returned by `DELETE /documents/{doc_id}` once the document and its chunks
    have been removed from MongoDB, Qdrant, and Elasticsearch.
    """

    doc_id: str
    filename: str
    message: str = "Document deleted."
