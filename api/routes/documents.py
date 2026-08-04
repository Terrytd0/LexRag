"""`GET /documents`, `DELETE /documents/{doc_id}`: list and remove ingested
documents. HTTP concerns only -- listing/deletion logic lives in
`ingestion.repository.DocumentRepository`/`ingestion.pipeline.IngestionPipeline`
(CLAUDE.md "Keep business logic out of API routes"). `GET /documents` is the
source for selecting documents in future UI work, and for `document_ids` in
`POST /query`.
"""

from __future__ import annotations

import logging
import shutil

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from api.dependencies import get_document_repository, get_ingestion_pipeline
from api.schemas.documents import DeleteDocumentResponse, DocumentListResponse, DocumentSummary
from api.storage import RAW_STORAGE_DIR
from domain.enums import DocumentStatus
from ingestion.pipeline import IngestionPipeline
from ingestion.repository import DocumentRepository

logger = logging.getLogger(__name__)
router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    repository: DocumentRepository = Depends(get_document_repository),
) -> DocumentListResponse:
    """Return every ingested document, most recently uploaded first."""
    documents = await run_in_threadpool(repository.list_documents)
    summaries = [
        DocumentSummary(
            doc_id=document.doc_id,
            filename=document.filename,
            upload_timestamp=document.upload_timestamp,
            status=document.status,
            page_count=document.page_count,
            file_size=document.file_size,
            chunk_count=document.chunk_count,
            embedding_status=document.status,
            indexed_at=document.indexed_at,
            retrieval_ready=document.status == DocumentStatus.READY,
        )
        for document in documents
    ]
    return DocumentListResponse(documents=summaries, total=len(summaries))


@router.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    doc_id: str,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> DeleteDocumentResponse:
    """Delete a document and its chunks from MongoDB, Qdrant, and Elasticsearch.

    404s if `doc_id` doesn't exist rather than silently succeeding, so a typo'd
    doc_id is caught rather than mistaken for a completed deletion.
    """
    document = await run_in_threadpool(pipeline.delete, doc_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found.")

    raw_dir = RAW_STORAGE_DIR / doc_id
    if raw_dir.exists():
        shutil.rmtree(raw_dir, ignore_errors=True)

    logger.info("document delete request complete doc_id=%s filename=%s", doc_id, document.filename)
    return DeleteDocumentResponse(doc_id=document.doc_id, filename=document.filename)
