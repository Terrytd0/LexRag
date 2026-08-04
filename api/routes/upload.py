"""`POST /upload`: accept a PDF, run it through `IngestionPipeline`, return
document metadata. HTTP concerns only -- ingestion logic lives in
`ingestion.pipeline.IngestionPipeline` (CLAUDE.md "Keep business logic out of
API routes").
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from api.dependencies import get_ingestion_pipeline
from api.schemas.upload import DuplicateUploadResponse, UploadResponse
from api.storage import RAW_STORAGE_DIR
from ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingestion"])

_ALLOWED_CONTENT_TYPE = "application/pdf"


@router.post(
    "/upload",
    response_model=UploadResponse | DuplicateUploadResponse,
    responses={
        201: {"model": UploadResponse, "description": "Document ingested successfully."},
        200: {
            "model": DuplicateUploadResponse,
            "description": "Uploaded bytes match an already-ingested document; "
            "ingestion was skipped.",
        },
    },
)
async def upload_document(
    file: UploadFile,
    response: Response,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> UploadResponse | DuplicateUploadResponse:
    """Ingest an uploaded PDF and return its stored `doc_id` and status (FR-1).

    Duplicate detection runs *before* any embedding/indexing work: the raw
    bytes are hashed (SHA-256) and checked against `content_hash` on existing
    documents. A match skips the pipeline entirely -- no MongoDB write, no
    Qdrant/Elasticsearch indexing -- and returns `DuplicateUploadResponse`
    for a near-instant response instead of re-running ingestion.
    """
    if file.content_type != _ALLOWED_CONTENT_TYPE and not (file.filename or "").lower().endswith(
        ".pdf"
    ):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Only PDF uploads are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    content_hash = hashlib.sha256(contents).hexdigest()
    existing = await run_in_threadpool(pipeline.find_existing_document, content_hash)
    if existing is not None:
        logger.info(
            "upload skipped (duplicate) doc_id=%s filename=%s content_hash=%s",
            existing.doc_id,
            existing.filename,
            content_hash,
        )
        response.status_code = status.HTTP_200_OK
        return DuplicateUploadResponse(doc_id=existing.doc_id, filename=existing.filename)

    doc_id = str(uuid.uuid4())
    # `IngestionPipeline.ingest` derives `Document.filename`/`Chunk.source_filename`
    # from the saved path's name, so the original filename is preserved by storing
    # under a per-doc_id directory rather than renaming the file to the doc_id --
    # citations should show "contract.pdf", not a UUID. `Path(...).name` strips any
    # directory components from the client-supplied filename to prevent writing
    # outside `_RAW_STORAGE_DIR` (path traversal).
    original_filename = Path(file.filename or f"{doc_id}.pdf").name
    dest_dir = RAW_STORAGE_DIR / doc_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / original_filename
    dest.write_bytes(contents)

    logger.info(
        "upload request received doc_id=%s filename=%s size_bytes=%d",
        doc_id,
        file.filename,
        len(contents),
    )
    start = time.monotonic()
    try:
        document = await run_in_threadpool(
            pipeline.ingest, doc_id, dest, content_hash, len(contents)
        )
    except Exception as exc:
        logger.exception("upload failed doc_id=%s", doc_id)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to process the uploaded document.",
        ) from exc

    duration = time.monotonic() - start
    logger.info(
        "upload request complete doc_id=%s status=%s duration=%.3f",
        doc_id,
        document.status,
        duration,
    )
    response.status_code = status.HTTP_201_CREATED
    return UploadResponse(
        doc_id=document.doc_id, filename=document.filename, document_status=document.status
    )
