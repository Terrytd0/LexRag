"""Ingestion pipeline: load -> chunk -> persist.

Orchestration only -- loading, chunking, and storage logic live in
`ingestion.loaders`, `ingestion.chunking`, and `ingestion.repository`
respectively; this module just wires them together in order.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from configs.settings import Settings, get_settings
from domain.chunk import Chunk
from domain.document import Document
from domain.enums import DocumentStatus
from ingestion.chunking.chunker import chunk_document
from ingestion.loaders.pdf_loader import load_pdf
from ingestion.repository import DocumentRepository

logger = logging.getLogger(__name__)


class ChunkIndexer(Protocol):
    """A derived index a document's chunks are written to after Mongo persistence.

    Qdrant and Elasticsearch (Sprint 5 Day 3) each satisfy this Protocol so
    adding one is a new class plus an entry in `IngestionPipeline`'s
    `indexers` list, not a change to the orchestration logic below
    (`docs/architecture.md` §3).
    """

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index the given chunks for retrieval."""
        ...

    def delete_document(self, doc_id: str) -> None:
        """Remove every indexed chunk belonging to `doc_id`."""
        ...


class IngestionPipeline:
    """Coordinates document loading, chunking, and metadata persistence for one upload."""

    def __init__(
        self,
        repository: DocumentRepository,
        indexers: list[ChunkIndexer] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._repository = repository
        self._indexers = indexers or []
        self._settings = settings or get_settings()

    def find_existing_document(self, content_hash: str) -> Document | None:
        """Return the already-ingested Document matching `content_hash`, if any.

        Checked by the caller (`api/routes/upload.py`) *before* `ingest` runs --
        a byte-identical re-upload never touches embedding generation, MongoDB
        chunk writes, or the Qdrant/Elasticsearch indexers, and gets a
        near-instant response instead.
        """
        return self._repository.get_document_by_hash(content_hash)

    def delete(self, doc_id: str) -> Document | None:
        """Delete a document and its chunks from Mongo and every indexer.

        Returns the deleted `Document`, or `None` if `doc_id` doesn't exist
        (callers use this to distinguish "deleted" from "not found" -- e.g.
        `DELETE /documents/{doc_id}` returning 404 vs 200).
        """
        document = self._repository.get_document(doc_id)
        if document is None:
            return None

        for indexer in self._indexers:
            indexer.delete_document(doc_id)
        self._repository.delete_document(doc_id)
        logger.info("document deleted doc_id=%s filename=%s", doc_id, document.filename)
        return document

    def ingest(self, doc_id: str, path: Path, content_hash: str, file_size: int) -> Document:
        """Load, chunk, and persist a single PDF. Returns the stored Document.

        Ingestion is atomic per NFR-4: `status` only reaches READY once
        chunking, the Mongo write, and every indexer succeed. Any failure
        marks the document FAILED (never left `processing`) and re-raises,
        so a partial write is never mistaken for a queryable document.

        `content_hash`/`file_size` are computed by the caller from the raw
        upload bytes (before this method ever touches the filesystem) so
        `find_existing_document` can be checked ahead of a call to `ingest`.
        """
        start = time.monotonic()
        filename = path.name
        logger.info("upload started doc_id=%s filename=%s", doc_id, filename)

        document = Document(
            doc_id=doc_id,
            filename=filename,
            status=DocumentStatus.PROCESSING,
            content_hash=content_hash,
            file_size=file_size,
        )
        self._repository.save_document(document)

        try:
            loaded = load_pdf(path)
            logger.info(
                "document loaded doc_id=%s filename=%s pages=%d",
                doc_id,
                filename,
                len(loaded.pages),
            )

            chunks = chunk_document(
                doc_id=doc_id,
                filename=filename,
                pages=loaded.pages,
                settings=self._settings,
            )
            logger.info(
                "chunking complete doc_id=%s filename=%s chunk_count=%d",
                doc_id,
                filename,
                len(chunks),
            )

            self._repository.save_chunks(chunks)
            for indexer in self._indexers:
                indexer.index_chunks(chunks)
        except Exception:
            # Broad by design: any failure above must flip status to FAILED
            # before propagating, so ingestion stays atomic per NFR-4.
            document.status = DocumentStatus.FAILED
            self._repository.save_document(document)
            logger.exception("ingestion failed doc_id=%s filename=%s", doc_id, filename)
            raise

        document.status = DocumentStatus.READY
        document.page_count = len(loaded.pages)
        document.chunk_count = len(chunks)
        document.indexed_at = datetime.now(UTC)
        self._repository.save_document(document)
        logger.info(
            "document stored doc_id=%s filename=%s status=%s",
            doc_id,
            filename,
            document.status,
        )

        duration = time.monotonic() - start
        logger.info(
            "ingestion completed doc_id=%s filename=%s chunk_count=%d duration=%.3f",
            doc_id,
            filename,
            len(chunks),
            duration,
        )
        return document
