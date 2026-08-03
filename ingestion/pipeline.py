"""Ingestion pipeline: load -> chunk -> persist.

Orchestration only -- loading, chunking, and storage logic live in
`ingestion.loaders`, `ingestion.chunking`, and `ingestion.repository`
respectively; this module just wires them together in order.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Protocol

from configs.settings import Settings, get_settings
from domain.chunk import Chunk
from domain.document import Document, DocumentStatus
from ingestion.chunking.chunker import chunk_document
from ingestion.loaders.pdf_loader import load_pdf
from ingestion.repository import DocumentRepository

logger = logging.getLogger(__name__)


class ChunkIndexer(Protocol):
    """A derived index a document's chunks are written to after Mongo persistence.

    No concrete implementation exists yet -- Qdrant and Elasticsearch indexers
    are added on Sprint 5 Day 3, each satisfying this Protocol so that adding
    them is a new class plus an entry in `IngestionPipeline`'s `indexers` list,
    not a change to the orchestration logic below (`docs/architecture.md` §3).
    """

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index the given chunks for retrieval."""
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

    def ingest(self, doc_id: str, path: Path) -> Document:
        """Load, chunk, and persist a single PDF. Returns the stored Document.

        Ingestion is atomic per NFR-4: `status` only reaches READY once
        chunking, the Mongo write, and every indexer succeed. Any failure
        marks the document FAILED (never left `processing`) and re-raises,
        so a partial write is never mistaken for a queryable document.
        """
        start = time.monotonic()
        filename = path.name
        logger.info("upload started doc_id=%s filename=%s", doc_id, filename)

        document = Document(doc_id=doc_id, filename=filename, status=DocumentStatus.PROCESSING)
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
