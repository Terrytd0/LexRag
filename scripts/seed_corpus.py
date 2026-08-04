"""Ingest every PDF in data/raw/sample_contracts/ through the ingestion pipeline.

Usage:
    uv run python scripts/seed_corpus.py
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path

from configs.logging import configure_logging
from ingestion.pipeline import ChunkIndexer, IngestionPipeline
from ingestion.repository import DocumentRepository, get_mongo_client
from retrieval.embedding import get_embedding_service
from retrieval.keyword_store.elasticsearch_store import (
    ElasticsearchKeywordStore,
    get_elasticsearch_client,
)
from retrieval.vector_store.qdrant_store import QdrantVectorStore, get_qdrant_client

logger = logging.getLogger(__name__)

_CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "sample_contracts"


def main() -> None:
    """Walk `_CORPUS_DIR` and ingest each `.pdf` found there.

    Dual-writes every document to MongoDB, Qdrant, and Elasticsearch
    (`docs/architecture.md` §3/§4) -- Qdrant and Elasticsearch are registered
    as `ChunkIndexer`s alongside the Day 2 MongoDB write, not a parallel
    pipeline.
    """
    configure_logging()

    pdf_paths = sorted(_CORPUS_DIR.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("no PDFs found in %s -- see its README.md", _CORPUS_DIR)
        return

    repository = DocumentRepository(get_mongo_client())
    embedding_service = get_embedding_service()
    indexers: list[ChunkIndexer] = [
        QdrantVectorStore(get_qdrant_client(), embedding_service),
        ElasticsearchKeywordStore(get_elasticsearch_client()),
    ]
    pipeline = IngestionPipeline(repository, indexers=indexers)

    for path in pdf_paths:
        doc_id = str(uuid.uuid4())
        contents = path.read_bytes()
        content_hash = hashlib.sha256(contents).hexdigest()
        if pipeline.find_existing_document(content_hash) is not None:
            logger.info("skipping already-ingested file path=%s", path)
            continue
        pipeline.ingest(doc_id, path, content_hash, len(contents))


if __name__ == "__main__":
    main()
