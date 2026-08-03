"""MongoDB persistence for Document and Chunk domain models -- the source of truth
for provenance (`docs/architecture.md` §2.2). Qdrant/Elasticsearch (Day 3) are
derived indexes rebuilt from here, never the reverse.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection

from configs.settings import Settings, get_settings
from domain.chunk import Chunk
from domain.document import Document

_DOCUMENTS_COLLECTION = "documents"
_CHUNKS_COLLECTION = "chunks"
_EXCLUDE_MONGO_ID = {"_id": 0}


@lru_cache
def get_mongo_client() -> MongoClient[dict[str, Any]]:
    """Process-wide MongoClient singleton, mirroring `configs.settings.get_settings`."""
    return MongoClient(get_settings().mongodb_uri)


class DocumentRepository:
    """CRUD access to the `documents` and `chunks` collections."""

    def __init__(
        self,
        client: MongoClient[dict[str, Any]],
        settings: Settings | None = None,
    ) -> None:
        settings = settings or get_settings()
        db = client[settings.mongodb_db_name]
        self._documents: Collection[dict[str, Any]] = db[_DOCUMENTS_COLLECTION]
        self._chunks: Collection[dict[str, Any]] = db[_CHUNKS_COLLECTION]

    def save_document(self, document: Document) -> None:
        """Upsert a Document, keyed by `doc_id`."""
        self._documents.replace_one(
            {"doc_id": document.doc_id},
            document.model_dump(mode="json"),
            upsert=True,
        )

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Insert chunks for a document. No-op for an empty list."""
        if not chunks:
            return
        self._chunks.insert_many([chunk.model_dump(mode="json") for chunk in chunks])

    def get_document(self, doc_id: str) -> Document | None:
        """Fetch a Document by `doc_id`, or None if it doesn't exist."""
        raw = self._documents.find_one({"doc_id": doc_id}, _EXCLUDE_MONGO_ID)
        return Document.model_validate(raw) if raw else None

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        """Fetch all chunks for a document, ordered by `chunk_index`."""
        cursor = self._chunks.find({"doc_id": doc_id}, _EXCLUDE_MONGO_ID).sort("chunk_index", 1)
        return [Chunk.model_validate(raw) for raw in cursor]
