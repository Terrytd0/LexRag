"""Qdrant client wrapper: collection management, upsert, and vector search.

Satisfies `ingestion.pipeline.ChunkIndexer` so it plugs into the existing
ingestion pipeline as just another registered indexer (`docs/architecture.md`
§2.3) -- it does not generate its own embeddings; that's `retrieval.embedding`.
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models

from configs.settings import Settings, get_settings
from domain.chunk import Chunk
from retrieval.embedding import EmbeddingService

logger = logging.getLogger(__name__)

_DISTANCE_BY_NAME: dict[str, models.Distance] = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
}


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Process-wide QdrantClient singleton, mirroring `ingestion.repository.get_mongo_client`."""
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


def _point_id(chunk_id: str) -> str:
    """Deterministic UUID5 for a chunk_id -- Qdrant point IDs must be an int or UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantVectorStore:
    """Dense vector storage and search over chunk embeddings.

    Collection creation is lazy and idempotent: the first `index_chunks` or
    `search` call ensures the collection exists, using the configured name,
    vector size, and distance metric, so a fresh Qdrant instance needs no
    separate provisioning step.
    """

    def __init__(
        self,
        client: QdrantClient,
        embedding_service: EmbeddingService,
        settings: Settings | None = None,
    ) -> None:
        self._client = client
        self._embedding_service = embedding_service
        self._settings = settings or get_settings()
        self._collection = self._settings.qdrant_collection
        self._ensured = False

    def _ensure_collection(self) -> None:
        if self._ensured:
            return
        if not self._client.collection_exists(self._collection):
            distance = _DISTANCE_BY_NAME.get(
                self._settings.qdrant_distance_metric.lower(), models.Distance.COSINE
            )
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._settings.embedding_dimensions,
                    distance=distance,
                ),
            )
            logger.info(
                "qdrant collection created collection=%s size=%d distance=%s",
                self._collection,
                self._settings.embedding_dimensions,
                distance.value,
            )
        self._ensured = True

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and upsert `chunks`, keyed by a UUID5 derived from `chunk_id`.

        The full chunk (text + provenance) is stored as the point payload so
        a dense search result can be turned back into a `Chunk` without a
        round trip to MongoDB (`docs/architecture.md` §2.2 -- Mongo stays the
        source of truth; Qdrant is a self-contained derived index).
        """
        if not chunks:
            return
        self._ensure_collection()

        vectors = self._embedding_service.embed_texts([chunk.text for chunk in chunks])
        points = [
            models.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=vector,
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        logger.info(
            "qdrant indexing complete collection=%s chunk_count=%d",
            self._collection,
            len(chunks),
        )

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        """Return up to `top_k` nearest chunks to `query_vector`, most similar first."""
        self._ensure_collection()
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
        )
        return [(Chunk.model_validate(point.payload), point.score) for point in response.points]
