"""Dense retrieval: embed the query, search Qdrant, return ranked RetrievalResults."""

from __future__ import annotations

import logging

from configs.settings import Settings, get_settings
from domain.retrieval import RetrievalResult
from retrieval.embedding import EmbeddingService
from retrieval.vector_store.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Vector similarity search over Qdrant, returning ranked `RetrievalResult`s."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_service: EmbeddingService,
        settings: Settings | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Return the `top_k` most similar chunks to `query`, best dense score first.

        `document_ids`, if given, restricts the search to those documents
        (`POST /query`'s document-scoped filter) via `QdrantVectorStore`'s
        native filter -- narrowing happens in the ANN search, not by
        post-filtering results here.
        """
        limit = top_k or self._settings.retrieval_top_k
        query_vector = self._embedding_service.embed_query(query)
        matches = self._vector_store.search(query_vector, limit, document_ids=document_ids)
        logger.debug("dense retrieval complete result_count=%d", len(matches))
        return [RetrievalResult(chunk=chunk, dense_score=score) for chunk, score in matches]
