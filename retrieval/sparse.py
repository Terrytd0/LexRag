"""Sparse retrieval: BM25 keyword search over Elasticsearch, returning ranked RetrievalResults."""

from __future__ import annotations

import logging

from configs.settings import Settings, get_settings
from domain.retrieval import RetrievalResult
from retrieval.keyword_store.elasticsearch_store import ElasticsearchKeywordStore

logger = logging.getLogger(__name__)


class SparseRetriever:
    """BM25 keyword search over Elasticsearch, returning ranked `RetrievalResult`s."""

    def __init__(
        self,
        keyword_store: ElasticsearchKeywordStore,
        settings: Settings | None = None,
    ) -> None:
        self._keyword_store = keyword_store
        self._settings = settings or get_settings()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Return the `top_k` best BM25 matches for `query`, best sparse score first."""
        limit = top_k or self._settings.retrieval_top_k
        matches = self._keyword_store.search(query, limit)
        logger.debug("sparse retrieval complete result_count=%d", len(matches))
        return [RetrievalResult(chunk=chunk, sparse_score=score) for chunk, score in matches]
