"""Hybrid retrieval: dense + sparse search run concurrently, merged via RRF.

This is the retrieval package's public interface for Day 4 -- generation
should depend only on `HybridRetriever.retrieve` and `RetrievalResult`, never
reach into `retrieval.vector_store`/`retrieval.keyword_store` or their
storage clients directly (`docs/architecture.md` §2.3/§2.4, NFR-7).
"""

from __future__ import annotations

import asyncio
import logging
import time

from configs.settings import Settings, get_settings
from domain.retrieval import RetrievalResult
from retrieval.dense import DenseRetriever
from retrieval.fusion.rrf import reciprocal_rank_fusion
from retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines `DenseRetriever` and `SparseRetriever` results via Reciprocal Rank Fusion.

    Dense and sparse search run concurrently via `asyncio.gather` over
    `asyncio.to_thread` -- the underlying Qdrant/Elasticsearch calls are
    synchronous, so this is the thread-pool pattern CLAUDE.md calls out for
    blocking I/O, applied to FR-7's "two independent network calls, not two
    sequential ones."
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        settings: Settings | None = None,
    ) -> None:
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._settings = settings or get_settings()

    async def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Run dense + sparse retrieval in parallel and return the RRF-merged ranking."""
        start = time.monotonic()
        dense_results, sparse_results = await asyncio.gather(
            asyncio.to_thread(self._dense.retrieve, query, top_k),
            asyncio.to_thread(self._sparse.retrieve, query, top_k),
        )
        merged = reciprocal_rank_fusion(dense_results, sparse_results, k=self._settings.rrf_k)

        duration = time.monotonic() - start
        logger.info(
            "hybrid retrieval complete dense_count=%d sparse_count=%d hybrid_count=%d "
            "duration=%.3f",
            len(dense_results),
            len(sparse_results),
            len(merged),
            duration,
        )
        return merged
