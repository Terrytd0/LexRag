"""End-to-end query pipeline: `HybridRetriever` -> trim to `RERANK_INPUT_TOP_K` ->
`CrossEncoderReranker` -> `GenerationService`. `api/routes/query.py` is the one
caller today; `evaluation/` runs the same `QueryPipeline` against the golden
dataset on Day 5, so both paths exercise identical retrieval/rerank/generation
logic (`docs/architecture.md` §2.1).

Only depends on the retrieval/generation abstractions in that chain -- never on
Qdrant/Elasticsearch/MongoDB clients directly (CLAUDE.md "Maintain architectural
boundaries").
"""

from __future__ import annotations

import asyncio
import logging
import time

from configs.settings import Settings, get_settings
from domain.generation import GenerationResult
from generation.generator import GenerationService
from retrieval.hybrid import HybridRetriever
from retrieval.reranker.cross_encoder import CrossEncoderReranker

logger = logging.getLogger(__name__)


class QueryPipeline:
    """Composes retrieval, reranking, and generation into one query-answering call."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        generator: GenerationService,
        settings: Settings | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._settings = settings or get_settings()

    async def answer(
        self, question: str, document_ids: list[str] | None = None
    ) -> GenerationResult:
        """Retrieve, rerank, and generate an answer for `question`.

        `document_ids`, if given, scopes retrieval to those documents only
        (`POST /query`'s document-scoped filter) -- passed straight through
        to `HybridRetriever.retrieve`, which applies it as a native
        Qdrant/Elasticsearch filter. Every citation in the result is
        therefore already restricted to `document_ids` by construction; no
        separate citation-filtering step is needed.

        The RRF-fused ranking is trimmed to `RERANK_INPUT_TOP_K` before
        reranking -- the cross-encoder is CPU-bound and by far the dominant
        cost in query latency (measured ~3.4s/candidate), so this bounds how
        many candidates pay that cost without touching `RETRIEVAL_TOP_K` (and
        therefore not affecting dense/sparse recall or RRF fusion itself --
        `HybridRetriever.retrieve` is untouched, so retrieval-only metrics
        stay measurable independent of this trim, per `docs/architecture.md`
        §2.6's "measured separately" goal).

        Reranking and generation are both blocking (cross-encoder inference,
        synchronous LLM SDK call) so each runs in a thread pool rather than
        blocking the event loop (CLAUDE.md "Async-first").
        """
        start = time.monotonic()
        retrieved = await self._retriever.retrieve(question, document_ids=document_ids)
        rerank_input = retrieved[: self._settings.rerank_input_top_k]
        reranked = await asyncio.to_thread(self._reranker.rerank, question, rerank_input)
        result = await asyncio.to_thread(self._generator.generate, question, reranked)

        duration = time.monotonic() - start
        logger.info(
            "query pipeline complete retrieved_count=%d rerank_input_count=%d "
            "reranked_count=%d citation_count=%d refused=%s document_ids=%s duration=%.3f",
            len(retrieved),
            len(rerank_input),
            len(reranked),
            len(result.citations),
            result.refused,
            document_ids,
            duration,
        )
        return result
