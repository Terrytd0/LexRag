"""Runs one golden case through retrieval -> rerank -> generation, timing each stage.

Deliberately mirrors `generation.pipeline.QueryPipeline.answer()`'s exact call
sequence against the *same* retriever/reranker/generator instances used in
production (`docs/architecture.md` §2.1: "evaluation/ runs the same QueryPipeline
against the golden dataset ... so both paths exercise identical retrieval/rerank/
generation logic"). The sequence is reimplemented here, rather than calling
`QueryPipeline.answer()` directly, only so each stage's latency can be captured
without adding evaluation-only instrumentation to the production pipeline.
`retrieval_latency_s` times the real `HybridRetriever.retrieve()` call alone --
the extra dense-only/sparse-only calls (needed for the retrieval-strategy
comparison, `evaluation.metrics.retrieval`) run after that timed call, and are
never counted toward it or toward `end_to_end_latency_s`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from configs.settings import Settings, get_settings
from domain.generation import GenerationResult
from domain.retrieval import RetrievalResult
from generation.generator import GenerationService
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker.cross_encoder import CrossEncoderReranker
from retrieval.sparse import SparseRetriever


@dataclass(frozen=True)
class CaseRun:
    """Everything the evaluation metrics need from running one golden case: rankings
    from all three retrieval strategies, the reranked evidence, the generated
    result, and per-stage latency. Internal to the harness, never serialized
    directly -- the report is built from metrics derived from this, not from it.
    """

    dense_results: list[RetrievalResult]
    sparse_results: list[RetrievalResult]
    hybrid_results: list[RetrievalResult]
    reranked: list[RetrievalResult]
    result: GenerationResult
    retrieval_latency_s: float
    reranker_latency_s: float
    generation_latency_s: float

    @property
    def end_to_end_latency_s(self) -> float:
        return self.retrieval_latency_s + self.reranker_latency_s + self.generation_latency_s


async def run_case(
    question: str,
    dense: DenseRetriever,
    sparse: SparseRetriever,
    hybrid: HybridRetriever,
    reranker: CrossEncoderReranker,
    generator: GenerationService,
    settings: Settings | None = None,
) -> CaseRun:
    """Run `question` through hybrid retrieval, rerank, and generation -- the same
    path `QueryPipeline.answer()` takes -- plus separate dense-only/sparse-only
    retrieval for the strategy comparison.
    """
    settings = settings or get_settings()

    retrieval_start = time.monotonic()
    hybrid_results = await hybrid.retrieve(question)
    retrieval_latency = time.monotonic() - retrieval_start

    dense_results, sparse_results = await asyncio.gather(
        asyncio.to_thread(dense.retrieve, question),
        asyncio.to_thread(sparse.retrieve, question),
    )

    rerank_input = hybrid_results[: settings.rerank_input_top_k]
    rerank_start = time.monotonic()
    reranked = await asyncio.to_thread(reranker.rerank, question, rerank_input)
    reranker_latency = time.monotonic() - rerank_start

    generation_start = time.monotonic()
    result = await asyncio.to_thread(generator.generate, question, reranked)
    generation_latency = time.monotonic() - generation_start

    return CaseRun(
        dense_results=dense_results,
        sparse_results=sparse_results,
        hybrid_results=hybrid_results,
        reranked=reranked,
        result=result,
        retrieval_latency_s=retrieval_latency,
        reranker_latency_s=reranker_latency,
        generation_latency_s=generation_latency,
    )
