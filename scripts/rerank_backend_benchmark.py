"""Reproducible latency/consistency comparison of the cross-encoder reranker's
"torch" vs. "onnx" inference backends (Sprint 5 Day 6, backing
`docs/adr/001-reranker-onnx-backend.md`).

Day 4/5 profiling (`docs/experiments/retrieval_debugging.md`,
`docs/experiments/evaluation_notes.md`) established the cross-encoder as ~95% of
end-to-end query latency on this CPU-only machine (no GPU/DirectML acceleration
path). Switching `CrossEncoder`'s backend does not change the model weights or
`rerank_score` computation in principle, so this script measures both (a) the
latency delta and (b) whether reranked ordering/scores actually stay consistent
between backends, against the real golden-set questions (not a single query) --
so the latency win can be trusted without a separate quality re-evaluation.

Uses live retrieval (Qdrant/Elasticsearch) but never calls the LLM -- free to
re-run.

Usage (with `docker compose up -d mongo qdrant elasticsearch` and the corpus seeded):
    uv run python scripts/rerank_backend_benchmark.py
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from configs.logging import configure_logging
from configs.settings import get_settings
from domain.retrieval import RetrievalResult
from evaluation.dataset import load_golden_dataset
from retrieval.dense import DenseRetriever
from retrieval.embedding import get_embedding_service
from retrieval.hybrid import HybridRetriever
from retrieval.keyword_store.elasticsearch_store import (
    ElasticsearchKeywordStore,
    get_elasticsearch_client,
)
from retrieval.reranker.cross_encoder import CrossEncoderReranker
from retrieval.sparse import SparseRetriever
from retrieval.vector_store.qdrant_store import QdrantVectorStore, get_qdrant_client

BACKENDS = ["torch", "onnx"]


@dataclass
class _BackendRun:
    backend: str
    total_latency_s: float
    per_case_reranked: list[list[RetrievalResult]]


async def _collect_rerank_inputs(
    hybrid: HybridRetriever, questions: list[str]
) -> list[list[RetrievalResult]]:
    settings = get_settings()
    inputs = []
    for question in questions:
        retrieved = await hybrid.retrieve(question)
        inputs.append(retrieved[: settings.rerank_input_top_k])
    return inputs


def _run_backend(
    backend: str, questions: list[str], rerank_inputs: list[list[RetrievalResult]]
) -> _BackendRun:
    settings = get_settings()
    reranker = CrossEncoderReranker(
        model_name=settings.rerank_model,
        top_k=settings.rerank_top_k,
        batch_size=settings.rerank_batch_size,
        backend=backend,
    )
    per_case: list[list[RetrievalResult]] = []
    start = time.monotonic()
    for question, candidates in zip(questions, rerank_inputs, strict=True):
        per_case.append(reranker.rerank(question, candidates))
    total = time.monotonic() - start
    return _BackendRun(backend=backend, total_latency_s=total, per_case_reranked=per_case)


def _compare(reference: _BackendRun, other: _BackendRun) -> None:
    max_score_delta = 0.0
    top1_agreement = 0
    n = len(reference.per_case_reranked)
    for ref_case, other_case in zip(
        reference.per_case_reranked, other.per_case_reranked, strict=True
    ):
        ref_by_id = {r.chunk.chunk_id: r.rerank_score or 0.0 for r in ref_case}
        other_by_id = {r.chunk.chunk_id: r.rerank_score or 0.0 for r in other_case}
        for chunk_id, ref_score in ref_by_id.items():
            other_score = other_by_id.get(chunk_id)
            if other_score is not None:
                max_score_delta = max(max_score_delta, abs(ref_score - other_score))
        ref_top1 = ref_case[0].chunk.chunk_id if ref_case else None
        other_top1 = other_case[0].chunk.chunk_id if other_case else None
        if ref_top1 == other_top1:
            top1_agreement += 1

    print(f"\n--- {reference.backend} vs {other.backend} ---")
    print(f"{reference.backend} total latency: {reference.total_latency_s:.3f}s")
    print(f"{other.backend} total latency:     {other.total_latency_s:.3f}s")
    speedup = (
        reference.total_latency_s / other.total_latency_s if other.total_latency_s else float("inf")
    )
    print(f"speedup: {speedup:.2f}x")
    print(f"max |rerank_score| delta across all cases/chunks: {max_score_delta:.6f}")
    print(f"top-1 chunk agreement: {top1_agreement}/{n}")


async def main() -> None:
    configure_logging()
    embedding_service = get_embedding_service()
    vector_store = QdrantVectorStore(get_qdrant_client(), embedding_service)
    keyword_store = ElasticsearchKeywordStore(get_elasticsearch_client())
    dense = DenseRetriever(vector_store, embedding_service)
    sparse = SparseRetriever(keyword_store)
    hybrid = HybridRetriever(dense, sparse)

    cases = load_golden_dataset()
    questions = [case.question for case in cases]
    print(f"Retrieving RRF-fused candidates for {len(questions)} golden questions...")
    rerank_inputs = await _collect_rerank_inputs(hybrid, questions)

    runs: dict[str, _BackendRun] = {}
    for backend in BACKENDS:
        print(f"\nRunning reranker backend={backend!r} across {len(questions)} cases...")
        runs[backend] = _run_backend(backend, questions, rerank_inputs)
        avg = runs[backend].total_latency_s / len(questions)
        print(f"backend={backend} total={runs[backend].total_latency_s:.3f}s avg/case={avg:.3f}s")

    _compare(runs["torch"], runs["onnx"])


if __name__ == "__main__":
    asyncio.run(main())
