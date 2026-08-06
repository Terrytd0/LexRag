"""Validates candidate `RERANK_INPUT_TOP_K` values against the real golden-dataset
ground truth (Sprint 5 Day 6) -- the follow-up `docs/experiments/retrieval_debugging.md`
(Day 4) explicitly asked for: that sweep only had one query and a heuristic
"materially changed" proxy, because no golden dataset existed yet. This script
retrieves each golden question's RRF-fused candidates once, then reranks the same
candidates at each candidate `RERANK_INPUT_TOP_K` value and checks whether every
positive case's expected document still has a chunk in the top `RERANK_TOP_K`
reranked output -- the same set generation actually sees -- against the resolved
`doc_id` ground truth `evaluation.dataset` already provides. This is a real
recall-after-rerank check, not an internal-consistency heuristic.

Retrieval-only, no LLM calls -- free to re-run.

Usage (with `docker compose up -d mongo qdrant elasticsearch` and the corpus seeded):
    uv run python scripts/rerank_input_topk_validation.py
"""

from __future__ import annotations

import asyncio
import time

from configs.logging import configure_logging
from configs.settings import get_settings
from domain.retrieval import RetrievalResult
from evaluation.dataset import load_golden_dataset, resolve_filename_doc_ids
from ingestion.repository import DocumentRepository, get_mongo_client
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

CANDIDATE_INPUT_TOP_K = [20, 12]


async def main() -> None:
    configure_logging()
    settings = get_settings()

    embedding_service = get_embedding_service()
    vector_store = QdrantVectorStore(get_qdrant_client(), embedding_service)
    keyword_store = ElasticsearchKeywordStore(get_elasticsearch_client())
    dense = DenseRetriever(vector_store, embedding_service)
    sparse = SparseRetriever(keyword_store)
    hybrid = HybridRetriever(dense, sparse)
    repository = DocumentRepository(get_mongo_client())

    cases = load_golden_dataset()
    filename_to_doc_id = resolve_filename_doc_ids(cases, repository)

    max_k = max(CANDIDATE_INPUT_TOP_K)
    print(f"Retrieving RRF-fused top-{max_k} candidates for {len(cases)} golden questions...")
    rrf_candidates: list[list[RetrievalResult]] = []
    for case in cases:
        retrieved = await hybrid.retrieve(case.question)
        rrf_candidates.append(retrieved[:max_k])

    for input_top_k in CANDIDATE_INPUT_TOP_K:
        reranker = CrossEncoderReranker(
            model_name=settings.rerank_model,
            top_k=settings.rerank_top_k,
            batch_size=settings.rerank_batch_size,
            backend=settings.rerank_backend,
        )
        start = time.monotonic()
        missed: list[str] = []
        for case, candidates in zip(cases, rrf_candidates, strict=True):
            trimmed = candidates[:input_top_k]
            reranked = reranker.rerank(case.question, trimmed)
            if case.category != "positive":
                continue
            expected_doc_ids = {filename_to_doc_id[f] for f in case.expected_documents}
            reranked_doc_ids = {r.chunk.doc_id for r in reranked}
            if not (expected_doc_ids & reranked_doc_ids):
                missed.append(case.id)
        elapsed = time.monotonic() - start
        avg = elapsed / len(cases)
        print(
            f"\nRERANK_INPUT_TOP_K={input_top_k}: total={elapsed:.1f}s avg/case={avg:.3f}s "
            f"positive cases losing their expected document from the top "
            f"{settings.rerank_top_k}: {len(missed)}/22 {missed}"
        )


if __name__ == "__main__":
    asyncio.run(main())
