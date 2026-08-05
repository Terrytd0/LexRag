"""Run the full golden-dataset evaluation end-to-end and write a report.

Usage:
    uv run python scripts/run_evaluation.py

Requires the same live stack as `scripts/seed_corpus.py` (MongoDB, Qdrant,
Elasticsearch) plus a valid `OPENAI_API_KEY`, since generation and the RAGAS judge
both call the configured LLM provider.
"""

from __future__ import annotations

import asyncio
import logging

from configs.logging import configure_logging
from evaluation.harness import run_evaluation
from evaluation.metrics.generation import GenerationJudge
from evaluation.report_writer import write_reports
from generation.generator import GenerationService
from generation.providers import get_llm_provider
from ingestion.repository import DocumentRepository, get_mongo_client
from retrieval.dense import DenseRetriever
from retrieval.embedding import get_embedding_service
from retrieval.hybrid import HybridRetriever
from retrieval.keyword_store.elasticsearch_store import (
    ElasticsearchKeywordStore,
    get_elasticsearch_client,
)
from retrieval.reranker.cross_encoder import get_reranker
from retrieval.sparse import SparseRetriever
from retrieval.vector_store.qdrant_store import QdrantVectorStore, get_qdrant_client

logger = logging.getLogger(__name__)


async def main() -> None:
    """Wire the real pipeline dependencies, run the evaluation, and print a summary."""
    configure_logging()

    embedding_service = get_embedding_service()
    vector_store = QdrantVectorStore(get_qdrant_client(), embedding_service)
    keyword_store = ElasticsearchKeywordStore(get_elasticsearch_client())
    dense = DenseRetriever(vector_store, embedding_service)
    sparse = SparseRetriever(keyword_store)
    hybrid = HybridRetriever(dense, sparse)
    reranker = get_reranker()
    generator = GenerationService(get_llm_provider())
    repository = DocumentRepository(get_mongo_client())
    judge = GenerationJudge()

    report = await run_evaluation(dense, sparse, hybrid, reranker, generator, repository, judge)
    md_path, json_path = write_reports(report)
    logger.info("evaluation complete report=%s json=%s", md_path, json_path)

    hybrid_retrieval = report.retrieval["hybrid"]
    print(f"\nEvaluation report written to:\n  {md_path}\n  {json_path}\n")
    print(
        f"Retrieval (hybrid): recall@10={hybrid_retrieval.recall_at_10:.2f} "
        f"precision@5={hybrid_retrieval.precision_at_5:.2f}"
    )
    print(f"Generation: faithfulness={report.generation.faithfulness:.2f}")
    print(f"Refusal accuracy: {report.refusal.accuracy:.2%}")
    print(f"Failures: {len(report.failures)}")


if __name__ == "__main__":
    asyncio.run(main())
