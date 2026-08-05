"""One-off model-comparison run (Sprint 5 Day 5 follow-up): re-runs the full
golden-dataset evaluation with only `LLM_MODEL` changed, capturing token
usage/cost alongside the same metrics `scripts/run_evaluation.py` produces.

Does not modify `scripts/run_evaluation.py`, `evaluation/harness.py`, or any
other production harness/config code -- this script wires its own
usage-instrumented OpenAI clients (`evaluation/cost_tracking.py`) and calls the
same `evaluation.harness.run_evaluation` unchanged, writing its report to
`evaluation/reports/model_comparison/<model>/` so it never touches
`evaluation/reports/latest.*` (the existing gpt-5.6-luna baseline).

Usage:
    LLM_MODEL=gpt-5.4-nano uv run python scripts/run_model_comparison.py
    LLM_MODEL=gpt-5.6-luna uv run python scripts/run_model_comparison.py

Pricing (standard, non-batch tier; USD per 1M tokens) below was fetched from
https://developers.openai.com/api/docs/pricing on 2026-08-05 -- not estimated.
See docs/experiments/evaluation_notes_gpt54nano.md for the verification method.
Add a verified (not guessed) entry to PRICING before running against a third
model.
"""

from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI, OpenAI

from configs.logging import configure_logging
from configs.settings import get_settings
from evaluation.cost_tracking import (
    ModelPricing,
    UsageTracker,
    instrument_async_chat_client,
    instrument_async_embeddings_client,
    instrument_sync_chat_client,
)
from evaluation.harness import run_evaluation
from evaluation.metrics.generation import GenerationJudge
from evaluation.report_writer import REPORTS_DIR, write_reports
from generation.generator import GenerationService
from generation.providers import OpenAIProvider
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

PRICING: dict[str, ModelPricing] = {
    "gpt-5.4-nano": ModelPricing(input_per_million=0.20, output_per_million=1.25),
    # Short-context standard rate -- this evaluation's prompts (system + up to
    # 8 reranked chunks + question) run to a few thousand tokens, nowhere near
    # any documented long-context threshold for this model family.
    "gpt-5.6-luna": ModelPricing(input_per_million=0.20, output_per_million=1.20),
}
EMBEDDING_PRICING = ModelPricing(input_per_million=0.02, output_per_million=0.0)


async def main() -> None:
    """Wire usage-instrumented clients for `settings.llm_model`, run the full
    golden dataset, and print a cost + metrics summary.
    """
    configure_logging()
    settings = get_settings()
    model = settings.llm_model
    if model not in PRICING:
        raise ValueError(
            f"No verified pricing for {model!r} -- add an entry to PRICING in "
            "this script (verified against OpenAI's pricing page, not guessed) "
            "before running the comparison against it."
        )

    tracker = UsageTracker()

    generation_client = OpenAI(api_key=settings.openai_api_key or None)
    instrument_sync_chat_client(generation_client, tracker, "generation")
    generator = GenerationService(OpenAIProvider(client=generation_client, model=model))

    judge_client = AsyncOpenAI(api_key=settings.openai_api_key or None)
    instrument_async_chat_client(judge_client, tracker, "judge")
    instrument_async_embeddings_client(judge_client, tracker, "judge_embedding")
    judge = GenerationJudge(settings=settings, client=judge_client)

    embedding_service = get_embedding_service()
    vector_store = QdrantVectorStore(get_qdrant_client(), embedding_service)
    keyword_store = ElasticsearchKeywordStore(get_elasticsearch_client())
    dense = DenseRetriever(vector_store, embedding_service)
    sparse = SparseRetriever(keyword_store)
    hybrid = HybridRetriever(dense, sparse)
    reranker = get_reranker()
    repository = DocumentRepository(get_mongo_client())

    report = await run_evaluation(dense, sparse, hybrid, reranker, generator, repository, judge)

    out_dir = REPORTS_DIR / "model_comparison" / model
    md_path, json_path = write_reports(report, out_dir=out_dir)

    chat_pricing = PRICING[model]
    generation_cost = tracker.cost_for_labels({"generation"}, chat_pricing)
    judge_chat_cost = tracker.cost_for_labels({"judge"}, chat_pricing)
    judge_embedding_cost = tracker.cost_for_labels({"judge_embedding"}, EMBEDDING_PRICING)
    total_cost = generation_cost + judge_chat_cost + judge_embedding_cost
    cost_per_question = total_cost / report.case_count if report.case_count else 0.0

    hybrid_retrieval = report.retrieval["hybrid"]
    print(f"\nModel comparison run complete: model={model}")
    print(f"Report: {md_path}\nJSON: {json_path}")
    print(f"Prompt tokens: {tracker.prompt_tokens}")
    print(f"Completion tokens: {tracker.completion_tokens}")
    print(f"Call counts: {tracker.call_count}")
    print(f"Generation cost: ${generation_cost:.4f}")
    print(f"Judge (chat) cost: ${judge_chat_cost:.4f}")
    print(f"Judge (embedding) cost: ${judge_embedding_cost:.4f}")
    print(f"TOTAL cost: ${total_cost:.4f}")
    print(f"Cost per evaluated question ({report.case_count} cases): ${cost_per_question:.4f}")
    print(
        f"Retrieval (hybrid): recall@10={hybrid_retrieval.recall_at_10:.2f} "
        f"precision@5={hybrid_retrieval.precision_at_5:.2f}"
    )
    print(
        f"Generation: faithfulness={report.generation.faithfulness:.2f} "
        f"context_precision={report.generation.context_precision:.2f} "
        f"context_recall={report.generation.context_recall:.2f} "
        f"answer_relevancy={report.generation.answer_relevancy:.2f}"
    )
    print(
        f"Refusal accuracy: {report.refusal.accuracy:.2%} "
        f"false_refusals={report.refusal.false_refusals} "
        f"false_acceptances={report.refusal.false_acceptances}"
    )
    print(
        f"Latency: avg_generation={report.latency.avg_generation_latency_s:.2f}s "
        f"avg_end_to_end={report.latency.avg_end_to_end_latency_s:.2f}s"
    )


if __name__ == "__main__":
    asyncio.run(main())
