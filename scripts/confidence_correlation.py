"""One-off analysis (not part of the regular evaluation harness): correlates the
API's exposed `confidence` field (`domain.generation.GenerationResult.confidence`
-- the top reranker score) against measured answer correctness and faithfulness
across the golden dataset.

Requested follow-up to Sprint 5 Day 5's evaluation harness: confirms or refutes
whether callers can currently treat `confidence` as an answer-quality signal,
without redesigning the metric itself.
`docs/experiments/evaluation_notes.md`'s "What the confidence field means"
section is the authoritative doc on current behaviour; this script only measures
against the golden dataset -- see that doc's new "Confidence correlation"
section for the results.

Reuses `evaluation.runner.run_case` (same retrieval/rerank/generation call
sequence as the main harness) so `confidence` here is defined identically to
production (`GenerationService._top_score`: the top reranked chunk's
`rerank_score`, or `None` if nothing was retrieved). Only scores Faithfulness
(not the other three RAGAS metrics) for the 22 positive, non-refused cases --
cheaper than a full `run_evaluation.py` pass, since this only needs one
correlate, not the full metric suite.

Usage:
    uv run python scripts/confidence_correlation.py
"""

from __future__ import annotations

import asyncio
import logging

from configs.logging import configure_logging
from configs.settings import get_settings
from evaluation.dataset import load_golden_dataset, resolve_filename_doc_ids
from evaluation.error_analysis import classify_case
from evaluation.metrics.correlation import pearson_r
from evaluation.metrics.generation import GenerationCaseMetrics, GenerationJudge
from evaluation.runner import run_case
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
    """Run retrieval+rerank+generation (+faithfulness-only judging for positive,
    non-refused cases) for every golden case, then report Pearson correlations
    between `confidence` and (a) binary correctness and (b) faithfulness.
    """
    configure_logging()
    settings = get_settings()

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

    cases = load_golden_dataset()
    filename_to_doc_id = resolve_filename_doc_ids(cases, repository)

    rows: list[tuple[str, str, float | None, float | None, bool]] = []
    for case in cases:
        logger.info("scoring case case_id=%s category=%s", case.id, case.category)
        run = await run_case(case.question, dense, sparse, hybrid, reranker, generator, settings)
        confidence = run.reranked[0].rerank_score if run.reranked else None

        faithfulness: float | None = None
        generation_metrics: GenerationCaseMetrics | None = None
        if case.category == "positive" and not run.result.refused and case.expected_answer:
            contexts = [r.chunk.text for r in run.reranked[: settings.rerank_top_k]]
            score = await judge.faithfulness.ascore(
                user_input=case.question, response=run.result.answer, retrieved_contexts=contexts
            )
            faithfulness = score.value
            # classify_case only reads `.faithfulness` off this -- the other three
            # fields aren't measured here and are not meaningful placeholders.
            generation_metrics = GenerationCaseMetrics(
                case_id=case.id,
                faithfulness=faithfulness,
                context_precision=0.0,
                context_recall=0.0,
                answer_relevancy=0.0,
            )

        failure = classify_case(case, run, filename_to_doc_id, generation_metrics)
        rows.append((case.id, case.category, confidence, faithfulness, failure is None))

    _report(rows)


def _report(rows: list[tuple[str, str, float | None, float | None, bool]]) -> None:
    all_confidence = [c for _, _, c, _, _ in rows if c is not None]
    all_correct = [1.0 if correct else 0.0 for _, _, c, _, correct in rows if c is not None]
    r_all = pearson_r(all_confidence, all_correct)

    positive_rows = [
        (c, correct) for _, cat, c, _, correct in rows if cat == "positive" and c is not None
    ]
    positive_confidence = [c for c, _ in positive_rows]
    positive_correct = [1.0 if correct else 0.0 for _, correct in positive_rows]
    r_positive_correct = pearson_r(positive_confidence, positive_correct)

    scored_confidence = [c for _, _, c, f, _ in rows if c is not None and f is not None]
    scored_faithfulness = [f for _, _, c, f, _ in rows if c is not None and f is not None]
    r_faithfulness = pearson_r(scored_confidence, scored_faithfulness)

    print(f"\n{'case_id':<28}{'category':<10}{'confidence':<12}{'faithfulness':<14}{'correct'}")
    for case_id, category, confidence, faithfulness, correct in rows:
        conf_str = f"{confidence:.3f}" if confidence is not None else "-"
        faith_str = f"{faithfulness:.3f}" if faithfulness is not None else "-"
        print(f"{case_id:<28}{category:<10}{conf_str:<12}{faith_str:<14}{correct}")

    print(f"\nPearson r, confidence vs. correct (all {len(all_confidence)} cases): {r_all:.3f}")
    print(
        f"Pearson r, confidence vs. correct (22 positive cases only): "
        f"{r_positive_correct:.3f} (n={len(positive_confidence)})"
    )
    print(
        f"Pearson r, confidence vs. faithfulness (scored cases only): "
        f"{r_faithfulness:.3f} (n={len(scored_confidence)})"
    )


if __name__ == "__main__":
    asyncio.run(main())
