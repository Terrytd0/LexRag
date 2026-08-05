"""Top-level evaluation orchestrator (Sprint 5 Day 5): runs every golden case through
the real retrieval/rerank/generation pipeline, computes all metrics, and returns one
`EvaluationReport`. `scripts/run_evaluation.py` is the CLI entrypoint that wires real
dependencies and calls `run_evaluation`; this module stays import-only so
`tests/unit/evaluation/` can exercise it against mocks (CLAUDE.md "unit tests never
hit a real ... store/LLM").
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from configs.settings import Settings, get_settings
from evaluation.dataset import load_golden_dataset, resolve_filename_doc_ids
from evaluation.error_analysis import FailureRecord, classify_case
from evaluation.metrics.generation import (
    GENERATION_THRESHOLDS,
    GenerationCaseMetrics,
    GenerationJudge,
    GenerationSummary,
    PrecomputedScoreMetric,
    score_generation_case,
    summarize_generation,
)
from evaluation.metrics.refusal import (
    RefusalCaseResult,
    RefusalSummary,
    score_refusal_case,
    summarize_refusal,
)
from evaluation.metrics.retrieval import (
    RetrievalCaseMetrics,
    RetrievalStrategyReport,
    aggregate_retrieval_metrics,
    relevant_doc_ids,
    score_retrieval_case,
)
from evaluation.runner import CaseRun, run_case
from generation.generator import GenerationService
from ingestion.repository import DocumentRepository
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker.cross_encoder import CrossEncoderReranker
from retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)

_DEEPEVAL_METRIC_NAMES = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")


class LatencySummary(BaseModel):
    """Average per-stage latency across every golden case (task section 7)."""

    avg_retrieval_latency_s: float
    avg_reranker_latency_s: float
    avg_generation_latency_s: float
    avg_end_to_end_latency_s: float
    case_count: int


class EvaluationReport(BaseModel):
    """The full output of one evaluation run: every metric, latency, and failure."""

    generated_at: datetime
    dataset_path: str
    case_count: int
    retrieval: dict[str, RetrievalStrategyReport]
    generation: GenerationSummary
    generation_deepeval_pass_rate: dict[str, float]
    refusal: RefusalSummary
    latency: LatencySummary
    failures: list[FailureRecord]


async def run_evaluation(
    dense: DenseRetriever,
    sparse: SparseRetriever,
    hybrid: HybridRetriever,
    reranker: CrossEncoderReranker,
    generator: GenerationService,
    repository: DocumentRepository,
    judge: GenerationJudge,
    dataset_path: Path | None = None,
    settings: Settings | None = None,
) -> EvaluationReport:
    """Run the full golden dataset end-to-end and return one aggregated report."""
    settings = settings or get_settings()
    cases = load_golden_dataset(dataset_path) if dataset_path else load_golden_dataset()
    filename_to_doc_id = resolve_filename_doc_ids(cases, repository)

    retrieval_per_strategy: dict[str, list[RetrievalCaseMetrics]] = {
        "dense": [],
        "sparse": [],
        "hybrid": [],
    }
    generation_per_case: list[GenerationCaseMetrics] = []
    refusal_results: list[RefusalCaseResult] = []
    failures: list[FailureRecord] = []
    runs: list[CaseRun] = []
    deepeval_pass: dict[str, list[bool]] = {name: [] for name in _DEEPEVAL_METRIC_NAMES}

    for case in cases:
        logger.info("evaluating case case_id=%s category=%s", case.id, case.category)
        run = await run_case(case.question, dense, sparse, hybrid, reranker, generator, settings)
        runs.append(run)

        refusal_results.append(score_refusal_case(case, run.result.refused))

        generation_metrics: GenerationCaseMetrics | None = None
        if case.category == "positive":
            relevant = relevant_doc_ids(case.expected_documents, filename_to_doc_id)
            retrieval_per_strategy["dense"].append(
                score_retrieval_case(case.id, "dense", run.dense_results, relevant)
            )
            retrieval_per_strategy["sparse"].append(
                score_retrieval_case(case.id, "sparse", run.sparse_results, relevant)
            )
            retrieval_per_strategy["hybrid"].append(
                score_retrieval_case(case.id, "hybrid", run.hybrid_results, relevant)
            )

            if not run.result.refused and case.expected_answer:
                contexts = [r.chunk.text for r in run.reranked[: settings.rerank_top_k]]
                generation_metrics = await score_generation_case(
                    judge,
                    case.id,
                    case.question,
                    run.result.answer,
                    case.expected_answer,
                    contexts,
                )
                generation_per_case.append(generation_metrics)
                for name in _DEEPEVAL_METRIC_NAMES:
                    score = getattr(generation_metrics, name)
                    metric = PrecomputedScoreMetric(name, score, GENERATION_THRESHOLDS[name])
                    deepeval_pass[name].append(bool(metric.is_successful()))

        failure = classify_case(case, run, filename_to_doc_id, generation_metrics)
        if failure:
            failures.append(failure)

    retrieval_report = {
        strategy: aggregate_retrieval_metrics(strategy, per_case)
        for strategy, per_case in retrieval_per_strategy.items()
    }
    pass_rates = {
        name: (sum(passes) / len(passes) if passes else 0.0)
        for name, passes in deepeval_pass.items()
    }

    return EvaluationReport(
        generated_at=datetime.now(UTC),
        dataset_path=str(dataset_path) if dataset_path else "data/golden/golden_qa.jsonl",
        case_count=len(cases),
        retrieval=retrieval_report,
        generation=summarize_generation(generation_per_case),
        generation_deepeval_pass_rate=pass_rates,
        refusal=summarize_refusal(refusal_results),
        latency=_summarize_latency(runs),
        failures=failures,
    )


def _summarize_latency(runs: list[CaseRun]) -> LatencySummary:
    n = len(runs)
    if n == 0:
        return LatencySummary(
            avg_retrieval_latency_s=0.0,
            avg_reranker_latency_s=0.0,
            avg_generation_latency_s=0.0,
            avg_end_to_end_latency_s=0.0,
            case_count=0,
        )
    return LatencySummary(
        avg_retrieval_latency_s=sum(r.retrieval_latency_s for r in runs) / n,
        avg_reranker_latency_s=sum(r.reranker_latency_s for r in runs) / n,
        avg_generation_latency_s=sum(r.generation_latency_s for r in runs) / n,
        avg_end_to_end_latency_s=sum(r.end_to_end_latency_s for r in runs) / n,
        case_count=n,
    )
