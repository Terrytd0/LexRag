from __future__ import annotations

from evaluation.metrics.generation import (
    GenerationCaseMetrics,
    PrecomputedScoreMetric,
    summarize_generation,
)


def test_precomputed_score_metric_succeeds_at_or_above_threshold() -> None:
    metric = PrecomputedScoreMetric("faithfulness", score=0.9, threshold=0.9)

    assert metric.is_successful() is True


def test_precomputed_score_metric_fails_below_threshold() -> None:
    metric = PrecomputedScoreMetric("faithfulness", score=0.5, threshold=0.9)

    assert metric.is_successful() is False


def test_precomputed_score_metric_exposes_its_name() -> None:
    metric = PrecomputedScoreMetric("context_recall", score=1.0, threshold=0.7)

    assert metric.__name__ == "context_recall"


def _case(case_id: str, score: float) -> GenerationCaseMetrics:
    return GenerationCaseMetrics(
        case_id=case_id,
        faithfulness=score,
        context_precision=score,
        context_recall=score,
        answer_relevancy=score,
    )


def test_summarize_generation_averages_across_cases() -> None:
    summary = summarize_generation([_case("c1", 1.0), _case("c2", 0.0)])

    assert summary.faithfulness == 0.5
    assert summary.case_count == 2


def test_summarize_generation_empty_is_zero_not_a_crash() -> None:
    summary = summarize_generation([])

    assert summary.faithfulness == 0.0
    assert summary.case_count == 0
