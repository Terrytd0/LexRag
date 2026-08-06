from __future__ import annotations

from datetime import UTC, datetime

from evaluation.gate import evaluate_gate, render_gate_result
from evaluation.harness import EvaluationReport, LatencySummary
from evaluation.metrics.generation import GenerationSummary
from evaluation.metrics.refusal import RefusalSummary
from evaluation.metrics.retrieval import RetrievalStrategyReport


def _strategy_report(
    strategy: str, recall_at_10: float = 0.9, precision_at_5: float = 0.8
) -> RetrievalStrategyReport:
    return RetrievalStrategyReport(
        strategy=strategy,
        recall_at_5=recall_at_10,
        recall_at_10=recall_at_10,
        precision_at_5=precision_at_5,
        precision_at_10=precision_at_5,
        case_count=22,
        per_case=[],
    )


def _report(
    *,
    recall_at_10: float = 0.9,
    precision_at_5: float = 0.8,
    faithfulness: float = 0.95,
    false_acceptances: int = 0,
    false_acceptance_case_ids: list[str] | None = None,
) -> EvaluationReport:
    return EvaluationReport(
        generated_at=datetime(2026, 8, 5, tzinfo=UTC),
        dataset_path="data/golden/golden_qa.jsonl",
        case_count=30,
        retrieval={
            "dense": _strategy_report("dense"),
            "sparse": _strategy_report("sparse"),
            "hybrid": _strategy_report("hybrid", recall_at_10, precision_at_5),
        },
        generation=GenerationSummary(
            faithfulness=faithfulness,
            context_precision=0.8,
            context_recall=0.85,
            answer_relevancy=0.9,
            case_count=22,
        ),
        generation_deepeval_pass_rate={
            "faithfulness": 0.95,
            "context_precision": 0.8,
            "context_recall": 0.85,
            "answer_relevancy": 0.9,
        },
        refusal=RefusalSummary(
            accuracy=1.0 if not false_acceptances else 0.93,
            false_refusals=0,
            false_refusal_case_ids=[],
            false_acceptances=false_acceptances,
            false_acceptance_case_ids=false_acceptance_case_ids or [],
            case_count=30,
        ),
        latency=LatencySummary(
            avg_retrieval_latency_s=0.5,
            avg_reranker_latency_s=1.2,
            avg_generation_latency_s=2.0,
            avg_end_to_end_latency_s=3.7,
            case_count=30,
        ),
        failures=[],
    )


def test_gate_passes_when_every_threshold_is_met() -> None:
    result = evaluate_gate(_report())

    assert result.passed
    assert all(check.passed for check in result.checks)


def test_gate_fails_on_low_recall() -> None:
    result = evaluate_gate(_report(recall_at_10=0.5))

    assert not result.passed
    recall_check = next(c for c in result.checks if c.name == "recall_at_10")
    assert not recall_check.passed
    assert recall_check.measured == 0.5


def test_gate_fails_on_low_precision() -> None:
    result = evaluate_gate(_report(precision_at_5=0.4))

    assert not result.passed


def test_gate_fails_on_low_faithfulness() -> None:
    result = evaluate_gate(_report(faithfulness=0.6))

    assert not result.passed


def test_gate_fails_on_any_false_acceptance() -> None:
    result = evaluate_gate(
        _report(false_acceptances=1, false_acceptance_case_ids=["negative-unrelated-01"])
    )

    assert not result.passed
    refusal_check = next(c for c in result.checks if c.name == "refusal_false_acceptances")
    assert not refusal_check.passed


def test_render_gate_result_reports_pass_and_fail() -> None:
    passing = render_gate_result(evaluate_gate(_report()))
    failing = render_gate_result(evaluate_gate(_report(recall_at_10=0.1)))

    assert "GATE: PASS" in passing
    assert "GATE: FAIL" in failing
    assert "[FAIL] recall_at_10" in failing
