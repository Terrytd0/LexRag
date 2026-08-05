from __future__ import annotations

from evaluation.dataset import GoldenCase
from evaluation.metrics.refusal import score_refusal_case, summarize_refusal


def _positive_case(case_id: str) -> GoldenCase:
    return GoldenCase(
        id=case_id,
        topic="termination",
        category="positive",
        question="q",
        expected_answer="a",
        expected_documents=["contract.pdf"],
        expected_citations=[],
        expected_refusal=False,
    )


def _negative_case(case_id: str) -> GoldenCase:
    return GoldenCase(
        id=case_id,
        topic="unrelated",
        category="negative",
        negative_subtype="unrelated",
        question="q",
        expected_refusal=True,
    )


def test_score_refusal_case_correct_when_positive_case_answered() -> None:
    result = score_refusal_case(_positive_case("p1"), actual_refusal=False)

    assert result.correct is True


def test_score_refusal_case_false_refusal_when_positive_case_refused() -> None:
    result = score_refusal_case(_positive_case("p1"), actual_refusal=True)

    assert result.correct is False


def test_score_refusal_case_false_acceptance_when_negative_case_answered() -> None:
    result = score_refusal_case(_negative_case("n1"), actual_refusal=False)

    assert result.correct is False


def test_score_refusal_case_correct_when_negative_case_refused() -> None:
    result = score_refusal_case(_negative_case("n1"), actual_refusal=True)

    assert result.correct is True


def test_summarize_refusal_computes_accuracy_and_error_buckets() -> None:
    results = [
        score_refusal_case(_positive_case("p1"), actual_refusal=False),  # correct
        score_refusal_case(_positive_case("p2"), actual_refusal=True),  # false refusal
        score_refusal_case(_negative_case("n1"), actual_refusal=True),  # correct
        score_refusal_case(_negative_case("n2"), actual_refusal=False),  # false acceptance
    ]

    summary = summarize_refusal(results)

    assert summary.accuracy == 0.5
    assert summary.false_refusals == 1
    assert summary.false_refusal_case_ids == ["p2"]
    assert summary.false_acceptances == 1
    assert summary.false_acceptance_case_ids == ["n2"]
    assert summary.case_count == 4


def test_summarize_refusal_empty_is_zero_not_a_crash() -> None:
    summary = summarize_refusal([])

    assert summary.accuracy == 0.0
    assert summary.case_count == 0
