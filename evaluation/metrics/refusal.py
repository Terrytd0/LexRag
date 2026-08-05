"""Refusal-behaviour metrics: accuracy, false refusals, false acceptances
(`docs/01-requirements.md` §7.5, FR-10). Negative golden cases are scored as
first-class cases here, not folded into generation metrics -- a false acceptance
(answering a case with no supporting evidence) is the more dangerous failure mode
for a legal tool and is reported as its own number, not averaged away.
"""

from __future__ import annotations

from pydantic import BaseModel

from evaluation.dataset import GoldenCase


class RefusalCaseResult(BaseModel):
    """Expected vs. actual refusal for one golden case."""

    case_id: str
    category: str
    expected_refusal: bool
    actual_refusal: bool

    @property
    def correct(self) -> bool:
        return self.expected_refusal == self.actual_refusal


class RefusalSummary(BaseModel):
    """Aggregate refusal-behaviour metrics across the whole golden set."""

    accuracy: float
    false_refusals: int
    false_refusal_case_ids: list[str]
    false_acceptances: int
    false_acceptance_case_ids: list[str]
    case_count: int


def score_refusal_case(case: GoldenCase, actual_refusal: bool) -> RefusalCaseResult:
    """Compare `case`'s expected refusal to what the pipeline actually did."""
    return RefusalCaseResult(
        case_id=case.id,
        category=case.category,
        expected_refusal=case.expected_refusal,
        actual_refusal=actual_refusal,
    )


def summarize_refusal(results: list[RefusalCaseResult]) -> RefusalSummary:
    """Accuracy plus the two error types, each broken out with their case IDs."""
    n = len(results)
    correct = sum(1 for r in results if r.correct)
    false_refusals = [r for r in results if not r.expected_refusal and r.actual_refusal]
    false_acceptances = [r for r in results if r.expected_refusal and not r.actual_refusal]
    return RefusalSummary(
        accuracy=correct / n if n else 0.0,
        false_refusals=len(false_refusals),
        false_refusal_case_ids=[r.case_id for r in false_refusals],
        false_acceptances=len(false_acceptances),
        false_acceptance_case_ids=[r.case_id for r in false_acceptances],
        case_count=n,
    )
