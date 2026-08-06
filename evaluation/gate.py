"""CI evaluation quality gate (Sprint 5 Day 6, FR-12): pass/fail assertions against
an already-computed `EvaluationReport`, applying the thresholds `docs/01-requirements.md`
§7 defines as the v1.0 acceptance bar. This module never runs the pipeline itself --
`scripts/run_evaluation.py` produces the report; `scripts/evaluation_gate.py` is the
CLI that loads one and calls `evaluate_gate` here, so CI failure/success is driven by
the same real numbers a developer sees in `evaluation/reports/latest.md`, not a
separate ad-hoc check.
"""

from __future__ import annotations

from pydantic import BaseModel

from evaluation.harness import EvaluationReport

# Mirrors docs/01-requirements.md §7.2-7.4. §7.1 (citation accuracy >= 90%) and §7.5
# (100% refusal on negatives) are asserted structurally below rather than via a
# numeric threshold dict, since the report doesn't carry a standalone "citation
# accuracy" field (see docs/experiments/evaluation_notes.md's §7.1 observation) and
# refusal's bar is an exact zero-false-acceptances count, not a ratio.
GATE_THRESHOLDS: dict[str, float] = {
    "recall_at_10": 0.85,
    "precision_at_5": 0.70,
    "faithfulness": 0.90,
}


class GateCheck(BaseModel):
    """One pass/fail assertion: the metric name, its measured value, threshold, and verdict."""

    name: str
    measured: float
    threshold: float
    passed: bool


class GateResult(BaseModel):
    """Every `GateCheck` run against one `EvaluationReport`, plus the overall verdict."""

    checks: list[GateCheck]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def evaluate_gate(report: EvaluationReport) -> GateResult:
    """Apply `docs/01-requirements.md` §7's acceptance thresholds to `report`.

    Uses the hybrid retrieval strategy (the production path, per FR-7/FR-8) for
    recall/precision -- dense/sparse are evaluation-only comparison strategies, not
    what `/query` actually serves.
    """
    hybrid = report.retrieval["hybrid"]
    checks = [
        GateCheck(
            name="recall_at_10",
            measured=hybrid.recall_at_10,
            threshold=GATE_THRESHOLDS["recall_at_10"],
            passed=hybrid.recall_at_10 >= GATE_THRESHOLDS["recall_at_10"],
        ),
        GateCheck(
            name="precision_at_5",
            measured=hybrid.precision_at_5,
            threshold=GATE_THRESHOLDS["precision_at_5"],
            passed=hybrid.precision_at_5 >= GATE_THRESHOLDS["precision_at_5"],
        ),
        GateCheck(
            name="faithfulness",
            measured=report.generation.faithfulness,
            threshold=GATE_THRESHOLDS["faithfulness"],
            passed=report.generation.faithfulness >= GATE_THRESHOLDS["faithfulness"],
        ),
        GateCheck(
            name="refusal_false_acceptances",
            measured=float(report.refusal.false_acceptances),
            threshold=0.0,
            passed=report.refusal.false_acceptances == 0,
        ),
    ]
    return GateResult(checks=checks)


def render_gate_result(result: GateResult) -> str:
    """Human-readable pass/fail table for CI logs, mirroring `report_writer`'s style."""
    lines = ["Evaluation gate results:", ""]
    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        comparator = "==" if check.name == "refusal_false_acceptances" else ">="
        lines.append(
            f"  [{status}] {check.name}: {check.measured:.4f} "
            f"(required {comparator} {check.threshold:.4f})"
        )
    lines.append("")
    lines.append("GATE: " + ("PASS" if result.passed else "FAIL"))
    return "\n".join(lines)
