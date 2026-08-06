"""CI evaluation quality gate entrypoint (FR-12, `docs/01-requirements.md` §7.6).

Loads an already-generated `EvaluationReport` JSON (default:
`evaluation/reports/latest.json`, written by `scripts/run_evaluation.py`) and exits
non-zero if any acceptance threshold in `evaluation.gate.GATE_THRESHOLDS` is missed.
Deliberately a separate script from `run_evaluation.py` -- the gate is a pure
pass/fail assertion over a report that already exists, not another pipeline run, so
CI can re-check a report without re-paying retrieval/rerank/LLM cost.

Usage:
    uv run python scripts/evaluation_gate.py [path/to/report.json]
"""

from __future__ import annotations

import sys
from pathlib import Path

from evaluation.gate import evaluate_gate, render_gate_result
from evaluation.harness import EvaluationReport

DEFAULT_REPORT_PATH = (
    Path(__file__).resolve().parent.parent / "evaluation" / "reports" / "latest.json"
)


def main() -> int:
    """Load the report, run the gate, print the result, and return an exit code."""
    report_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPORT_PATH
    if not report_path.exists():
        print(
            f"No evaluation report found at {report_path}. "
            "Run `uv run python scripts/run_evaluation.py` (or `make evaluate`) first.",
            file=sys.stderr,
        )
        return 2

    report = EvaluationReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    result = evaluate_gate(report)
    print(render_gate_result(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
