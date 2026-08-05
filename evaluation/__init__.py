"""Evaluation harness (Sprint 5 Day 5): golden-dataset loader (`dataset.py`), the
pipeline runner that executes one case with per-stage timing (`runner.py`), metric
wrappers (`metrics/`), failure classification (`error_analysis.py`), the top-level
orchestrator (`harness.py`), and the report writer (`report_writer.py`).
`scripts/run_evaluation.py` is the CLI entrypoint.

`_ragas_compat.py` is a self-disabling upstream-bug workaround (ragas#2708), not
part of the harness's own logic -- see its docstring. `cost_tracking.py` is
token-usage/cost instrumentation used only by the one-off
`scripts/run_model_comparison.py`, not the regular harness (which tracks no
cost) -- see `docs/experiments/evaluation_notes_gpt54nano.md`.
"""
