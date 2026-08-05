from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from evaluation.harness import EvaluationReport, LatencySummary
from evaluation.metrics.generation import GenerationSummary
from evaluation.metrics.refusal import RefusalSummary
from evaluation.metrics.retrieval import RetrievalStrategyReport
from evaluation.report_writer import render_markdown, write_reports


def _empty_strategy_report(strategy: str) -> RetrievalStrategyReport:
    return RetrievalStrategyReport(
        strategy=strategy,
        recall_at_5=0.8,
        recall_at_10=0.9,
        precision_at_5=0.6,
        precision_at_10=0.5,
        case_count=10,
        per_case=[],
    )


def _report() -> EvaluationReport:
    return EvaluationReport(
        generated_at=datetime(2026, 8, 5, tzinfo=UTC),
        dataset_path="data/golden/golden_qa.jsonl",
        case_count=30,
        retrieval={
            "dense": _empty_strategy_report("dense"),
            "sparse": _empty_strategy_report("sparse"),
            "hybrid": _empty_strategy_report("hybrid"),
        },
        generation=GenerationSummary(
            faithfulness=0.92,
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
            accuracy=0.93,
            false_refusals=1,
            false_refusal_case_ids=["termination-01"],
            false_acceptances=1,
            false_acceptance_case_ids=["negative-unrelated-01"],
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


def test_render_markdown_includes_every_required_section() -> None:
    markdown = render_markdown(_report())

    assert "## Retrieval" in markdown
    assert "## Generation (RAGAS)" in markdown
    assert "## Refusal" in markdown
    assert "## Latency" in markdown
    assert "## Failures" in markdown
    assert "hybrid" in markdown
    assert "None." in markdown  # no failures in this report


def test_write_reports_writes_timestamped_and_latest_files(tmp_path: Path) -> None:
    md_path, json_path = write_reports(_report(), out_dir=tmp_path)

    assert md_path.exists()
    assert json_path.exists()
    assert (tmp_path / "latest.md").exists()
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "latest.md").read_text(encoding="utf-8") == md_path.read_text(
        encoding="utf-8"
    )


def test_write_reports_json_round_trips_the_report(tmp_path: Path) -> None:
    _, json_path = write_reports(_report(), out_dir=tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["case_count"] == 30
    assert payload["refusal"]["accuracy"] == 0.93
