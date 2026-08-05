"""Writes an `EvaluationReport` to disk under `evaluation/reports/`: a
machine-readable JSON file and a human-readable Markdown summary (task section
7/9). `evaluation/reports/` holds generated output only (gitignored except
`.gitkeep` -- see `.gitignore`), not source, which is why this module lives at the
`evaluation/` top level rather than inside that directory. Every run writes both a
timestamped copy (history) and overwrites `latest.md`/`latest.json` -- a stable
link target for README.md and `docs/experiments/evaluation_notes.md`.
"""

from __future__ import annotations

from pathlib import Path

from evaluation.harness import EvaluationReport

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def write_reports(report: EvaluationReport, out_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    """Write the timestamped + `latest` JSON and Markdown reports. Returns (md, json) paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.strftime("%Y%m%d-%H%M%S")

    payload = report.model_dump_json(indent=2)
    json_path = out_dir / f"{stamp}.json"
    json_path.write_text(payload, encoding="utf-8")
    (out_dir / "latest.json").write_text(payload, encoding="utf-8")

    markdown = render_markdown(report)
    md_path = out_dir / f"{stamp}.md"
    md_path.write_text(markdown, encoding="utf-8")
    (out_dir / "latest.md").write_text(markdown, encoding="utf-8")

    return md_path, json_path


def render_markdown(report: EvaluationReport) -> str:
    """Render `report` as the Markdown summary described in task section 7."""
    lines: list[str] = [
        "# LexRAG Evaluation Report",
        "",
        f"Generated: {report.generated_at.isoformat()}",
        f"Dataset: `{report.dataset_path}` ({report.case_count} cases)",
        "",
        "## Retrieval",
        "",
        "| Strategy | Recall@5 | Recall@10 | Precision@5 | Precision@10 | Cases |",
        "|---|---|---|---|---|---|",
    ]
    for strategy in ("dense", "sparse", "hybrid"):
        r = report.retrieval[strategy]
        lines.append(
            f"| {strategy} | {r.recall_at_5:.2f} | {r.recall_at_10:.2f} | "
            f"{r.precision_at_5:.2f} | {r.precision_at_10:.2f} | {r.case_count} |"
        )

    g = report.generation
    lines += [
        "",
        "## Generation (RAGAS)",
        "",
        f"- Faithfulness: {g.faithfulness:.2f}",
        f"- Context Precision: {g.context_precision:.2f}",
        f"- Context Recall: {g.context_recall:.2f}",
        f"- Answer Relevancy: {g.answer_relevancy:.2f}",
        f"- Scored cases: {g.case_count}",
        "",
        "DeepEval threshold pass rate:",
    ]
    lines += [
        f"- {name}: {rate:.0%}" for name, rate in report.generation_deepeval_pass_rate.items()
    ]

    ref = report.refusal
    lines += [
        "",
        "## Refusal",
        "",
        f"- Accuracy: {ref.accuracy:.2%} ({ref.case_count} cases)",
        f"- False refusals: {ref.false_refusals} {ref.false_refusal_case_ids}",
        f"- False acceptances: {ref.false_acceptances} {ref.false_acceptance_case_ids}",
    ]

    lat = report.latency
    lines += [
        "",
        "## Latency",
        "",
        f"- Avg retrieval: {lat.avg_retrieval_latency_s:.3f}s",
        f"- Avg reranker: {lat.avg_reranker_latency_s:.3f}s",
        f"- Avg generation: {lat.avg_generation_latency_s:.3f}s",
        f"- Avg end-to-end: {lat.avg_end_to_end_latency_s:.3f}s",
        "",
        "## Failures",
        "",
    ]
    if not report.failures:
        lines.append("None.")
    else:
        lines.append("| Case | Stage | Question | Expected | Retrieved |")
        lines.append("|---|---|---|---|---|")
        for f in report.failures:
            expected = "; ".join(f.expected_citations) or "-"
            retrieved = "; ".join(f.retrieved_citations) or "-"
            lines.append(
                f"| {f.case_id} | {f.failure_stage} | {f.question} | {expected} | {retrieved} |"
            )

    return "\n".join(lines) + "\n"
