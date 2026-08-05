from __future__ import annotations

from domain.chunk import Chunk
from domain.retrieval import RetrievalResult
from evaluation.metrics.retrieval import (
    aggregate_retrieval_metrics,
    relevant_doc_ids,
    score_retrieval_case,
)


def _result(doc_id: str, index: int) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=f"{doc_id}:{index}",
        doc_id=doc_id,
        chunk_index=index,
        text="text",
        token_count=1,
        page_number=1,
        section="unspecified",
        source_filename=f"{doc_id}.pdf",
    )
    return RetrievalResult(chunk=chunk)


def test_relevant_doc_ids_resolves_filenames_to_live_doc_ids() -> None:
    mapping = {"a.pdf": "doc-a", "b.pdf": "doc-b"}

    assert relevant_doc_ids(["a.pdf", "b.pdf"], mapping) == {"doc-a", "doc-b"}


def test_score_retrieval_case_perfect_hit_at_rank_one() -> None:
    results = [_result("doc-a", 0)] + [_result(f"doc-x{i}", i) for i in range(4)]

    metrics = score_retrieval_case("case-1", "hybrid", results, {"doc-a"})

    assert metrics.recall_at_5 == 1.0
    assert metrics.recall_at_10 == 1.0
    assert metrics.precision_at_5 == 0.2  # 1 relevant out of the 5 results retrieved


def test_score_retrieval_case_precision_divides_by_results_available_below_k() -> None:
    # Fewer than 5 results were retrieved -- precision divides by what's actually
    # there (2), not by the cutoff k (5), since RETRIEVAL_TOP_K normally guarantees
    # k results are available and this is only an edge case in tests.
    results = [_result("doc-a", 0), _result("doc-x", 1)]

    metrics = score_retrieval_case("case-1", "hybrid", results, {"doc-a"})

    assert metrics.precision_at_5 == 0.5


def test_score_retrieval_case_no_hit_within_top_10() -> None:
    results = [_result(f"doc-x{i}", i) for i in range(10)]

    metrics = score_retrieval_case("case-1", "dense", results, {"doc-a"})

    assert metrics.recall_at_5 == 0.0
    assert metrics.recall_at_10 == 0.0
    assert metrics.precision_at_10 == 0.0


def test_score_retrieval_case_hit_only_within_top_10_not_top_5() -> None:
    results = [_result(f"doc-x{i}", i) for i in range(7)] + [_result("doc-a", 7)]

    metrics = score_retrieval_case("case-1", "hybrid", results, {"doc-a"})

    assert metrics.recall_at_5 == 0.0
    assert metrics.recall_at_10 == 1.0


def test_score_retrieval_case_empty_relevant_set_is_zero_not_a_crash() -> None:
    results = [_result("doc-a", 0)]

    metrics = score_retrieval_case("case-1", "hybrid", results, set())

    assert metrics.recall_at_5 == 0.0
    assert metrics.recall_at_10 == 0.0


def test_score_retrieval_case_empty_results_is_zero_not_a_crash() -> None:
    metrics = score_retrieval_case("case-1", "hybrid", [], {"doc-a"})

    assert metrics.recall_at_5 == 0.0
    assert metrics.precision_at_5 == 0.0


def test_aggregate_retrieval_metrics_averages_across_cases() -> None:
    perfect = score_retrieval_case("case-1", "hybrid", [_result("doc-a", 0)], {"doc-a"})
    miss = score_retrieval_case("case-2", "hybrid", [_result("doc-x", 0)], {"doc-a"})

    report = aggregate_retrieval_metrics("hybrid", [perfect, miss])

    assert report.recall_at_5 == 0.5
    assert report.case_count == 2


def test_aggregate_retrieval_metrics_empty_is_all_zero() -> None:
    report = aggregate_retrieval_metrics("hybrid", [])

    assert report.recall_at_5 == 0.0
    assert report.case_count == 0
