from __future__ import annotations

import pytest

from domain.chunk import Chunk
from domain.retrieval import RetrievalResult
from retrieval.fusion.rrf import reciprocal_rank_fusion


def _result(
    chunk_id: str, *, dense_score: float | None = None, sparse_score: float | None = None
) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        chunk_index=0,
        text="text",
        token_count=1,
        page_number=1,
        section="unspecified",
        source_filename="contract.pdf",
    )
    return RetrievalResult(chunk=chunk, dense_score=dense_score, sparse_score=sparse_score)


def test_fuses_disjoint_lists_preserving_each_strategys_score() -> None:
    dense = [_result("a", dense_score=0.9), _result("b", dense_score=0.8)]
    sparse = [_result("c", sparse_score=12.0)]

    merged = reciprocal_rank_fusion(dense, sparse, k=60)

    by_id = {r.chunk.chunk_id: r for r in merged}
    assert by_id["a"].dense_score == 0.9
    assert by_id["a"].sparse_score is None
    assert by_id["c"].sparse_score == 12.0
    assert by_id["c"].dense_score is None
    assert by_id["a"].rrf_score == pytest.approx(1 / 61)
    assert by_id["c"].rrf_score == pytest.approx(1 / 61)


def test_duplicate_chunk_merges_both_scores_and_sums_rank_contributions() -> None:
    dense = [_result("a", dense_score=0.9)]  # rank 1 in dense
    sparse = [_result("x", sparse_score=5.0), _result("a", sparse_score=3.0)]  # rank 2 in sparse

    merged = reciprocal_rank_fusion(dense, sparse, k=60)

    a = next(r for r in merged if r.chunk.chunk_id == "a")
    assert a.dense_score == 0.9
    assert a.sparse_score == 3.0
    assert a.rrf_score == pytest.approx(1 / 61 + 1 / 62)


def test_results_ranked_by_position_not_by_raw_score_value() -> None:
    # 'a' outranks 'b' in the dense list (rank 1 vs rank 2) despite a lower raw score --
    # RRF only consumes rank order, never the underlying score.
    dense = [_result("a", dense_score=0.5), _result("b", dense_score=0.9)]

    merged = reciprocal_rank_fusion(dense, [], k=60)

    assert [r.chunk.chunk_id for r in merged] == ["a", "b"]


def test_deterministic_tiebreak_by_chunk_id_on_equal_rrf_score() -> None:
    dense = [_result("z", dense_score=1.0)]
    sparse = [_result("a", sparse_score=1.0)]

    merged = reciprocal_rank_fusion(dense, sparse, k=60)

    assert [r.chunk.chunk_id for r in merged] == ["a", "z"]


def test_repeated_fusion_of_same_inputs_yields_identical_order() -> None:
    dense = [_result("a", dense_score=0.9), _result("b", dense_score=0.8)]
    sparse = [_result("b", sparse_score=9.0), _result("c", sparse_score=1.0)]

    first = reciprocal_rank_fusion(dense, sparse, k=60)
    second = reciprocal_rank_fusion(dense, sparse, k=60)

    assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in second]


def test_empty_inputs_return_empty_list() -> None:
    assert reciprocal_rank_fusion([], [], k=60) == []


def test_smaller_k_weights_top_ranks_more_heavily() -> None:
    dense = [_result("a", dense_score=0.9), _result("b", dense_score=0.8)]

    small_k = reciprocal_rank_fusion(dense, [], k=1)
    large_k = reciprocal_rank_fusion(dense, [], k=1000)

    top_id = next(r for r in small_k if r.chunk.chunk_id == "a")
    other_id = next(r for r in small_k if r.chunk.chunk_id == "b")
    assert top_id.rrf_score is not None
    assert other_id.rrf_score is not None
    small_k_gap = top_id.rrf_score - other_id.rrf_score

    top_large = next(r for r in large_k if r.chunk.chunk_id == "a")
    other_large = next(r for r in large_k if r.chunk.chunk_id == "b")
    assert top_large.rrf_score is not None
    assert other_large.rrf_score is not None
    large_k_gap = top_large.rrf_score - other_large.rrf_score

    assert small_k_gap > large_k_gap
