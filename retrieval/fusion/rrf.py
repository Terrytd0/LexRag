"""Reciprocal Rank Fusion merging vector and BM25 ranked result lists.

`score(d) = sum over strategies i of 1 / (k + rank_i(d))` -- see
`docs/architecture.md` §2.6 for why RRF (rank-based, no cross-strategy score
normalization) was chosen over weighted score fusion.
"""

from __future__ import annotations

from domain.retrieval import RetrievalResult

# Mirrors `Settings.rrf_k`'s default (60, the RRF paper's commonly cited
# constant) so this function is directly usable/testable without wiring
# through settings, while `HybridRetriever` always passes the configured value.
_DEFAULT_K = 60


def reciprocal_rank_fusion(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    k: int = _DEFAULT_K,
) -> list[RetrievalResult]:
    """Merge two ranked result lists into one, ordered by combined RRF score.

    Each input list must already be ranked best-first by its own retrieval
    stage; rank is derived from list position (1-indexed), not from any score
    field. A chunk present in both lists is merged into a single
    `RetrievalResult` carrying both `dense_score` and `sparse_score`, with its
    rank contribution summed across strategies. Output is sorted by
    `rrf_score` descending, with `chunk_id` as a deterministic tiebreak so
    fusing the same two inputs always yields the same order.
    """
    merged: dict[str, RetrievalResult] = {}

    for results in (dense_results, sparse_results):
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id
            contribution = 1.0 / (k + rank)
            existing = merged.get(chunk_id)
            if existing is None:
                merged[chunk_id] = result.model_copy(update={"rrf_score": contribution})
                continue
            merged[chunk_id] = existing.model_copy(
                update={
                    "dense_score": (
                        existing.dense_score
                        if existing.dense_score is not None
                        else result.dense_score
                    ),
                    "sparse_score": (
                        existing.sparse_score
                        if existing.sparse_score is not None
                        else result.sparse_score
                    ),
                    "rrf_score": (existing.rrf_score or 0.0) + contribution,
                }
            )

    return sorted(merged.values(), key=lambda r: (-(r.rrf_score or 0.0), r.chunk.chunk_id))
