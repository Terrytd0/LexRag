"""Canonical RetrievalResult model: one candidate chunk as it moves through retrieval."""

from __future__ import annotations

from pydantic import BaseModel

from domain.chunk import Chunk


class RetrievalResult(BaseModel):
    """A candidate chunk plus the scores assigned to it at each retrieval stage.

    Populated incrementally by the Day 3-4 retrieval pipeline: `dense_score`/
    `sparse_score` after the parallel Qdrant/Elasticsearch search, `rrf_score`
    after fusion, `rerank_score` after the cross-encoder pass. All score
    fields are optional so a stage can construct or update a result without
    the later stages' data yet existing.
    """

    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
