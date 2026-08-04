from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from domain.chunk import Chunk
from domain.retrieval import RetrievalResult
from retrieval.reranker.cross_encoder import CrossEncoderReranker, get_reranker


class _FakeModel:
    """Stand-in for CrossEncoder: returns a caller-supplied score per pair, in order."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.predict_calls: list[dict[str, Any]] = []

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> np.ndarray:
        self.predict_calls.append({"pairs": pairs, **kwargs})
        return np.array(self._scores[: len(pairs)])


@pytest.fixture
def fake_model_factory(monkeypatch: pytest.MonkeyPatch) -> Any:
    def _install(scores: list[float]) -> _FakeModel:
        model = _FakeModel(scores)
        monkeypatch.setattr(
            "retrieval.reranker.cross_encoder.CrossEncoder", MagicMock(return_value=model)
        )
        return model

    return _install


def _result(chunk_id: str) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        chunk_index=0,
        text=f"text for {chunk_id}",
        token_count=3,
        page_number=1,
        section="unspecified",
        source_filename="contract.pdf",
    )
    return RetrievalResult(chunk=chunk, dense_score=0.5, sparse_score=1.0, rrf_score=0.01)


def test_rerank_orders_by_descending_score(fake_model_factory: Any) -> None:
    fake_model_factory([0.2, 0.9, 0.5])
    reranker = CrossEncoderReranker(model_name="fake-reranker", top_k=10)
    results = [_result("a"), _result("b"), _result("c")]

    reranked = reranker.rerank("query", results)

    assert [r.chunk.chunk_id for r in reranked] == ["b", "c", "a"]
    assert [r.rerank_score for r in reranked] == [0.9, 0.5, 0.2]


def test_rerank_ties_break_deterministically_by_chunk_id(fake_model_factory: Any) -> None:
    fake_model_factory([0.5, 0.5, 0.5])
    reranker = CrossEncoderReranker(model_name="fake-reranker", top_k=10)
    results = [_result("c"), _result("a"), _result("b")]

    reranked = reranker.rerank("query", results)

    assert [r.chunk.chunk_id for r in reranked] == ["a", "b", "c"]


def test_rerank_preserves_retrieval_result_metadata(fake_model_factory: Any) -> None:
    fake_model_factory([0.7])
    reranker = CrossEncoderReranker(model_name="fake-reranker", top_k=10)
    original = _result("a")

    reranked = reranker.rerank("query", [original])

    result = reranked[0]
    assert result.chunk == original.chunk
    assert result.dense_score == original.dense_score
    assert result.sparse_score == original.sparse_score
    assert result.rrf_score == original.rrf_score
    assert result.rerank_score == 0.7


def test_rerank_truncates_to_top_k(fake_model_factory: Any) -> None:
    fake_model_factory([0.1, 0.9, 0.5, 0.3])
    reranker = CrossEncoderReranker(model_name="fake-reranker", top_k=2)
    results = [_result("a"), _result("b"), _result("c"), _result("d")]

    reranked = reranker.rerank("query", results)

    assert [r.chunk.chunk_id for r in reranked] == ["b", "c"]


def test_rerank_top_k_override_takes_precedence(fake_model_factory: Any) -> None:
    fake_model_factory([0.1, 0.9, 0.5])
    reranker = CrossEncoderReranker(model_name="fake-reranker", top_k=10)
    results = [_result("a"), _result("b"), _result("c")]

    reranked = reranker.rerank("query", results, top_k=1)

    assert [r.chunk.chunk_id for r in reranked] == ["b"]


def test_rerank_empty_results_short_circuits(fake_model_factory: Any) -> None:
    model = fake_model_factory([])
    reranker = CrossEncoderReranker(model_name="fake-reranker", top_k=10)

    assert reranker.rerank("query", []) == []
    assert model.predict_calls == []


def test_rerank_passes_query_chunk_pairs_and_batch_size(fake_model_factory: Any) -> None:
    model = fake_model_factory([0.4, 0.6])
    reranker = CrossEncoderReranker(model_name="fake-reranker", top_k=10, batch_size=4)
    results = [_result("a"), _result("b")]

    reranker.rerank("my query", results)

    call = model.predict_calls[0]
    assert call["pairs"] == [("my query", "text for a"), ("my query", "text for b")]
    assert call["batch_size"] == 4


def test_rerank_is_deterministic_across_calls(fake_model_factory: Any) -> None:
    fake_model_factory([0.3, 0.8])
    reranker = CrossEncoderReranker(model_name="fake-reranker", top_k=10)
    results = [_result("a"), _result("b")]

    first = [r.chunk.chunk_id for r in reranker.rerank("query", results)]
    second = [r.chunk.chunk_id for r in reranker.rerank("query", results)]

    assert first == second


def test_rerank_logs_latency_and_counts_without_query_text(
    fake_model_factory: Any, caplog: pytest.LogCaptureFixture
) -> None:
    fake_model_factory([0.5])
    reranker = CrossEncoderReranker(model_name="fake-reranker", top_k=10)

    with caplog.at_level(logging.INFO, logger="retrieval.reranker.cross_encoder"):
        reranker.rerank("privileged legal query text", [_result("a")])

    messages = [record.getMessage() for record in caplog.records]
    joined = "\n".join(messages)
    assert any("candidate_count=1" in m for m in messages)
    assert any("result_count=1" in m for m in messages)
    assert "privileged legal query text" not in joined


def test_get_reranker_returns_a_cached_singleton(fake_model_factory: Any) -> None:
    fake_model_factory([0.5])
    get_reranker.cache_clear()
    try:
        assert get_reranker() is get_reranker()
    finally:
        get_reranker.cache_clear()
