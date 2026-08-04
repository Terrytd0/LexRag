from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from configs.settings import Settings
from domain.chunk import Chunk
from domain.retrieval import RetrievalResult
from retrieval.hybrid import HybridRetriever


def _result(chunk_id: str, **scores: float | None) -> RetrievalResult:
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
    return RetrievalResult(chunk=chunk, **scores)


@pytest.fixture
def settings() -> Settings:
    return Settings(RRF_K=60)


async def test_retrieve_calls_dense_and_sparse_with_the_same_query_and_top_k(
    settings: Settings,
) -> None:
    dense = MagicMock(name="dense")
    dense.retrieve.return_value = [_result("a", dense_score=0.9)]
    sparse = MagicMock(name="sparse")
    sparse.retrieve.return_value = [_result("b", sparse_score=5.0)]

    retriever = HybridRetriever(dense, sparse, settings=settings)
    await retriever.retrieve("query", top_k=10)

    dense.retrieve.assert_called_once_with("query", 10, None)
    sparse.retrieve.assert_called_once_with("query", 10, None)


async def test_retrieve_passes_document_ids_through_to_dense_and_sparse(
    settings: Settings,
) -> None:
    dense = MagicMock(name="dense")
    dense.retrieve.return_value = []
    sparse = MagicMock(name="sparse")
    sparse.retrieve.return_value = []

    retriever = HybridRetriever(dense, sparse, settings=settings)
    await retriever.retrieve("query", document_ids=["doc-1", "doc-2"])

    dense.retrieve.assert_called_once_with("query", None, ["doc-1", "doc-2"])
    sparse.retrieve.assert_called_once_with("query", None, ["doc-1", "doc-2"])


async def test_retrieve_returns_rrf_merged_results(settings: Settings) -> None:
    dense = MagicMock(name="dense")
    dense.retrieve.return_value = [_result("a", dense_score=0.9), _result("b", dense_score=0.8)]
    sparse = MagicMock(name="sparse")
    sparse.retrieve.return_value = [_result("b", sparse_score=9.0)]

    retriever = HybridRetriever(dense, sparse, settings=settings)
    results = await retriever.retrieve("query")

    by_id = {r.chunk.chunk_id: r for r in results}
    assert set(by_id) == {"a", "b"}
    assert by_id["b"].dense_score == 0.8
    assert by_id["b"].sparse_score == 9.0
    assert all(r.rrf_score is not None for r in results)
    # 'b' is rank 1 in sparse and rank 2 in dense; 'a' is only rank 1 in dense --
    # b's combined contribution should outrank a's single-list contribution.
    assert results[0].chunk.chunk_id == "b"


async def test_retrieve_handles_empty_result_sets(settings: Settings) -> None:
    dense = MagicMock(name="dense")
    dense.retrieve.return_value = []
    sparse = MagicMock(name="sparse")
    sparse.retrieve.return_value = []

    retriever = HybridRetriever(dense, sparse, settings=settings)
    results = await retriever.retrieve("query")

    assert results == []


async def test_retrieve_logs_dense_sparse_and_hybrid_counts_without_query_text(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    dense = MagicMock(name="dense")
    dense.retrieve.return_value = [_result("a", dense_score=0.9)]
    sparse = MagicMock(name="sparse")
    sparse.retrieve.return_value = [_result("a", sparse_score=1.0), _result("b", sparse_score=0.5)]

    retriever = HybridRetriever(dense, sparse, settings=settings)
    with caplog.at_level(logging.INFO, logger="retrieval.hybrid"):
        await retriever.retrieve("privileged legal query text")

    messages = [record.getMessage() for record in caplog.records]
    joined = "\n".join(messages)
    assert any("dense_count=1" in m for m in messages)
    assert any("sparse_count=2" in m for m in messages)
    assert any("hybrid_count=2" in m for m in messages)
    assert "privileged legal query text" not in joined
