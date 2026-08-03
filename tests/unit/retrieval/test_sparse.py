from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from configs.settings import Settings
from domain.chunk import Chunk
from retrieval.sparse import SparseRetriever


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        chunk_index=0,
        text="termination for convenience, Section 8.3",
        token_count=6,
        page_number=1,
        section="unspecified",
        source_filename="contract.pdf",
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(RETRIEVAL_TOP_K=5)


def test_retrieve_wraps_bm25_matches_as_retrieval_results(settings: Settings) -> None:
    keyword_store = MagicMock(name="keyword_store")
    keyword_store.search.return_value = [(_chunk("doc-1:0"), 12.4), (_chunk("doc-1:1"), 3.1)]

    retriever = SparseRetriever(keyword_store, settings=settings)
    results = retriever.retrieve("Section 8.3")

    keyword_store.search.assert_called_once_with("Section 8.3", 5)
    assert [r.sparse_score for r in results] == [12.4, 3.1]
    assert [r.dense_score for r in results] == [None, None]
    assert [r.chunk.chunk_id for r in results] == ["doc-1:0", "doc-1:1"]


def test_retrieve_falls_back_to_settings_top_k_when_not_given(settings: Settings) -> None:
    keyword_store = MagicMock(name="keyword_store")
    keyword_store.search.return_value = []

    SparseRetriever(keyword_store, settings=settings).retrieve("query")

    keyword_store.search.assert_called_once_with("query", 5)


def test_retrieve_uses_explicit_top_k_over_settings_default(settings: Settings) -> None:
    keyword_store = MagicMock(name="keyword_store")
    keyword_store.search.return_value = []

    SparseRetriever(keyword_store, settings=settings).retrieve("query", top_k=2)

    keyword_store.search.assert_called_once_with("query", 2)


def test_retrieve_preserves_chunk_metadata(settings: Settings) -> None:
    chunk = _chunk("doc-1:0")
    keyword_store = MagicMock(name="keyword_store")
    keyword_store.search.return_value = [(chunk, 12.4)]

    results = SparseRetriever(keyword_store, settings=settings).retrieve("query")

    assert results[0].chunk == chunk
