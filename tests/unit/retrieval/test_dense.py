from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from configs.settings import Settings
from domain.chunk import Chunk
from retrieval.dense import DenseRetriever


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        chunk_index=0,
        text="indemnification clause text",
        token_count=3,
        page_number=1,
        section="unspecified",
        source_filename="contract.pdf",
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(RETRIEVAL_TOP_K=5)


def test_retrieve_embeds_query_and_wraps_matches_as_retrieval_results(settings: Settings) -> None:
    vector_store = MagicMock(name="vector_store")
    vector_store.search.return_value = [(_chunk("doc-1:0"), 0.9), (_chunk("doc-1:1"), 0.5)]
    embedding_service = MagicMock(name="embedding_service")
    embedding_service.embed_query.return_value = [0.1, 0.2]

    retriever = DenseRetriever(vector_store, embedding_service, settings=settings)
    results = retriever.retrieve("indemnification clause")

    embedding_service.embed_query.assert_called_once_with("indemnification clause")
    vector_store.search.assert_called_once_with([0.1, 0.2], 5, document_ids=None)
    assert [r.dense_score for r in results] == [0.9, 0.5]
    assert [r.sparse_score for r in results] == [None, None]
    assert [r.chunk.chunk_id for r in results] == ["doc-1:0", "doc-1:1"]


def test_retrieve_falls_back_to_settings_top_k_when_not_given(settings: Settings) -> None:
    vector_store = MagicMock(name="vector_store")
    vector_store.search.return_value = []
    embedding_service = MagicMock(name="embedding_service")
    embedding_service.embed_query.return_value = [0.1]

    DenseRetriever(vector_store, embedding_service, settings=settings).retrieve("query")

    vector_store.search.assert_called_once_with([0.1], 5, document_ids=None)


def test_retrieve_uses_explicit_top_k_over_settings_default(settings: Settings) -> None:
    vector_store = MagicMock(name="vector_store")
    vector_store.search.return_value = []
    embedding_service = MagicMock(name="embedding_service")
    embedding_service.embed_query.return_value = [0.1]

    DenseRetriever(vector_store, embedding_service, settings=settings).retrieve("query", top_k=2)

    vector_store.search.assert_called_once_with([0.1], 2, document_ids=None)


def test_retrieve_preserves_chunk_metadata(settings: Settings) -> None:
    chunk = _chunk("doc-1:0")
    vector_store = MagicMock(name="vector_store")
    vector_store.search.return_value = [(chunk, 0.9)]
    embedding_service = MagicMock(name="embedding_service")
    embedding_service.embed_query.return_value = [0.1]

    results = DenseRetriever(vector_store, embedding_service, settings=settings).retrieve("query")

    assert results[0].chunk == chunk


def test_retrieve_passes_document_ids_through_to_vector_store(settings: Settings) -> None:
    vector_store = MagicMock(name="vector_store")
    vector_store.search.return_value = []
    embedding_service = MagicMock(name="embedding_service")
    embedding_service.embed_query.return_value = [0.1]

    DenseRetriever(vector_store, embedding_service, settings=settings).retrieve(
        "query", document_ids=["doc-1", "doc-2"]
    )

    vector_store.search.assert_called_once_with([0.1], 5, document_ids=["doc-1", "doc-2"])
