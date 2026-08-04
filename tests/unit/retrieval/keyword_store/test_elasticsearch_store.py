from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from configs.settings import Settings
from domain.chunk import Chunk
from retrieval.keyword_store.elasticsearch_store import ElasticsearchKeywordStore


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
    return Settings(ELASTICSEARCH_INDEX="test_index")


@pytest.fixture
def client() -> MagicMock:
    client = MagicMock(name="es_client")
    client.indices.exists.return_value = False
    return client


@pytest.fixture(autouse=True)
def mock_bulk(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock(name="bulk")
    monkeypatch.setattr("retrieval.keyword_store.elasticsearch_store.bulk", mock)
    return mock


def test_index_chunks_creates_index_with_bm25_mapping_when_missing(
    client: MagicMock, settings: Settings
) -> None:
    store = ElasticsearchKeywordStore(client, settings=settings)

    store.index_chunks([_chunk("doc-1:0")])

    client.indices.create.assert_called_once()
    kwargs = client.indices.create.call_args.kwargs
    assert kwargs["index"] == "test_index"
    assert "legal_text_analyzer" in kwargs["settings"]["analysis"]["analyzer"]
    assert kwargs["mappings"]["properties"]["text"]["similarity"] == "BM25"


def test_index_chunks_skips_creation_when_index_exists(
    client: MagicMock, settings: Settings
) -> None:
    client.indices.exists.return_value = True
    store = ElasticsearchKeywordStore(client, settings=settings)

    store.index_chunks([_chunk("doc-1:0")])

    client.indices.create.assert_not_called()


def test_index_chunks_ensures_index_only_once_across_calls(
    client: MagicMock, settings: Settings
) -> None:
    store = ElasticsearchKeywordStore(client, settings=settings)

    store.index_chunks([_chunk("doc-1:0")])
    store.index_chunks([_chunk("doc-1:1")])

    client.indices.create.assert_called_once()


def test_index_chunks_bulk_indexes_by_chunk_id_preserving_metadata(
    client: MagicMock, settings: Settings, mock_bulk: MagicMock
) -> None:
    chunk = _chunk("doc-1:0")
    store = ElasticsearchKeywordStore(client, settings=settings)

    store.index_chunks([chunk])

    mock_bulk.assert_called_once()
    call_client, actions = mock_bulk.call_args.args
    assert call_client is client
    assert actions == [
        {"_index": "test_index", "_id": "doc-1:0", "_source": chunk.model_dump(mode="json")}
    ]


def test_index_chunks_empty_list_is_a_noop(
    client: MagicMock, settings: Settings, mock_bulk: MagicMock
) -> None:
    store = ElasticsearchKeywordStore(client, settings=settings)

    store.index_chunks([])

    mock_bulk.assert_not_called()
    client.indices.create.assert_not_called()


def test_search_returns_chunks_with_bm25_scores(client: MagicMock, settings: Settings) -> None:
    client.indices.exists.return_value = True
    chunk = _chunk("doc-1:0")
    client.search.return_value = {
        "hits": {"hits": [{"_source": chunk.model_dump(mode="json"), "_score": 4.2}]}
    }

    store = ElasticsearchKeywordStore(client, settings=settings)
    results = store.search("termination clause", top_k=5)

    assert results == [(chunk, 4.2)]
    client.search.assert_called_once_with(
        index="test_index", query={"match": {"text": "termination clause"}}, size=5
    )


def test_delete_document_deletes_by_doc_id_term_query(
    client: MagicMock, settings: Settings
) -> None:
    client.indices.exists.return_value = True
    client.delete_by_query.return_value = {"deleted": 3}
    store = ElasticsearchKeywordStore(client, settings=settings)

    store.delete_document("doc-1")

    client.delete_by_query.assert_called_once_with(
        index="test_index", query={"term": {"doc_id": "doc-1"}}
    )


def test_search_returns_empty_list_when_no_hits(client: MagicMock, settings: Settings) -> None:
    client.indices.exists.return_value = True
    client.search.return_value = {"hits": {"hits": []}}

    store = ElasticsearchKeywordStore(client, settings=settings)

    assert store.search("no matches", top_k=5) == []


def test_search_with_document_ids_adds_a_bool_terms_filter(
    client: MagicMock, settings: Settings
) -> None:
    client.indices.exists.return_value = True
    client.search.return_value = {"hits": {"hits": []}}

    store = ElasticsearchKeywordStore(client, settings=settings)
    store.search("termination clause", top_k=5, document_ids=["doc-1", "doc-2"])

    client.search.assert_called_once_with(
        index="test_index",
        query={
            "bool": {
                "must": [{"match": {"text": "termination clause"}}],
                "filter": [{"terms": {"doc_id": ["doc-1", "doc-2"]}}],
            }
        },
        size=5,
    )


def test_search_with_empty_document_ids_uses_unfiltered_query(
    client: MagicMock, settings: Settings
) -> None:
    client.indices.exists.return_value = True
    client.search.return_value = {"hits": {"hits": []}}

    store = ElasticsearchKeywordStore(client, settings=settings)
    store.search("termination clause", top_k=5, document_ids=[])

    client.search.assert_called_once_with(
        index="test_index", query={"match": {"text": "termination clause"}}, size=5
    )
