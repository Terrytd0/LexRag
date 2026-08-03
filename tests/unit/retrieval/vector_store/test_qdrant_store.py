from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qdrant_client import models

from configs.settings import Settings
from domain.chunk import Chunk
from retrieval.vector_store.qdrant_store import QdrantVectorStore, _point_id


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        chunk_index=0,
        text="hello world",
        token_count=2,
        page_number=1,
        section="unspecified",
        source_filename="contract.pdf",
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        QDRANT_COLLECTION="test_collection",
        EMBEDDING_DIMENSIONS=3,
        QDRANT_DISTANCE_METRIC="Cosine",
    )


@pytest.fixture
def client() -> MagicMock:
    client = MagicMock(name="qdrant_client")
    client.collection_exists.return_value = False
    return client


@pytest.fixture
def embedding_service() -> MagicMock:
    service = MagicMock(name="embedding_service")
    service.embed_texts.return_value = [[0.1, 0.2, 0.3]]
    return service


def test_index_chunks_creates_collection_when_missing(
    client: MagicMock, embedding_service: MagicMock, settings: Settings
) -> None:
    store = QdrantVectorStore(client, embedding_service, settings=settings)

    store.index_chunks([_chunk("doc-1:0")])

    client.create_collection.assert_called_once()
    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "test_collection"
    assert kwargs["vectors_config"].size == 3
    assert kwargs["vectors_config"].distance == models.Distance.COSINE


def test_index_chunks_skips_creation_when_collection_exists(
    client: MagicMock, embedding_service: MagicMock, settings: Settings
) -> None:
    client.collection_exists.return_value = True
    store = QdrantVectorStore(client, embedding_service, settings=settings)

    store.index_chunks([_chunk("doc-1:0")])

    client.create_collection.assert_not_called()


def test_index_chunks_ensures_collection_only_once_across_calls(
    client: MagicMock, embedding_service: MagicMock, settings: Settings
) -> None:
    store = QdrantVectorStore(client, embedding_service, settings=settings)

    store.index_chunks([_chunk("doc-1:0")])
    store.index_chunks([_chunk("doc-1:1")])

    client.create_collection.assert_called_once()


def test_index_chunks_upserts_points_with_full_chunk_payload(
    client: MagicMock, embedding_service: MagicMock, settings: Settings
) -> None:
    chunk = _chunk("doc-1:0")
    store = QdrantVectorStore(client, embedding_service, settings=settings)

    store.index_chunks([chunk])

    client.upsert.assert_called_once()
    kwargs = client.upsert.call_args.kwargs
    assert kwargs["collection_name"] == "test_collection"
    points = kwargs["points"]
    assert len(points) == 1
    assert points[0].id == _point_id("doc-1:0")
    assert points[0].vector == [0.1, 0.2, 0.3]
    assert points[0].payload == chunk.model_dump(mode="json")


def test_index_chunks_embeds_chunk_text_not_metadata(
    client: MagicMock, embedding_service: MagicMock, settings: Settings
) -> None:
    chunk = _chunk("doc-1:0")
    store = QdrantVectorStore(client, embedding_service, settings=settings)

    store.index_chunks([chunk])

    embedding_service.embed_texts.assert_called_once_with(["hello world"])


def test_index_chunks_empty_list_is_a_noop(
    client: MagicMock, embedding_service: MagicMock, settings: Settings
) -> None:
    store = QdrantVectorStore(client, embedding_service, settings=settings)

    store.index_chunks([])

    client.upsert.assert_not_called()
    embedding_service.embed_texts.assert_not_called()
    client.create_collection.assert_not_called()


def test_search_returns_chunks_reconstructed_from_payload_with_scores(
    client: MagicMock, embedding_service: MagicMock, settings: Settings
) -> None:
    client.collection_exists.return_value = True
    chunk = _chunk("doc-1:0")
    point = MagicMock()
    point.payload = chunk.model_dump(mode="json")
    point.score = 0.87
    response = MagicMock()
    response.points = [point]
    client.query_points.return_value = response

    store = QdrantVectorStore(client, embedding_service, settings=settings)
    results = store.search([0.1, 0.2, 0.3], top_k=5)

    assert results == [(chunk, 0.87)]
    client.query_points.assert_called_once_with(
        collection_name="test_collection", query=[0.1, 0.2, 0.3], limit=5
    )


def test_point_id_is_deterministic_and_unique_per_chunk_id() -> None:
    assert _point_id("doc-1:0") == _point_id("doc-1:0")
    assert _point_id("doc-1:0") != _point_id("doc-1:1")
