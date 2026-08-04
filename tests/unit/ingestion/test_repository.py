from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from configs.settings import Settings
from domain.chunk import Chunk
from domain.document import Document
from domain.enums import DocumentStatus
from ingestion.repository import _EXCLUDE_MONGO_ID, DocumentRepository


@pytest.fixture
def settings() -> Settings:
    return Settings(MONGODB_DB_NAME="lexrag_test")


@pytest.fixture
def mock_client() -> MagicMock:
    """A MongoClient stand-in: client[db][collection] resolves to plain mocks."""
    documents_collection = MagicMock(name="documents_collection")
    chunks_collection = MagicMock(name="chunks_collection")
    collections = {"documents": documents_collection, "chunks": chunks_collection}

    db = MagicMock(name="db")
    db.__getitem__.side_effect = collections.__getitem__

    client = MagicMock(name="client")
    client.__getitem__.return_value = db
    return client


def _collection(client: MagicMock, name: str) -> MagicMock:
    return client.__getitem__.return_value.__getitem__(name)  # type: ignore[no-any-return]


def _document() -> Document:
    return Document(doc_id="doc-1", filename="contract.pdf", status=DocumentStatus.PROCESSING)


def _chunk(index: int) -> Chunk:
    return Chunk(
        chunk_id=f"doc-1:{index}",
        doc_id="doc-1",
        chunk_index=index,
        text=f"chunk {index}",
        token_count=2,
        page_number=1,
        section="unspecified",
        source_filename="contract.pdf",
    )


def test_save_document_upserts_by_doc_id(mock_client: MagicMock, settings: Settings) -> None:
    repository = DocumentRepository(mock_client, settings)
    document = _document()

    repository.save_document(document)

    documents_collection = _collection(mock_client, "documents")
    documents_collection.replace_one.assert_called_once_with(
        {"doc_id": "doc-1"}, document.model_dump(mode="json"), upsert=True
    )


def test_save_chunks_inserts_all_chunks(mock_client: MagicMock, settings: Settings) -> None:
    repository = DocumentRepository(mock_client, settings)
    chunks = [_chunk(0), _chunk(1)]

    repository.save_chunks(chunks)

    chunks_collection = _collection(mock_client, "chunks")
    chunks_collection.insert_many.assert_called_once_with(
        [chunk.model_dump(mode="json") for chunk in chunks]
    )


def test_save_chunks_empty_list_is_a_noop(mock_client: MagicMock, settings: Settings) -> None:
    repository = DocumentRepository(mock_client, settings)

    repository.save_chunks([])

    _collection(mock_client, "chunks").insert_many.assert_not_called()


def test_get_document_returns_none_when_missing(mock_client: MagicMock, settings: Settings) -> None:
    _collection(mock_client, "documents").find_one.return_value = None
    repository = DocumentRepository(mock_client, settings)

    assert repository.get_document("missing") is None


def test_get_document_parses_stored_document(mock_client: MagicMock, settings: Settings) -> None:
    document = _document()
    _collection(mock_client, "documents").find_one.return_value = document.model_dump(mode="json")
    repository = DocumentRepository(mock_client, settings)

    result = repository.get_document("doc-1")

    assert result == document


def test_get_document_by_hash_returns_none_when_missing(
    mock_client: MagicMock, settings: Settings
) -> None:
    _collection(mock_client, "documents").find_one.return_value = None
    repository = DocumentRepository(mock_client, settings)

    assert repository.get_document_by_hash("abc123") is None


def test_get_document_by_hash_parses_stored_document(
    mock_client: MagicMock, settings: Settings
) -> None:
    document = _document()
    _collection(mock_client, "documents").find_one.return_value = document.model_dump(mode="json")
    repository = DocumentRepository(mock_client, settings)

    result = repository.get_document_by_hash("abc123")

    assert result == document
    _collection(mock_client, "documents").find_one.assert_called_once_with(
        {"content_hash": "abc123"}, _EXCLUDE_MONGO_ID
    )


def test_list_documents_returns_all_sorted_by_upload_timestamp_descending(
    mock_client: MagicMock, settings: Settings
) -> None:
    document = _document()
    find_result = MagicMock()
    find_result.sort.return_value = [document.model_dump(mode="json")]
    _collection(mock_client, "documents").find.return_value = find_result
    repository = DocumentRepository(mock_client, settings)

    result = repository.list_documents()

    assert result == [document]
    find_result.sort.assert_called_once_with("upload_timestamp", -1)


def test_list_documents_empty(mock_client: MagicMock, settings: Settings) -> None:
    find_result = MagicMock()
    find_result.sort.return_value = []
    _collection(mock_client, "documents").find.return_value = find_result
    repository = DocumentRepository(mock_client, settings)

    assert repository.list_documents() == []


def test_delete_document_deletes_from_both_collections(
    mock_client: MagicMock, settings: Settings
) -> None:
    repository = DocumentRepository(mock_client, settings)

    repository.delete_document("doc-1")

    _collection(mock_client, "documents").delete_one.assert_called_once_with({"doc_id": "doc-1"})
    _collection(mock_client, "chunks").delete_many.assert_called_once_with({"doc_id": "doc-1"})


def test_get_chunks_returns_chunks_sorted_by_index(
    mock_client: MagicMock, settings: Settings
) -> None:
    stored: list[dict[str, Any]] = [c.model_dump(mode="json") for c in [_chunk(0), _chunk(1)]]
    find_result = MagicMock()
    find_result.sort.return_value = stored
    _collection(mock_client, "chunks").find.return_value = find_result
    repository = DocumentRepository(mock_client, settings)

    result = repository.get_chunks("doc-1")

    assert result == [_chunk(0), _chunk(1)]
    find_result.sort.assert_called_once_with("chunk_index", 1)
