from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_document_repository, get_ingestion_pipeline
from api.main import app
from domain.document import Document
from domain.enums import DocumentStatus


@pytest.fixture(autouse=True)
def _default_repository_override() -> Iterator[None]:
    repository = MagicMock(name="repository")
    repository.list_documents.return_value = []
    app.dependency_overrides[get_document_repository] = lambda: repository

    pipeline = MagicMock(name="pipeline")
    pipeline.delete.return_value = None
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline

    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _use_tmp_raw_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.routes.documents.RAW_STORAGE_DIR", tmp_path)


def _document(
    doc_id: str,
    *,
    upload_timestamp: datetime,
    status: DocumentStatus = DocumentStatus.READY,
    page_count: int | None = 10,
    file_size: int | None = 1024,
    chunk_count: int | None = 5,
    indexed_at: datetime | None = None,
) -> Document:
    return Document(
        doc_id=doc_id,
        filename=f"{doc_id}.pdf",
        upload_timestamp=upload_timestamp,
        status=status,
        page_count=page_count,
        file_size=file_size,
        chunk_count=chunk_count,
        indexed_at=indexed_at,
    )


def test_list_documents_returns_empty_list_when_no_documents(client: TestClient) -> None:
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == {"documents": [], "total": 0}


def test_list_documents_preserves_repository_ordering(client: TestClient) -> None:
    newest = _document("doc-2", upload_timestamp=datetime(2026, 8, 4, tzinfo=UTC))
    oldest = _document("doc-1", upload_timestamp=datetime(2026, 8, 1, tzinfo=UTC))
    repository = MagicMock(name="repository")
    repository.list_documents.return_value = [newest, oldest]
    app.dependency_overrides[get_document_repository] = lambda: repository

    response = client.get("/documents")

    body = response.json()
    assert [d["doc_id"] for d in body["documents"]] == ["doc-2", "doc-1"]
    assert body["total"] == 2


def test_list_documents_includes_full_stats(client: TestClient) -> None:
    indexed_at = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
    document = _document(
        "doc-1",
        upload_timestamp=datetime(2026, 8, 4, tzinfo=UTC),
        status=DocumentStatus.READY,
        page_count=64,
        file_size=673562,
        chunk_count=58,
        indexed_at=indexed_at,
    )
    repository = MagicMock(name="repository")
    repository.list_documents.return_value = [document]
    app.dependency_overrides[get_document_repository] = lambda: repository

    response = client.get("/documents")

    summary = response.json()["documents"][0]
    assert summary["doc_id"] == "doc-1"
    assert summary["filename"] == "doc-1.pdf"
    assert summary["status"] == "ready"
    assert summary["page_count"] == 64
    assert summary["file_size"] == 673562
    assert summary["chunk_count"] == 58
    assert summary["embedding_status"] == "ready"
    assert summary["retrieval_ready"] is True


def test_list_documents_processing_document_is_not_retrieval_ready(client: TestClient) -> None:
    document = _document(
        "doc-1",
        upload_timestamp=datetime(2026, 8, 4, tzinfo=UTC),
        status=DocumentStatus.PROCESSING,
        page_count=None,
        chunk_count=None,
        indexed_at=None,
    )
    repository = MagicMock(name="repository")
    repository.list_documents.return_value = [document]
    app.dependency_overrides[get_document_repository] = lambda: repository

    response = client.get("/documents")

    summary = response.json()["documents"][0]
    assert summary["status"] == "processing"
    assert summary["embedding_status"] == "processing"
    assert summary["retrieval_ready"] is False
    assert summary["indexed_at"] is None
    assert summary["page_count"] is None


def test_delete_document_success(client: TestClient) -> None:
    existing = Document(doc_id="doc-1", filename="contract.pdf", status=DocumentStatus.READY)
    pipeline = MagicMock(name="pipeline")
    pipeline.delete.return_value = existing
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline

    response = client.delete("/documents/doc-1")

    assert response.status_code == 200
    assert response.json() == {
        "doc_id": "doc-1",
        "filename": "contract.pdf",
        "message": "Document deleted.",
    }
    pipeline.delete.assert_called_once_with("doc-1")


def test_delete_document_removes_raw_files_from_disk(client: TestClient, tmp_path: Path) -> None:
    existing = Document(doc_id="doc-1", filename="contract.pdf", status=DocumentStatus.READY)
    pipeline = MagicMock(name="pipeline")
    pipeline.delete.return_value = existing
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline

    raw_dir = tmp_path / "doc-1"
    raw_dir.mkdir()
    (raw_dir / "contract.pdf").write_bytes(b"pdf bytes")

    response = client.delete("/documents/doc-1")

    assert response.status_code == 200
    assert not raw_dir.exists()


def test_delete_document_returns_404_when_missing(client: TestClient) -> None:
    pipeline = MagicMock(name="pipeline")
    pipeline.delete.return_value = None
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline

    response = client.delete("/documents/no-such-doc")

    assert response.status_code == 404


def test_delete_document_missing_raw_files_is_not_an_error(client: TestClient) -> None:
    existing = Document(doc_id="doc-1", filename="contract.pdf", status=DocumentStatus.READY)
    pipeline = MagicMock(name="pipeline")
    pipeline.delete.return_value = existing
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline

    response = client.delete("/documents/doc-1")

    assert response.status_code == 200
