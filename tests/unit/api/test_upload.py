from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_ingestion_pipeline
from api.main import app
from domain.document import Document
from domain.enums import DocumentStatus


@pytest.fixture(autouse=True)
def _use_tmp_raw_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.routes.upload.RAW_STORAGE_DIR", tmp_path)


@pytest.fixture(autouse=True)
def _default_pipeline_override() -> Iterator[None]:
    # FastAPI resolves `Depends()` sub-dependencies even for requests that fail
    # body validation, so every test needs an override in place -- otherwise a
    # "malformed request" test would construct the real IngestionPipeline
    # (and its real embedding/reranker models) before ever reaching the route.
    pipeline = MagicMock(name="pipeline")
    pipeline.find_existing_document.return_value = None
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline
    yield
    app.dependency_overrides.clear()


def test_upload_success_returns_document_metadata(client: TestClient) -> None:
    pipeline = MagicMock(name="pipeline")
    pipeline.find_existing_document.return_value = None
    pipeline.ingest.return_value = Document(
        doc_id="doc-1", filename="contract.pdf", status=DocumentStatus.READY
    )
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline

    response = client.post(
        "/upload",
        files={"file": ("contract.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "success"
    assert body["doc_id"] == "doc-1"
    assert body["filename"] == "contract.pdf"
    assert body["document_status"] == "ready"
    pipeline.ingest.assert_called_once()


def test_upload_rejects_non_pdf_content(client: TestClient) -> None:
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 400


def test_upload_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/upload",
        files={"file": ("contract.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400


def test_upload_returns_422_when_ingestion_fails(client: TestClient) -> None:
    pipeline = MagicMock(name="pipeline")
    pipeline.find_existing_document.return_value = None
    pipeline.ingest.side_effect = ValueError("corrupt PDF")
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline

    response = client.post(
        "/upload",
        files={"file": ("contract.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
    )

    assert response.status_code == 422
    assert "stack" not in response.text.lower()
    assert "traceback" not in response.text.lower()


def test_upload_missing_file_is_a_validation_error(client: TestClient) -> None:
    response = client.post("/upload")

    assert response.status_code == 422


def test_upload_duplicate_content_skips_ingestion(client: TestClient) -> None:
    pipeline = MagicMock(name="pipeline")
    existing = Document(doc_id="doc-1", filename="contract.pdf", status=DocumentStatus.READY)
    pipeline.find_existing_document.return_value = existing
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline

    response = client.post(
        "/upload",
        files={"file": ("contract-reupload.pdf", b"same bytes as before", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "already_exists",
        "doc_id": "doc-1",
        "filename": "contract.pdf",
        "message": "Document has already been ingested.",
    }
    pipeline.ingest.assert_not_called()


def test_upload_same_content_hashes_identically_regardless_of_filename(
    client: TestClient,
) -> None:
    pipeline = MagicMock(name="pipeline")
    pipeline.find_existing_document.return_value = None
    pipeline.ingest.return_value = Document(
        doc_id="doc-1", filename="a.pdf", status=DocumentStatus.READY
    )
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline

    client.post("/upload", files={"file": ("a.pdf", b"identical content", "application/pdf")})
    client.post("/upload", files={"file": ("b.pdf", b"identical content", "application/pdf")})

    first_hash = pipeline.find_existing_document.call_args_list[0].args[0]
    second_hash = pipeline.find_existing_document.call_args_list[1].args[0]
    assert first_hash == second_hash


def test_upload_different_content_hashes_differently_with_same_filename(
    client: TestClient,
) -> None:
    pipeline = MagicMock(name="pipeline")
    pipeline.find_existing_document.return_value = None
    pipeline.ingest.return_value = Document(
        doc_id="doc-1", filename="a.pdf", status=DocumentStatus.READY
    )
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline

    client.post("/upload", files={"file": ("a.pdf", b"content version one", "application/pdf")})
    client.post("/upload", files={"file": ("a.pdf", b"content version two", "application/pdf")})

    first_hash = pipeline.find_existing_document.call_args_list[0].args[0]
    second_hash = pipeline.find_existing_document.call_args_list[1].args[0]
    assert first_hash != second_hash


def test_upload_ingest_called_with_content_hash_and_file_size(client: TestClient) -> None:
    pipeline = MagicMock(name="pipeline")
    pipeline.find_existing_document.return_value = None
    pipeline.ingest.return_value = Document(
        doc_id="doc-1", filename="a.pdf", status=DocumentStatus.READY
    )
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline

    content = b"some pdf bytes"
    client.post("/upload", files={"file": ("a.pdf", content, "application/pdf")})

    call_args = pipeline.ingest.call_args.args
    content_hash_arg = call_args[2]
    file_size_arg = call_args[3]
    assert content_hash_arg == pipeline.find_existing_document.call_args.args[0]
    assert file_size_arg == len(content)
