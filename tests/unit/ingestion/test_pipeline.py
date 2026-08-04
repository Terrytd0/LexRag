from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from configs.settings import Settings
from domain.chunk import Chunk
from domain.enums import DocumentStatus
from ingestion.loaders.pdf_loader import LoadedDocument
from ingestion.pipeline import IngestionPipeline


@pytest.fixture
def settings() -> Settings:
    return Settings(CHUNK_SIZE_TOKENS=50, CHUNK_OVERLAP_TOKENS=5)


@pytest.fixture
def repository() -> MagicMock:
    return MagicMock(name="repository")


_SAMPLE_TEXT = (
    "This Master Services Agreement is entered into by and between the "
    "parties for the provision of consulting services, subject to the "
    "termination and indemnification clauses set forth below."
)


def test_ingest_success_stores_document_and_chunks(
    repository: MagicMock, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "ingestion.pipeline.load_pdf",
        lambda path: LoadedDocument(filename=path.name, pages=[_SAMPLE_TEXT]),
    )
    pipeline = IngestionPipeline(repository, settings=settings)

    saved_statuses: list[DocumentStatus] = []
    repository.save_document.side_effect = lambda doc: saved_statuses.append(doc.status)

    result = pipeline.ingest("doc-1", Path("contract.pdf"), "content-hash-abc", 2048)

    assert result.doc_id == "doc-1"
    assert result.filename == "contract.pdf"
    assert result.status == DocumentStatus.READY
    assert result.content_hash == "content-hash-abc"
    assert result.file_size == 2048
    assert result.page_count == 1
    assert result.chunk_count is not None and result.chunk_count > 0
    assert result.indexed_at is not None
    # Captured at call time (not read back from the mock afterward) since the
    # pipeline mutates and re-saves the same Document instance in place.
    assert saved_statuses == [DocumentStatus.PROCESSING, DocumentStatus.READY]

    repository.save_chunks.assert_called_once()
    saved_chunks: list[Chunk] = repository.save_chunks.call_args.args[0]
    assert len(saved_chunks) > 0
    assert all(chunk.doc_id == "doc-1" for chunk in saved_chunks)


def test_ingest_passes_chunks_to_every_indexer(
    repository: MagicMock, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "ingestion.pipeline.load_pdf",
        lambda path: LoadedDocument(filename=path.name, pages=[_SAMPLE_TEXT]),
    )
    indexer = MagicMock(name="indexer")
    pipeline = IngestionPipeline(repository, indexers=[indexer], settings=settings)

    pipeline.ingest("doc-1", Path("contract.pdf"), "content-hash-abc", 2048)

    indexer.index_chunks.assert_called_once()
    indexed_chunks: list[Chunk] = indexer.index_chunks.call_args.args[0]
    assert len(indexed_chunks) > 0


def test_ingest_failure_marks_document_failed_and_reraises(
    repository: MagicMock, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(path: Path) -> LoadedDocument:
        raise ValueError("corrupt PDF")

    monkeypatch.setattr("ingestion.pipeline.load_pdf", _boom)
    pipeline = IngestionPipeline(repository, settings=settings)

    with pytest.raises(ValueError, match="corrupt PDF"):
        pipeline.ingest("doc-1", Path("contract.pdf"), "content-hash-abc", 2048)

    save_document_calls = [call.args[0] for call in repository.save_document.call_args_list]
    assert save_document_calls[-1].status == DocumentStatus.FAILED
    repository.save_chunks.assert_not_called()


def test_ingest_logs_lifecycle_events_without_document_text(
    repository: MagicMock,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        "ingestion.pipeline.load_pdf",
        lambda path: LoadedDocument(filename=path.name, pages=[_SAMPLE_TEXT]),
    )
    pipeline = IngestionPipeline(repository, settings=settings)

    with caplog.at_level(logging.INFO, logger="ingestion.pipeline"):
        pipeline.ingest("doc-1", Path("contract.pdf"), "content-hash-abc", 2048)

    messages = [record.getMessage() for record in caplog.records]
    joined = "\n".join(messages)

    assert any("upload started" in m for m in messages)
    assert any("document loaded" in m for m in messages)
    assert any("chunking complete" in m for m in messages)
    assert any("document stored" in m for m in messages)
    assert any("ingestion completed" in m for m in messages)
    assert all("doc_id=doc-1" in m for m in messages if "doc_id" in m)
    assert _SAMPLE_TEXT not in joined


def test_find_existing_document_delegates_to_repository(
    repository: MagicMock, settings: Settings
) -> None:
    from domain.document import Document

    existing = Document(doc_id="doc-1", filename="contract.pdf", content_hash="abc123")
    repository.get_document_by_hash.return_value = existing
    pipeline = IngestionPipeline(repository, settings=settings)

    result = pipeline.find_existing_document("abc123")

    assert result == existing
    repository.get_document_by_hash.assert_called_once_with("abc123")


def test_find_existing_document_returns_none_when_no_match(
    repository: MagicMock, settings: Settings
) -> None:
    repository.get_document_by_hash.return_value = None
    pipeline = IngestionPipeline(repository, settings=settings)

    assert pipeline.find_existing_document("no-such-hash") is None


def test_delete_removes_document_from_repository_and_every_indexer(
    repository: MagicMock, settings: Settings
) -> None:
    from domain.document import Document

    existing = Document(doc_id="doc-1", filename="contract.pdf")
    repository.get_document.return_value = existing
    indexer_a = MagicMock(name="indexer_a")
    indexer_b = MagicMock(name="indexer_b")
    pipeline = IngestionPipeline(repository, indexers=[indexer_a, indexer_b], settings=settings)

    result = pipeline.delete("doc-1")

    assert result == existing
    indexer_a.delete_document.assert_called_once_with("doc-1")
    indexer_b.delete_document.assert_called_once_with("doc-1")
    repository.delete_document.assert_called_once_with("doc-1")


def test_delete_returns_none_and_does_nothing_when_document_missing(
    repository: MagicMock, settings: Settings
) -> None:
    repository.get_document.return_value = None
    indexer = MagicMock(name="indexer")
    pipeline = IngestionPipeline(repository, indexers=[indexer], settings=settings)

    result = pipeline.delete("no-such-doc")

    assert result is None
    indexer.delete_document.assert_not_called()
    repository.delete_document.assert_not_called()
