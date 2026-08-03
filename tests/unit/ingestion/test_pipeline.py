from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from configs.settings import Settings
from domain.chunk import Chunk
from domain.document import DocumentStatus
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

    result = pipeline.ingest("doc-1", Path("contract.pdf"))

    assert result.doc_id == "doc-1"
    assert result.filename == "contract.pdf"
    assert result.status == DocumentStatus.READY
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

    pipeline.ingest("doc-1", Path("contract.pdf"))

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
        pipeline.ingest("doc-1", Path("contract.pdf"))

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
        pipeline.ingest("doc-1", Path("contract.pdf"))

    messages = [record.getMessage() for record in caplog.records]
    joined = "\n".join(messages)

    assert any("upload started" in m for m in messages)
    assert any("document loaded" in m for m in messages)
    assert any("chunking complete" in m for m in messages)
    assert any("document stored" in m for m in messages)
    assert any("ingestion completed" in m for m in messages)
    assert all("doc_id=doc-1" in m for m in messages if "doc_id" in m)
    assert _SAMPLE_TEXT not in joined
