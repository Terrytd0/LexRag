from __future__ import annotations

from domain.chunk import Chunk
from domain.citation import Citation
from domain.document import Document
from domain.enums import DocumentStatus
from domain.retrieval import RetrievalResult


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="doc-1:0",
        doc_id="doc-1",
        chunk_index=0,
        text="The term of this Agreement shall commence on the Effective Date.",
        token_count=12,
        page_number=1,
        section="unspecified",
        source_filename="contract.pdf",
    )


def test_document_defaults_to_pending_with_empty_metadata() -> None:
    document = Document(doc_id="doc-1", filename="contract.pdf")

    assert document.status == DocumentStatus.PENDING
    assert document.metadata == {}
    assert document.upload_timestamp is not None


def test_document_round_trips_through_json() -> None:
    document = Document(doc_id="doc-1", filename="contract.pdf", status=DocumentStatus.READY)

    restored = Document.model_validate_json(document.model_dump_json())

    assert restored == document


def test_citation_carries_full_provenance() -> None:
    citation = Citation(
        doc_id="doc-1",
        filename="contract.pdf",
        page_number=3,
        section="Termination",
        chunk_id="doc-1:2",
        snippet="either party may terminate for convenience",
    )

    assert citation.doc_id == "doc-1"
    assert citation.page_number == 3


def test_retrieval_result_scores_are_optional_until_populated() -> None:
    result = RetrievalResult(chunk=_chunk())

    assert result.dense_score is None
    assert result.sparse_score is None
    assert result.rrf_score is None
    assert result.rerank_score is None

    result.dense_score = 0.82
    assert result.dense_score == 0.82
