from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from domain.document import Document
from domain.enums import DocumentStatus
from evaluation.dataset import GoldenCase, load_golden_dataset, resolve_filename_doc_ids
from ingestion.repository import DocumentRepository


def _positive_case(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "termination-01",
        "topic": "termination",
        "category": "positive",
        "negative_subtype": None,
        "question": "What is the notice period?",
        "expected_answer": "Thirty days.",
        "expected_documents": ["contract.pdf"],
        "expected_citations": ["Section 8"],
        "expected_refusal": False,
    }
    base.update(overrides)
    return base


def _negative_case(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "negative-01",
        "topic": "unrelated",
        "category": "negative",
        "negative_subtype": "unrelated",
        "question": "What is the boiling point of water?",
        "expected_answer": None,
        "expected_documents": [],
        "expected_citations": [],
        "expected_refusal": True,
    }
    base.update(overrides)
    return base


def test_golden_case_accepts_a_valid_positive_case() -> None:
    case = GoldenCase.model_validate(_positive_case())
    assert case.category == "positive"
    assert case.expected_refusal is False


def test_golden_case_accepts_a_valid_negative_case() -> None:
    case = GoldenCase.model_validate(_negative_case())
    assert case.category == "negative"
    assert case.expected_refusal is True


def test_golden_case_rejects_negative_case_with_expected_refusal_false() -> None:
    with pytest.raises(ValidationError):
        GoldenCase.model_validate(_negative_case(expected_refusal=False))


def test_golden_case_rejects_positive_case_with_expected_refusal_true() -> None:
    with pytest.raises(ValidationError):
        GoldenCase.model_validate(_positive_case(expected_refusal=True))


def test_golden_case_rejects_positive_case_with_no_expected_documents() -> None:
    with pytest.raises(ValidationError):
        GoldenCase.model_validate(_positive_case(expected_documents=[]))


def test_load_golden_dataset_parses_every_line(tmp_path: Path) -> None:
    dataset_path = tmp_path / "golden_qa.jsonl"
    dataset_path.write_text(
        json.dumps(_positive_case()) + "\n" + json.dumps(_negative_case()) + "\n",
        encoding="utf-8",
    )

    cases = load_golden_dataset(dataset_path)

    assert [c.id for c in cases] == ["termination-01", "negative-01"]


def test_load_golden_dataset_skips_blank_lines(tmp_path: Path) -> None:
    dataset_path = tmp_path / "golden_qa.jsonl"
    dataset_path.write_text(json.dumps(_positive_case()) + "\n\n", encoding="utf-8")

    cases = load_golden_dataset(dataset_path)

    assert len(cases) == 1


def _document(doc_id: str, filename: str) -> Document:
    return Document(doc_id=doc_id, filename=filename, status=DocumentStatus.READY)


def test_resolve_filename_doc_ids_maps_every_referenced_filename() -> None:
    cases = [GoldenCase.model_validate(_positive_case(expected_documents=["a.pdf", "b.pdf"]))]
    repository = MagicMock(spec=DocumentRepository)
    repository.list_documents.return_value = [
        _document("doc-a", "a.pdf"),
        _document("doc-b", "b.pdf"),
        _document("doc-c", "c.pdf"),
    ]

    mapping = resolve_filename_doc_ids(cases, repository)

    assert mapping == {"a.pdf": "doc-a", "b.pdf": "doc-b", "c.pdf": "doc-c"}


def test_resolve_filename_doc_ids_raises_on_missing_document() -> None:
    cases = [GoldenCase.model_validate(_positive_case(expected_documents=["missing.pdf"]))]
    repository = MagicMock(spec=DocumentRepository)
    repository.list_documents.return_value = [_document("doc-a", "a.pdf")]

    with pytest.raises(ValueError, match="missing.pdf"):
        resolve_filename_doc_ids(cases, repository)
