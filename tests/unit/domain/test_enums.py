from __future__ import annotations

from enum import StrEnum

from domain.enums import DocumentStatus


def test_document_status_is_a_str_enum() -> None:
    assert issubclass(DocumentStatus, StrEnum)
    assert DocumentStatus.READY == "ready"


def test_document_status_values() -> None:
    assert {member.value for member in DocumentStatus} == {
        "pending",
        "processing",
        "ready",
        "failed",
    }
