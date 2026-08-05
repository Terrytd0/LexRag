"""Golden evaluation dataset: typed records loaded from `data/golden/golden_qa.jsonl`
(Sprint 5 Day 5). Cases live in that JSONL file, not in Python, so extending the
dataset never requires a code change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ingestion.repository import DocumentRepository

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "golden" / "golden_qa.jsonl"
)

NegativeSubtype = Literal["nonexistent_clause", "misleading", "hallucination_trap", "unrelated"]


class GoldenCase(BaseModel):
    """One golden-set question: an expected answer/citations for a positive case, or
    an expected refusal for a negative one.

    `expected_documents` names source PDFs by filename, not `doc_id` -- `doc_id`s are
    randomly generated per ingestion run (`scripts/seed_corpus.py`), so filenames are
    the only identifier stable across re-ingestion. `resolve_filename_doc_ids`
    resolves filename -> live `doc_id` once per evaluation run.
    """

    id: str
    topic: str
    category: Literal["positive", "negative"]
    negative_subtype: NegativeSubtype | None = None
    question: str
    expected_answer: str | None = None
    expected_documents: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    expected_refusal: bool

    @model_validator(mode="after")
    def _check_category_consistency(self) -> GoldenCase:
        if self.category == "negative" and not self.expected_refusal:
            raise ValueError(f"{self.id}: negative case must have expected_refusal=True")
        if self.category == "positive" and self.expected_refusal:
            raise ValueError(f"{self.id}: positive case must have expected_refusal=False")
        if self.category == "positive" and not self.expected_documents:
            raise ValueError(f"{self.id}: positive case must name an expected document")
        return self


def load_golden_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[GoldenCase]:
    """Load and validate every case in the golden JSONL dataset, in file order."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [GoldenCase.model_validate(json.loads(line)) for line in lines if line.strip()]


def resolve_filename_doc_ids(
    cases: list[GoldenCase], repository: DocumentRepository
) -> dict[str, str]:
    """Map every filename referenced by `cases` to its current `doc_id` in the live corpus.

    Raises if a case references a filename that isn't ingested -- the golden dataset
    and the seeded corpus must agree, so a silent skip doesn't masquerade as an
    inexplicable retrieval failure later in the harness.
    """
    by_filename = {doc.filename: doc.doc_id for doc in repository.list_documents()}
    referenced = {f for case in cases for f in case.expected_documents}
    missing = referenced - by_filename.keys()
    if missing:
        raise ValueError(
            f"golden dataset references documents not found in the corpus: {sorted(missing)}"
        )
    return by_filename
