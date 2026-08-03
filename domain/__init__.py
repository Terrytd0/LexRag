"""Canonical domain models shared across ingestion, retrieval, generation, evaluation,
and the API. Single source of truth for business objects -- other modules import
these rather than defining duplicate representations. Introduced Sprint 5 Day 2;
grows as later sprint days introduce new concepts (e.g. a QueryResult wrapping
RetrievalResult + the generated answer, on Day 4).

Shared string-based enums (`DocumentStatus` and friends) live in `domain.enums`,
not scattered across individual model modules -- see that module's docstring.

All models here are Pydantic `BaseModel`s rather than dataclasses. Every one of
them either crosses a persistence boundary (Document/Chunk, written to and read
back from MongoDB), an API boundary (Citation, returned in `POST /query`
responses), or is built up field-by-field across pipeline stages owned by
different modules (RetrievalResult) -- in all three cases Pydantic's validation
and `model_dump`/`model_validate` (de)serialization is doing real work, not
ceremony. A dataclass would still need that (de)serialization hand-written at
every store/API boundary. Nothing in this package is a purely internal,
construct-once-and-discard value with no serialization need, so there's no
current case where a dataclass is the better fit.
"""

from __future__ import annotations

from domain.chunk import Chunk
from domain.citation import Citation
from domain.document import Document
from domain.enums import DocumentStatus
from domain.retrieval import RetrievalResult

__all__ = [
    "Chunk",
    "Citation",
    "Document",
    "DocumentStatus",
    "RetrievalResult",
]
