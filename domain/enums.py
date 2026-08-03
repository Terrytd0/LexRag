"""Shared string-based enums for the domain layer -- single source of truth for
status/type constants used across ingestion, retrieval, generation, and evaluation,
instead of scattered string literals.

Add new enums here as later sprint days introduce them (e.g. `RetrievalStrategy`
distinguishing dense/sparse/hybrid, `EvaluationStatus` for golden-run outcomes,
`ChunkType` if non-text chunk sources are added) -- this module is the intended
home for all of them, not just `DocumentStatus`.
"""

from __future__ import annotations

from enum import StrEnum


class DocumentStatus(StrEnum):
    """Lifecycle status of a document through the ingestion pipeline."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
