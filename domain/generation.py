"""Canonical GenerationResult model: the structured output of the generation service."""

from __future__ import annotations

from pydantic import BaseModel, Field

from domain.citation import Citation


class GenerationResult(BaseModel):
    """A citation-grounded answer, or an explicit refusal when evidence is insufficient (FR-10)."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    confidence: float | None = Field(
        default=None,
        description="The reranker relevance score of the highest-ranked retrieved chunk. "
        "It is a retrieval confidence proxy, not a calibrated measure of answer "
        "correctness. Multi-part questions that require synthesis across several chunks "
        "may naturally report lower values despite producing accurate, well-supported "
        "answers.",
    )
    refused: bool = False
