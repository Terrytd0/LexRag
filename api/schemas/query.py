"""Pydantic contracts for `POST /query`."""

from __future__ import annotations

from pydantic import BaseModel, Field

from domain.citation import Citation


class QueryRequest(BaseModel):
    """A natural-language question submitted for citation-grounded RAG (FR-6)."""

    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] | None = Field(
        default=None,
        description="Restrict retrieval to these doc_ids. Omit or leave null to search "
        "the full corpus. An empty list or unknown doc_ids yields no candidates and a "
        "refusal, not an error.",
        examples=[["<doc_id_1>"]],
    )


class QueryResponse(BaseModel):
    """A generated answer grounded in retrieved evidence, or an explicit refusal (FR-10)."""

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
