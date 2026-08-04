"""Structured citation extraction from reranked RetrievalResults, and
post-generation validation that an answer's citation markers map to real
evidence (FR-9).
"""

from __future__ import annotations

import re

from domain.citation import Citation
from domain.retrieval import RetrievalResult

_SNIPPET_MAX_CHARS = 400
_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


def build_citations(results: list[RetrievalResult]) -> list[Citation]:
    """Build one `Citation` per unique chunk in `results`, preserving rank order.

    A chunk appearing more than once in `results` (shouldn't happen post-RRF,
    but not guaranteed by this function's contract) only produces one
    citation, keyed by `chunk_id`.
    """
    seen: set[str] = set()
    citations: list[Citation] = []
    for result in results:
        chunk = result.chunk
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        snippet = (
            chunk.text
            if len(chunk.text) <= _SNIPPET_MAX_CHARS
            else chunk.text[:_SNIPPET_MAX_CHARS].rstrip() + "..."
        )
        citations.append(
            Citation(
                doc_id=chunk.doc_id,
                filename=chunk.source_filename,
                page_number=chunk.page_number,
                section=chunk.section,
                chunk_id=chunk.chunk_id,
                snippet=snippet,
            )
        )
    return citations


def find_invalid_citation_markers(answer: str, evidence_count: int) -> set[int]:
    """Return citation markers (e.g. the `9` in `[9]`) referenced in `answer` that
    fall outside `1..evidence_count` -- i.e. don't map to a retrieved evidence block.
    """
    cited = {int(match) for match in _CITATION_MARKER_RE.findall(answer)}
    return {marker for marker in cited if marker < 1 or marker > evidence_count}
