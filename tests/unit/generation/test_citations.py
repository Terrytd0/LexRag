from __future__ import annotations

from domain.chunk import Chunk
from domain.retrieval import RetrievalResult
from generation.citations import build_citations, find_invalid_citation_markers


def _result(chunk_id: str, text: str = "short snippet") -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        chunk_index=0,
        text=text,
        token_count=3,
        page_number=2,
        section="Section 4",
        source_filename="contract.pdf",
    )
    return RetrievalResult(chunk=chunk, rerank_score=0.8)


def test_build_citations_preserves_ranking_order() -> None:
    results = [_result("b"), _result("a")]

    citations = build_citations(results)

    assert [c.chunk_id for c in citations] == ["b", "a"]


def test_build_citations_includes_required_fields() -> None:
    citations = build_citations([_result("a")])

    citation = citations[0]
    assert citation.filename == "contract.pdf"
    assert citation.page_number == 2
    assert citation.section == "Section 4"
    assert citation.chunk_id == "a"
    assert citation.snippet == "short snippet"


def test_build_citations_deduplicates_by_chunk_id() -> None:
    citations = build_citations([_result("a"), _result("a"), _result("b")])

    assert [c.chunk_id for c in citations] == ["a", "b"]


def test_build_citations_truncates_long_snippets() -> None:
    long_text = "x" * 500
    citations = build_citations([_result("a", text=long_text)])

    snippet = citations[0].snippet
    assert len(snippet) <= 403
    assert snippet.endswith("...")


def test_build_citations_empty_input() -> None:
    assert build_citations([]) == []


def test_find_invalid_citation_markers_flags_out_of_range_markers() -> None:
    answer = "The term is five years [1]. It renews automatically [3]."

    invalid = find_invalid_citation_markers(answer, evidence_count=2)

    assert invalid == {3}


def test_find_invalid_citation_markers_accepts_markers_in_range() -> None:
    answer = "Clause A [1] and clause B [2]."

    invalid = find_invalid_citation_markers(answer, evidence_count=2)

    assert invalid == set()


def test_find_invalid_citation_markers_no_markers() -> None:
    assert find_invalid_citation_markers("plain answer, no citations", evidence_count=2) == set()
