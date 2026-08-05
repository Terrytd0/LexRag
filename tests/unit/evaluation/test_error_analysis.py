from __future__ import annotations

from domain.chunk import Chunk
from domain.citation import Citation
from domain.generation import GenerationResult
from domain.retrieval import RetrievalResult
from evaluation.dataset import GoldenCase
from evaluation.error_analysis import classify_case
from evaluation.metrics.generation import GenerationCaseMetrics
from evaluation.runner import CaseRun

FILENAME_TO_DOC_ID = {"expected.pdf": "doc-expected"}


def _positive_case() -> GoldenCase:
    return GoldenCase(
        id="case-1",
        topic="termination",
        category="positive",
        question="q",
        expected_answer="a",
        expected_documents=["expected.pdf"],
        expected_citations=["Section 1"],
        expected_refusal=False,
    )


def _negative_case() -> GoldenCase:
    return GoldenCase(
        id="case-1",
        topic="unrelated",
        category="negative",
        negative_subtype="unrelated",
        question="q",
        expected_refusal=True,
    )


def _result(doc_id: str, index: int) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=f"{doc_id}:{index}",
        doc_id=doc_id,
        chunk_index=index,
        text="text",
        token_count=1,
        page_number=1,
        section="unspecified",
        source_filename=f"{doc_id}.pdf",
    )
    return RetrievalResult(chunk=chunk)


def _run(
    hybrid_docs: list[str],
    reranked_docs: list[str],
    cited_doc_ids: list[str],
    refused: bool = False,
) -> CaseRun:
    citations = [
        Citation(
            doc_id=doc_id,
            filename=f"{doc_id}.pdf",
            page_number=1,
            section=None,
            chunk_id=f"{doc_id}:0",
            snippet="snippet",
        )
        for doc_id in cited_doc_ids
    ]
    result = GenerationResult(answer="the answer", citations=citations, sources=[], refused=refused)
    return CaseRun(
        dense_results=[],
        sparse_results=[],
        hybrid_results=[_result(d, i) for i, d in enumerate(hybrid_docs)],
        reranked=[_result(d, i) for i, d in enumerate(reranked_docs)],
        result=result,
        retrieval_latency_s=0.0,
        reranker_latency_s=0.0,
        generation_latency_s=0.0,
    )


def test_classify_case_success_when_expected_doc_cited() -> None:
    run = _run(
        hybrid_docs=["doc-expected"],
        reranked_docs=["doc-expected"],
        cited_doc_ids=["doc-expected"],
    )

    failure = classify_case(_positive_case(), run, FILENAME_TO_DOC_ID, None)

    assert failure is None


def test_classify_case_retrieval_failure_when_expected_doc_never_in_top_10() -> None:
    run = _run(hybrid_docs=["doc-other"], reranked_docs=["doc-other"], cited_doc_ids=["doc-other"])

    failure = classify_case(_positive_case(), run, FILENAME_TO_DOC_ID, None)

    assert failure is not None
    assert failure.failure_stage == "retrieval_failure"


def test_classify_case_reranker_failure_when_expected_doc_dropped_before_generation() -> None:
    run = _run(
        hybrid_docs=["doc-expected", "doc-other"],
        reranked_docs=["doc-other"],
        cited_doc_ids=["doc-other"],
    )

    failure = classify_case(_positive_case(), run, FILENAME_TO_DOC_ID, None)

    assert failure is not None
    assert failure.failure_stage == "reranker_failure"


def test_classify_case_refusal_failure_when_positive_case_refused_despite_evidence() -> None:
    run = _run(
        hybrid_docs=["doc-expected"],
        reranked_docs=["doc-expected"],
        cited_doc_ids=[],
        refused=True,
    )

    failure = classify_case(_positive_case(), run, FILENAME_TO_DOC_ID, None)

    assert failure is not None
    assert failure.failure_stage == "refusal_failure"


def test_classify_case_generation_failure_when_wrong_doc_cited() -> None:
    run = _run(
        hybrid_docs=["doc-expected"],
        reranked_docs=["doc-expected"],
        cited_doc_ids=["doc-other"],
    )

    failure = classify_case(_positive_case(), run, FILENAME_TO_DOC_ID, None)

    assert failure is not None
    assert failure.failure_stage == "generation_failure"


def test_classify_case_generation_failure_when_faithfulness_below_review_threshold() -> None:
    run = _run(
        hybrid_docs=["doc-expected"],
        reranked_docs=["doc-expected"],
        cited_doc_ids=["doc-expected"],
    )
    low_faithfulness = GenerationCaseMetrics(
        case_id="case-1",
        faithfulness=0.1,
        context_precision=1.0,
        context_recall=1.0,
        answer_relevancy=1.0,
    )

    failure = classify_case(_positive_case(), run, FILENAME_TO_DOC_ID, low_faithfulness)

    assert failure is not None
    assert failure.failure_stage == "generation_failure"


def test_classify_case_negative_success_when_refused() -> None:
    run = _run(hybrid_docs=[], reranked_docs=[], cited_doc_ids=[], refused=True)

    failure = classify_case(_negative_case(), run, FILENAME_TO_DOC_ID, None)

    assert failure is None


def test_classify_case_negative_refusal_failure_when_answered() -> None:
    run = _run(hybrid_docs=["doc-x"], reranked_docs=["doc-x"], cited_doc_ids=["doc-x"])

    failure = classify_case(_negative_case(), run, FILENAME_TO_DOC_ID, None)

    assert failure is not None
    assert failure.failure_stage == "refusal_failure"
