"""Automatic failure classification (task section 8): for every case that isn't a
straightforward success, attributes the failure to the pipeline stage most likely
responsible, given the golden case, its retrieval/generation results, and (when
available) its generation metrics.

Classification cascades through the pipeline in execution order -- the first stage
that couldn't have produced a correct answer is blamed, even if a later stage also
looks imperfect (citations are moot if the reranker never surfaced the right chunk).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from domain.citation import Citation
from evaluation.dataset import GoldenCase
from evaluation.metrics.generation import GenerationCaseMetrics
from evaluation.runner import CaseRun

FailureStage = Literal[
    "retrieval_failure", "reranker_failure", "generation_failure", "refusal_failure"
]

# Below this, a scored answer is treated as unfaithful/ungrounded enough to flag as
# a generation failure even when citations happen to be correct. Distinct from (and
# looser than) the 0.90 acceptance-criterion threshold in docs/01-requirements.md
# §7.4 -- that's the ship/no-ship bar; this is only "worth a human looking at."
FAITHFULNESS_REVIEW_THRESHOLD = 0.5


class FailureRecord(BaseModel):
    """One failed golden case: what was expected, what the pipeline produced, and
    which stage is most likely at fault.
    """

    case_id: str
    question: str
    expected_citations: list[str]
    retrieved_citations: list[str]
    generated_answer: str
    failure_stage: FailureStage


def _expected_doc_ids(case: GoldenCase, filename_to_doc_id: dict[str, str]) -> set[str]:
    return {filename_to_doc_id[f] for f in case.expected_documents}


def _citation_labels(citations: list[Citation]) -> list[str]:
    return [f"{c.filename}" + (f" ({c.section})" if c.section else "") for c in citations]


def classify_case(
    case: GoldenCase,
    run: CaseRun,
    filename_to_doc_id: dict[str, str],
    generation_metrics: GenerationCaseMetrics | None,
) -> FailureRecord | None:
    """Return a `FailureRecord` if `case` failed, or `None` if it succeeded."""
    result = run.result
    retrieved_citation_labels = _citation_labels(result.citations)

    if case.category == "negative":
        if result.refused:
            return None
        return FailureRecord(
            case_id=case.id,
            question=case.question,
            expected_citations=case.expected_citations,
            retrieved_citations=retrieved_citation_labels,
            generated_answer=result.answer,
            failure_stage="refusal_failure",
        )

    expected = _expected_doc_ids(case, filename_to_doc_id)
    hybrid_top10_docs = {r.chunk.doc_id for r in run.hybrid_results[:10]}
    reranked_docs = {r.chunk.doc_id for r in run.reranked}
    cited_docs = {c.doc_id for c in result.citations}

    stage: FailureStage
    if not (expected & hybrid_top10_docs):
        stage = "retrieval_failure"
    elif not (expected & reranked_docs):
        stage = "reranker_failure"
    elif result.refused:
        stage = "refusal_failure"
    elif not (expected & cited_docs):
        stage = "generation_failure"
    elif generation_metrics is not None and (
        generation_metrics.faithfulness < FAITHFULNESS_REVIEW_THRESHOLD
    ):
        stage = "generation_failure"
    else:
        return None

    return FailureRecord(
        case_id=case.id,
        question=case.question,
        expected_citations=case.expected_citations,
        retrieved_citations=retrieved_citation_labels,
        generated_answer=result.answer,
        failure_stage=stage,
    )
