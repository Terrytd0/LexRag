"""Recall@K / Precision@K, computed directly against the golden set's expected
documents rather than delegated to RAGAS (`docs/architecture.md` §2.10 -- retrieval
metrics "are straightforward to compute directly against the golden set's
known-relevant chunks").

Relevance is judged at the document level (does a retrieved chunk belong to one of
`expected_documents`), not at the exact chunk: chunk IDs, like `doc_id`s, aren't
stable across re-ingestion, so there is no stable chunk-level ground truth to pin the
dataset to. See `docs/experiments/evaluation_notes.md` for this assumption.
"""

from __future__ import annotations

from pydantic import BaseModel

from domain.retrieval import RetrievalResult


class RetrievalCaseMetrics(BaseModel):
    """Recall@K / Precision@K for one golden case under one retrieval strategy."""

    case_id: str
    strategy: str
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    precision_at_10: float
    retrieved_doc_ids: list[str]


def relevant_doc_ids(expected_documents: list[str], filename_to_doc_id: dict[str, str]) -> set[str]:
    """The set of live `doc_id`s a case considers relevant, resolved from filenames."""
    return {filename_to_doc_id[f] for f in expected_documents}


def score_retrieval_case(
    case_id: str,
    strategy: str,
    results: list[RetrievalResult],
    relevant: set[str],
) -> RetrievalCaseMetrics:
    """Recall@K / Precision@K for one case, given `results` best-first and its
    relevant `doc_id`s. Callers should only score positive-category cases -- a
    case with no relevant document has no defined recall/precision.
    """
    doc_ids = [r.chunk.doc_id for r in results]

    def recall_at(k: int) -> float:
        if not relevant:
            return 0.0
        return len(relevant & set(doc_ids[:k])) / len(relevant)

    def precision_at(k: int) -> float:
        top_k = doc_ids[:k]
        if not top_k:
            return 0.0
        return sum(1 for d in top_k if d in relevant) / len(top_k)

    return RetrievalCaseMetrics(
        case_id=case_id,
        strategy=strategy,
        recall_at_5=recall_at(5),
        recall_at_10=recall_at(10),
        precision_at_5=precision_at(5),
        precision_at_10=precision_at(10),
        retrieved_doc_ids=doc_ids[:10],
    )


class RetrievalStrategyReport(BaseModel):
    """Recall@K / Precision@K for one retrieval strategy, averaged across cases."""

    strategy: str
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    precision_at_10: float
    case_count: int
    per_case: list[RetrievalCaseMetrics]


def aggregate_retrieval_metrics(
    strategy: str, per_case: list[RetrievalCaseMetrics]
) -> RetrievalStrategyReport:
    """Mean Recall@K / Precision@K across `per_case` for one strategy."""
    n = len(per_case)
    if n == 0:
        return RetrievalStrategyReport(
            strategy=strategy,
            recall_at_5=0.0,
            recall_at_10=0.0,
            precision_at_5=0.0,
            precision_at_10=0.0,
            case_count=0,
            per_case=[],
        )
    return RetrievalStrategyReport(
        strategy=strategy,
        recall_at_5=sum(c.recall_at_5 for c in per_case) / n,
        recall_at_10=sum(c.recall_at_10 for c in per_case) / n,
        precision_at_5=sum(c.precision_at_5 for c in per_case) / n,
        precision_at_10=sum(c.precision_at_10 for c in per_case) / n,
        case_count=n,
        per_case=per_case,
    )
