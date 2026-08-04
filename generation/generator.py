"""Provider-agnostic generation service: reranked evidence -> prompt -> LLM ->
structured, citation-grounded response, or an explicit refusal (FR-10).

Depends only on `PromptBuilder`, `domain.retrieval.RetrievalResult`, and the
`LLMProvider` interface -- never on retrieval stores (`docs/architecture.md`
§2.9, CLAUDE.md "Maintain architectural boundaries").
"""

from __future__ import annotations

import logging

from configs.settings import Settings, get_settings
from domain.generation import GenerationResult
from domain.retrieval import RetrievalResult
from generation.citations import build_citations, find_invalid_citation_markers
from generation.prompts import PromptBuilder
from generation.providers import LLMProvider

logger = logging.getLogger(__name__)

REFUSAL_ANSWER = "I don't have enough evidence in the retrieved documents to answer this question."


class GenerationService:
    """Builds a prompt from reranked evidence, calls the configured LLM, and returns
    a structured `GenerationResult` -- an answer with citations, or a refusal.

    Refusal (FR-10) is enforced structurally, not left to the prompt alone
    (CLAUDE.md "Do not rely solely on prompt instructions"):

    1. No retrieved evidence, or the top reranked score is below
       `Settings.generation_min_context_score` -- refuses before ever
       calling the LLM.
    2. The LLM itself judges the evidence insufficient and emits the exact
       `REFUSAL_ANSWER` sentence instructed by the active prompt template --
       detected post-generation and normalized into a `refused=True` result
       with citations cleared, rather than trusted verbatim.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_builder: PromptBuilder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._llm = llm_provider
        self._settings = settings or get_settings()
        self._prompts = prompt_builder or PromptBuilder(settings=self._settings)

    def generate(self, query: str, results: list[RetrievalResult]) -> GenerationResult:
        """Generate an answer for `query` grounded in reranked `results`."""
        top_score = self._top_score(results)
        if self._should_refuse(results, top_score):
            logger.info(
                "refusal issued reason=insufficient_evidence evidence_count=%d top_score=%s",
                len(results),
                top_score,
            )
            return self._refusal(top_score)

        evidence = results[: self._prompts.context_window]
        prompt = self._prompts.build(query, results)
        answer = self._llm.complete(prompt.system, prompt.user)

        if REFUSAL_ANSWER.lower() in answer.lower():
            logger.info(
                "refusal issued reason=model_judged_insufficient evidence_count=%d top_score=%s",
                len(results),
                top_score,
            )
            return self._refusal(top_score)

        invalid_markers = find_invalid_citation_markers(answer, len(evidence))
        if invalid_markers:
            logger.warning(
                "generation cited evidence markers with no matching chunk markers=%s",
                sorted(invalid_markers),
            )

        citations = build_citations(evidence)
        sources = list(dict.fromkeys(citation.filename for citation in citations))
        logger.info(
            "generation complete prompt_version=%s citation_count=%d",
            prompt.version,
            len(citations),
        )
        return GenerationResult(
            answer=answer,
            citations=citations,
            sources=sources,
            confidence=top_score,
            refused=False,
        )

    def _should_refuse(self, results: list[RetrievalResult], top_score: float | None) -> bool:
        if not results:
            return True
        return top_score is None or top_score < self._settings.generation_min_context_score

    def _refusal(self, top_score: float | None) -> GenerationResult:
        return GenerationResult(
            answer=REFUSAL_ANSWER,
            citations=[],
            sources=[],
            confidence=top_score,
            refused=True,
        )

    @staticmethod
    def _top_score(results: list[RetrievalResult]) -> float | None:
        """The reranker score of the single highest-ranked chunk -- reported to callers
        as `GenerationResult.confidence`. This is a retrieval-relevance proxy for one
        chunk, not a calibrated correctness score for the whole answer: a synthesis
        question whose evidence is spread across several chunks will report a lower
        value here even when the answer is accurate and well-supported, since no single
        chunk is a tight match for the full question.
        """
        if not results:
            return None
        return results[0].rerank_score
