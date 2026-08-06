"""Prompt assembly for citation-grounded generation.

Active prompt version: `LEGAL_RAG_V2` (`generation.prompt_versions`) -- promoted
Sprint 5 Day 6 after a measured before/after against the golden set found `V1`
accepting two negative cases whose retrieved evidence was topically real but
didn't answer the specific question asked (see
`docs/experiments/evaluation_notes_day6.md`). Bump `ACTIVE_PROMPT_VERSION`
below to switch templates -- the emitted `Prompt.version` records exactly
which template produced a given answer, so prompt experiments stay
reproducible.

Depends only on `domain.retrieval.RetrievalResult` and `configs.settings` --
never on retrieval stores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from configs.settings import Settings, get_settings
from domain.retrieval import RetrievalResult
from generation.prompt_versions import LEGAL_RAG_V2, PromptTemplate

logger = logging.getLogger(__name__)

ACTIVE_PROMPT_VERSION: PromptTemplate = LEGAL_RAG_V2

_NO_EVIDENCE_PLACEHOLDER = "(no evidence retrieved)"


@dataclass(frozen=True)
class Prompt:
    """An assembled system/user message pair, tagged with the template version that built it."""

    system: str
    user: str
    version: str


class PromptBuilder:
    """Assembles a deterministic, citation-preserving prompt from reranked evidence.

    `context_window` caps how many reranked results are included as evidence
    blocks -- callers (e.g. `generation.generator.GenerationService`) must
    slice their own citation list the same way so citations stay in sync with
    what the LLM actually saw.
    """

    def __init__(
        self,
        context_window: int | None = None,
        template: PromptTemplate = ACTIVE_PROMPT_VERSION,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.context_window = context_window or self._settings.rerank_top_k
        self._template = template

    def build(self, query: str, results: list[RetrievalResult]) -> Prompt:
        """Build a `Prompt` from `query` and the top `context_window` of `results`.

        Evidence blocks are numbered `[1]`, `[2]`, ... in `results` order
        (already ranked by the reranker), so the numbering is deterministic
        for a given input and directly usable as citation markers.
        """
        evidence_results = results[: self.context_window]
        evidence = "\n\n".join(
            f"[{index}] (source: {result.chunk.source_filename}, "
            f"page: {result.chunk.page_number}, section: {result.chunk.section})\n"
            f"{result.chunk.text}"
            for index, result in enumerate(evidence_results, start=1)
        )
        user = self._template.instructions.format(
            question=query,
            evidence=evidence or _NO_EVIDENCE_PLACEHOLDER,
        )
        logger.debug(
            "prompt built version=%s evidence_count=%d",
            self._template.name,
            len(evidence_results),
        )
        return Prompt(system=self._template.system, user=user, version=self._template.name)
