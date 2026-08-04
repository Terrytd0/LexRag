from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from configs.settings import Settings
from domain.chunk import Chunk
from domain.retrieval import RetrievalResult
from generation.generator import REFUSAL_ANSWER, GenerationService
from generation.prompts import PromptBuilder


def _result(
    chunk_id: str, rerank_score: float | None, text: str = "evidence text"
) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        chunk_index=0,
        text=text,
        token_count=3,
        page_number=1,
        section="unspecified",
        source_filename="contract.pdf",
    )
    return RetrievalResult(chunk=chunk, rerank_score=rerank_score)


@pytest.fixture
def settings() -> Settings:
    return Settings(GENERATION_MIN_CONTEXT_SCORE=0.35, RERANK_TOP_K=8)


def _service(settings: Settings, llm_response: str) -> tuple[GenerationService, MagicMock]:
    llm = MagicMock(name="llm")
    llm.complete.return_value = llm_response
    service = GenerationService(llm, PromptBuilder(settings=settings), settings=settings)
    return service, llm


def test_generate_returns_answer_with_citations(settings: Settings) -> None:
    service, llm = _service(settings, "The term is five years [1].")
    results = [_result("a", 0.9)]

    result = service.generate("How long is the term?", results)

    assert result.refused is False
    assert result.answer == "The term is five years [1]."
    assert [c.chunk_id for c in result.citations] == ["a"]
    assert result.sources == ["contract.pdf"]
    assert result.confidence == 0.9
    llm.complete.assert_called_once()


def test_generate_refuses_when_no_evidence(settings: Settings) -> None:
    service, llm = _service(settings, "should not be used")

    result = service.generate("question", [])

    assert result.refused is True
    assert result.answer == REFUSAL_ANSWER
    assert result.citations == []
    assert result.sources == []
    assert result.confidence is None
    llm.complete.assert_not_called()


def test_generate_refuses_when_top_score_below_threshold(settings: Settings) -> None:
    service, llm = _service(settings, "should not be used")
    results = [_result("a", 0.1)]

    result = service.generate("question", results)

    assert result.refused is True
    assert result.confidence == 0.1
    llm.complete.assert_not_called()


def test_generate_refuses_when_score_missing(settings: Settings) -> None:
    service, llm = _service(settings, "should not be used")
    results = [_result("a", None)]

    result = service.generate("question", results)

    assert result.refused is True
    llm.complete.assert_not_called()


def test_generate_normalizes_model_judged_refusal(settings: Settings) -> None:
    service, _ = _service(settings, REFUSAL_ANSWER)
    results = [_result("a", 0.9)]

    result = service.generate("question", results)

    assert result.refused is True
    assert result.answer == REFUSAL_ANSWER
    assert result.citations == []
    assert result.confidence == 0.9


def test_generate_logs_invalid_citation_markers(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    service, _ = _service(settings, "Claim one [1]. Claim two [9].")
    results = [_result("a", 0.9)]

    with caplog.at_level(logging.WARNING, logger="generation.generator"):
        result = service.generate("question", results)

    assert result.refused is False
    messages = [record.getMessage() for record in caplog.records]
    assert any("markers=[9]" in m for m in messages)


def test_generate_citations_limited_to_prompt_context_window(settings: Settings) -> None:
    narrow_settings = Settings(GENERATION_MIN_CONTEXT_SCORE=0.35, RERANK_TOP_K=1)
    service, _ = _service(narrow_settings, "Answer [1].")
    results = [_result("a", 0.9), _result("b", 0.8)]

    result = service.generate("question", results)

    assert [c.chunk_id for c in result.citations] == ["a"]
