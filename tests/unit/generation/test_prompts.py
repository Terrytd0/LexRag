from __future__ import annotations

from configs.settings import Settings
from domain.chunk import Chunk
from domain.retrieval import RetrievalResult
from generation.prompts import ACTIVE_PROMPT_VERSION, PromptBuilder


def _result(chunk_id: str, text: str = "evidence text") -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        chunk_index=0,
        text=text,
        token_count=3,
        page_number=4,
        section="Section 8.3",
        source_filename="contract.pdf",
    )
    return RetrievalResult(chunk=chunk, rerank_score=0.9)


def test_build_numbers_evidence_blocks_in_order() -> None:
    builder = PromptBuilder(context_window=5, settings=Settings())
    results = [_result("a", "first clause"), _result("b", "second clause")]

    prompt = builder.build("What is the termination clause?", results)

    first_index = prompt.user.index("[1]")
    second_index = prompt.user.index("[2]")
    assert first_index < second_index
    assert "first clause" in prompt.user
    assert "second clause" in prompt.user


def test_build_includes_citation_provenance() -> None:
    builder = PromptBuilder(context_window=5, settings=Settings())
    prompt = builder.build("question", [_result("a")])

    assert "contract.pdf" in prompt.user
    assert "page: 4" in prompt.user
    assert "Section 8.3" in prompt.user


def test_build_is_deterministic_for_the_same_input() -> None:
    builder = PromptBuilder(context_window=5, settings=Settings())
    results = [_result("a"), _result("b")]

    first = builder.build("question", results)
    second = builder.build("question", results)

    assert first.user == second.user
    assert first.system == second.system


def test_build_respects_configured_context_window() -> None:
    builder = PromptBuilder(context_window=1, settings=Settings())
    results = [_result("a", "first clause"), _result("b", "second clause")]

    prompt = builder.build("question", results)

    assert "first clause" in prompt.user
    assert "second clause" not in prompt.user
    assert "[2]" not in prompt.user


def test_build_handles_no_evidence() -> None:
    builder = PromptBuilder(context_window=5, settings=Settings())

    prompt = builder.build("question", [])

    assert "no evidence retrieved" in prompt.user


def test_build_uses_the_active_prompt_version() -> None:
    builder = PromptBuilder(context_window=5, settings=Settings())

    prompt = builder.build("question", [_result("a")])

    assert prompt.version == ACTIVE_PROMPT_VERSION.name
    assert prompt.system == ACTIVE_PROMPT_VERSION.system


def test_context_window_defaults_to_rerank_top_k() -> None:
    builder = PromptBuilder(settings=Settings(RERANK_TOP_K=3))

    assert builder.context_window == 3
