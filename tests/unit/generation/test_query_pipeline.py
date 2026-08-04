from __future__ import annotations

from unittest.mock import MagicMock

from configs.settings import Settings
from domain.chunk import Chunk
from domain.generation import GenerationResult
from domain.retrieval import RetrievalResult
from generation.pipeline import QueryPipeline


def _result(chunk_id: str) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        chunk_index=0,
        text="text",
        token_count=1,
        page_number=1,
        section="unspecified",
        source_filename="contract.pdf",
    )
    return RetrievalResult(chunk=chunk)


async def test_answer_chains_retrieve_rerank_and_generate() -> None:
    retriever = MagicMock(name="retriever")
    retrieved = [_result("a"), _result("b")]

    async def _retrieve(
        question: str, top_k: int | None = None, document_ids: list[str] | None = None
    ) -> list[RetrievalResult]:
        return retrieved

    retriever.retrieve = _retrieve

    reranker = MagicMock(name="reranker")
    reranked = [_result("b")]
    reranker.rerank.return_value = reranked

    generator = MagicMock(name="generator")
    expected = GenerationResult(answer="answer", citations=[], sources=[], confidence=0.9)
    generator.generate.return_value = expected

    pipeline = QueryPipeline(retriever, reranker, generator)
    result = await pipeline.answer("What is the term?")

    assert result == expected
    reranker.rerank.assert_called_once_with("What is the term?", retrieved)
    generator.generate.assert_called_once_with("What is the term?", reranked)


async def test_answer_handles_empty_retrieval() -> None:
    retriever = MagicMock(name="retriever")

    async def _retrieve(
        question: str, top_k: int | None = None, document_ids: list[str] | None = None
    ) -> list[RetrievalResult]:
        return []

    retriever.retrieve = _retrieve

    reranker = MagicMock(name="reranker")
    reranker.rerank.return_value = []

    generator = MagicMock(name="generator")
    refusal = GenerationResult(answer="refused", citations=[], sources=[], refused=True)
    generator.generate.return_value = refusal

    pipeline = QueryPipeline(retriever, reranker, generator)
    result = await pipeline.answer("question")

    assert result.refused is True
    reranker.rerank.assert_called_once_with("question", [])


async def test_answer_trims_to_rerank_input_top_k_before_reranking() -> None:
    retriever = MagicMock(name="retriever")
    retrieved = [_result("a"), _result("b"), _result("c"), _result("d")]

    async def _retrieve(
        question: str, top_k: int | None = None, document_ids: list[str] | None = None
    ) -> list[RetrievalResult]:
        return retrieved

    retriever.retrieve = _retrieve

    reranker = MagicMock(name="reranker")
    reranker.rerank.return_value = []

    generator = MagicMock(name="generator")
    generator.generate.return_value = GenerationResult(answer="a", citations=[], sources=[])

    settings = Settings(RERANK_INPUT_TOP_K=2)
    pipeline = QueryPipeline(retriever, reranker, generator, settings=settings)

    await pipeline.answer("question")

    reranker.rerank.assert_called_once_with("question", retrieved[:2])


async def test_answer_passes_through_all_candidates_under_the_limit() -> None:
    retriever = MagicMock(name="retriever")
    retrieved = [_result("a"), _result("b")]

    async def _retrieve(
        question: str, top_k: int | None = None, document_ids: list[str] | None = None
    ) -> list[RetrievalResult]:
        return retrieved

    retriever.retrieve = _retrieve

    reranker = MagicMock(name="reranker")
    reranker.rerank.return_value = []

    generator = MagicMock(name="generator")
    generator.generate.return_value = GenerationResult(answer="a", citations=[], sources=[])

    settings = Settings(RERANK_INPUT_TOP_K=20)
    pipeline = QueryPipeline(retriever, reranker, generator, settings=settings)

    await pipeline.answer("question")

    reranker.rerank.assert_called_once_with("question", retrieved)


async def test_answer_passes_document_ids_through_to_the_retriever() -> None:
    retriever = MagicMock(name="retriever")
    received: dict[str, object] = {}

    async def _retrieve(
        question: str, top_k: int | None = None, document_ids: list[str] | None = None
    ) -> list[RetrievalResult]:
        received["document_ids"] = document_ids
        return []

    retriever.retrieve = _retrieve

    reranker = MagicMock(name="reranker")
    reranker.rerank.return_value = []

    generator = MagicMock(name="generator")
    generator.generate.return_value = GenerationResult(answer="a", citations=[], sources=[])

    pipeline = QueryPipeline(retriever, reranker, generator)
    await pipeline.answer("question", document_ids=["doc-1", "doc-2"])

    assert received["document_ids"] == ["doc-1", "doc-2"]


async def test_answer_defaults_document_ids_to_none() -> None:
    retriever = MagicMock(name="retriever")
    received: dict[str, object] = {}

    async def _retrieve(
        question: str, top_k: int | None = None, document_ids: list[str] | None = None
    ) -> list[RetrievalResult]:
        received["document_ids"] = document_ids
        return []

    retriever.retrieve = _retrieve

    reranker = MagicMock(name="reranker")
    reranker.rerank.return_value = []

    generator = MagicMock(name="generator")
    generator.generate.return_value = GenerationResult(answer="a", citations=[], sources=[])

    pipeline = QueryPipeline(retriever, reranker, generator)
    await pipeline.answer("question")

    assert received["document_ids"] is None
