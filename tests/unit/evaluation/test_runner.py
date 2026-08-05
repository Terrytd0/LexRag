from __future__ import annotations

from unittest.mock import MagicMock

from configs.settings import Settings
from domain.chunk import Chunk
from domain.generation import GenerationResult
from domain.retrieval import RetrievalResult
from evaluation.runner import run_case


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


def _dense(results: list[RetrievalResult]) -> MagicMock:
    dense = MagicMock(name="dense")
    dense.retrieve.return_value = results
    return dense


def _sparse(results: list[RetrievalResult]) -> MagicMock:
    sparse = MagicMock(name="sparse")
    sparse.retrieve.return_value = results
    return sparse


def _hybrid(results: list[RetrievalResult]) -> MagicMock:
    hybrid = MagicMock(name="hybrid")

    async def _retrieve(question: str) -> list[RetrievalResult]:
        return results

    hybrid.retrieve = _retrieve
    return hybrid


def _reranker(results: list[RetrievalResult]) -> MagicMock:
    reranker = MagicMock(name="reranker")
    reranker.rerank.return_value = results
    return reranker


def _generator(result: GenerationResult) -> MagicMock:
    generator = MagicMock(name="generator")
    generator.generate.return_value = result
    return generator


async def test_run_case_returns_all_three_retrieval_strategies() -> None:
    dense_results = [_result("d1")]
    sparse_results = [_result("s1")]
    hybrid_results = [_result("h1")]
    generation_result = GenerationResult(answer="answer", citations=[], sources=[])

    run = await run_case(
        "question",
        _dense(dense_results),
        _sparse(sparse_results),
        _hybrid(hybrid_results),
        _reranker([]),
        _generator(generation_result),
    )

    assert run.dense_results == dense_results
    assert run.sparse_results == sparse_results
    assert run.hybrid_results == hybrid_results
    assert run.result == generation_result


async def test_run_case_reranks_hybrid_results_trimmed_to_rerank_input_top_k() -> None:
    hybrid_results = [_result("h1"), _result("h2"), _result("h3")]
    reranker = _reranker([])
    settings = Settings(RERANK_INPUT_TOP_K=2)

    await run_case(
        "question",
        _dense([]),
        _sparse([]),
        _hybrid(hybrid_results),
        reranker,
        _generator(GenerationResult(answer="a", citations=[], sources=[])),
        settings=settings,
    )

    reranker.rerank.assert_called_once_with("question", hybrid_results[:2])


async def test_run_case_generates_from_reranked_results() -> None:
    reranked = [_result("r1")]
    generator = _generator(GenerationResult(answer="a", citations=[], sources=[]))

    await run_case("question", _dense([]), _sparse([]), _hybrid([]), _reranker(reranked), generator)

    generator.generate.assert_called_once_with("question", reranked)


async def test_run_case_reports_non_negative_latencies() -> None:
    run = await run_case(
        "question",
        _dense([]),
        _sparse([]),
        _hybrid([]),
        _reranker([]),
        _generator(GenerationResult(answer="a", citations=[], sources=[])),
    )

    assert run.retrieval_latency_s >= 0.0
    assert run.reranker_latency_s >= 0.0
    assert run.generation_latency_s >= 0.0
    assert run.end_to_end_latency_s == (
        run.retrieval_latency_s + run.reranker_latency_s + run.generation_latency_s
    )
