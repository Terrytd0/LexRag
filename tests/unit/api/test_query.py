from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_query_pipeline
from api.main import app
from domain.citation import Citation
from domain.generation import GenerationResult


@pytest.fixture(autouse=True)
def _default_pipeline_override() -> Iterator[None]:
    # FastAPI resolves `Depends()` sub-dependencies even for requests that fail
    # body validation, so every test needs an override in place -- otherwise a
    # "malformed request" test would construct the real QueryPipeline (and its
    # real embedding/reranker models) before ever reaching the route.
    app.dependency_overrides[get_query_pipeline] = lambda: MagicMock(name="pipeline")
    yield
    app.dependency_overrides.clear()


def _pipeline(result: GenerationResult, captured: dict[str, object] | None = None) -> MagicMock:
    pipeline = MagicMock(name="pipeline")

    async def _answer(question: str, document_ids: list[str] | None = None) -> GenerationResult:
        if captured is not None:
            captured["question"] = question
            captured["document_ids"] = document_ids
        return result

    pipeline.answer = _answer
    return pipeline


def test_query_success_returns_answer_and_citations(client: TestClient) -> None:
    citation = Citation(
        doc_id="doc-1",
        filename="contract.pdf",
        page_number=3,
        section="Section 8.3",
        chunk_id="doc-1:0",
        snippet="termination for convenience",
    )
    result = GenerationResult(
        answer="The contract may be terminated for convenience [1].",
        citations=[citation],
        sources=["contract.pdf"],
        confidence=0.87,
        refused=False,
    )
    app.dependency_overrides[get_query_pipeline] = lambda: _pipeline(result)

    response = client.post("/query", json={"question": "Can the contract be terminated?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == result.answer
    assert body["citations"][0]["chunk_id"] == "doc-1:0"
    assert body["sources"] == ["contract.pdf"]
    assert body["confidence"] == 0.87
    assert body["refused"] is False


def test_query_refusal_path(client: TestClient) -> None:
    result = GenerationResult(
        answer="I don't have enough evidence in the retrieved documents to answer this question.",
        citations=[],
        sources=[],
        confidence=None,
        refused=True,
    )
    app.dependency_overrides[get_query_pipeline] = lambda: _pipeline(result)

    response = client.post("/query", json={"question": "What is the meaning of life?"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["citations"] == []


def test_query_malformed_request_missing_question(client: TestClient) -> None:
    response = client.post("/query", json={})

    assert response.status_code == 422


def test_query_rejects_empty_question(client: TestClient) -> None:
    response = client.post("/query", json={"question": ""})

    assert response.status_code == 422


def test_query_pipeline_failure_returns_502(client: TestClient) -> None:
    pipeline = MagicMock(name="pipeline")

    async def _answer(question: str, document_ids: list[str] | None = None) -> GenerationResult:
        raise RuntimeError("Elasticsearch is unreachable")

    pipeline.answer = _answer
    app.dependency_overrides[get_query_pipeline] = lambda: pipeline

    response = client.post("/query", json={"question": "Any question"})

    assert response.status_code == 502
    assert "Elasticsearch" not in response.text


def test_query_without_document_ids_defaults_to_none(client: TestClient) -> None:
    result = GenerationResult(answer="answer", citations=[], sources=[])
    captured: dict[str, object] = {}
    app.dependency_overrides[get_query_pipeline] = lambda: _pipeline(result, captured)

    response = client.post("/query", json={"question": "Any question"})

    assert response.status_code == 200
    assert captured["document_ids"] is None


def test_query_with_single_document_id_is_passed_through(client: TestClient) -> None:
    result = GenerationResult(answer="answer", citations=[], sources=[])
    captured: dict[str, object] = {}
    app.dependency_overrides[get_query_pipeline] = lambda: _pipeline(result, captured)

    response = client.post("/query", json={"question": "Any question", "document_ids": ["doc-1"]})

    assert response.status_code == 200
    assert captured["document_ids"] == ["doc-1"]


def test_query_with_multiple_document_ids_is_passed_through(client: TestClient) -> None:
    result = GenerationResult(answer="answer", citations=[], sources=[])
    captured: dict[str, object] = {}
    app.dependency_overrides[get_query_pipeline] = lambda: _pipeline(result, captured)

    response = client.post(
        "/query",
        json={"question": "Any question", "document_ids": ["doc-1", "doc-2"]},
    )

    assert response.status_code == 200
    assert captured["document_ids"] == ["doc-1", "doc-2"]


def test_query_with_nonexistent_document_ids_still_succeeds_and_refuses(
    client: TestClient,
) -> None:
    # Filtering to doc_ids with no matching chunks is a real-server empty-retrieval
    # case (already covered by GenerationService's refusal logic tests) -- at the
    # API layer this just needs to not error out.
    result = GenerationResult(
        answer="I don't have enough evidence in the retrieved documents to answer this question.",
        citations=[],
        sources=[],
        refused=True,
    )
    captured: dict[str, object] = {}
    app.dependency_overrides[get_query_pipeline] = lambda: _pipeline(result, captured)

    response = client.post(
        "/query",
        json={"question": "Any question", "document_ids": ["does-not-exist"]},
    )

    assert response.status_code == 200
    assert response.json()["refused"] is True
    assert captured["document_ids"] == ["does-not-exist"]
