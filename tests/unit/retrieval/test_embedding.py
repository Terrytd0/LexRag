from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from retrieval.embedding import EmbeddingService, get_embedding_service


class _FakeModel:
    """Stand-in for SentenceTransformer: deterministic, length-based "embeddings"."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.encode_calls: list[dict[str, Any]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.encode_calls.append({"texts": texts, **kwargs})
        return np.array([[float(len(text))] * self.dim for text in texts])


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> _FakeModel:
    model = _FakeModel()
    monkeypatch.setattr("retrieval.embedding.SentenceTransformer", MagicMock(return_value=model))
    return model


def test_embed_texts_returns_one_vector_per_input(fake_model: _FakeModel) -> None:
    service = EmbeddingService(model_name="fake-model", batch_size=8)

    vectors = service.embed_texts(["hello", "worldly!"])

    assert vectors == [[5.0] * fake_model.dim, [8.0] * fake_model.dim]


def test_embed_texts_empty_list_short_circuits(fake_model: _FakeModel) -> None:
    service = EmbeddingService(model_name="fake-model", batch_size=8)

    assert service.embed_texts([]) == []
    assert fake_model.encode_calls == []


def test_embed_texts_is_deterministic(fake_model: _FakeModel) -> None:
    service = EmbeddingService(model_name="fake-model", batch_size=8)

    assert service.embed_texts(["same text"]) == service.embed_texts(["same text"])


def test_embed_query_returns_a_single_vector(fake_model: _FakeModel) -> None:
    service = EmbeddingService(model_name="fake-model", batch_size=8)

    assert service.embed_query("hello") == [5.0] * fake_model.dim


def test_embed_texts_passes_configured_batch_size(fake_model: _FakeModel) -> None:
    service = EmbeddingService(model_name="fake-model", batch_size=16)

    service.embed_texts(["a", "b"])

    assert fake_model.encode_calls[0]["batch_size"] == 16


def test_get_embedding_service_returns_a_cached_singleton(fake_model: _FakeModel) -> None:
    get_embedding_service.cache_clear()
    try:
        assert get_embedding_service() is get_embedding_service()
    finally:
        get_embedding_service.cache_clear()
