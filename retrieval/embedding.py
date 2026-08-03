"""Dense embedding generation, shared by ingestion (chunk indexing) and retrieval
(query embedding) so the model is loaded once and both paths stay numerically
consistent with each other.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from configs.settings import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Loads one `sentence-transformers` model and encodes text into dense vectors.

    Encoding is deterministic: `SentenceTransformer.encode` runs the model in
    eval mode with no sampling, so the same text always yields the same
    vector, in-process or across processes/runs.
    """

    def __init__(self, model_name: str, batch_size: int) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed `texts` in batches of `batch_size`. Returns one vector per input, in order."""
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        logger.info(
            "embedding generated model=%s text_count=%d batch_size=%d",
            self._model_name,
            len(texts),
            self._batch_size,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Convenience wrapper around `embed_texts`."""
        return self.embed_texts([text])[0]


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Process-wide EmbeddingService singleton -- the model loads once per process."""
    settings = get_settings()
    return EmbeddingService(
        model_name=settings.embedding_model, batch_size=settings.embedding_batch_size
    )
