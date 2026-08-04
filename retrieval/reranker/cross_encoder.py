"""Cross-encoder reranking: refine the RRF-fused candidate set into a precise
final ranking before generation (`docs/architecture.md` §2.7, FR-8).

Operates only on `RetrievalResult` objects -- never queries Qdrant,
Elasticsearch, or MongoDB directly.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder

from configs.settings import get_settings
from domain.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks `RetrievalResult`s with a cross-encoder, jointly scoring `(query, chunk)` pairs.

    Scores are passed through a sigmoid so `rerank_score` is bounded in
    (0, 1) regardless of the underlying model's raw logit scale -- this is
    what makes `Settings.generation_min_context_score` a meaningful,
    model-independent refusal threshold (FR-10). Output is sorted by
    `rerank_score` descending with `chunk_id` as a deterministic tiebreak,
    mirroring `retrieval.fusion.rrf.reciprocal_rank_fusion`.
    """

    def __init__(
        self,
        model_name: str,
        top_k: int,
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._batch_size = batch_size
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int | None = None
    ) -> list[RetrievalResult]:
        """Score `results` against `query` and return the top `top_k`, best first."""
        if not results:
            return []

        start = time.monotonic()
        pairs = [(query, result.chunk.text) for result in results]
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            activation_fn=torch.nn.Sigmoid(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        scored = [
            result.model_copy(update={"rerank_score": float(score)})
            for result, score in zip(results, scores, strict=True)
        ]
        ranked = sorted(scored, key=lambda r: (-(r.rerank_score or 0.0), r.chunk.chunk_id))
        limit = top_k or self._top_k
        reranked = ranked[:limit]

        duration = time.monotonic() - start
        logger.info(
            "reranking complete model=%s candidate_count=%d result_count=%d duration=%.3f",
            self._model_name,
            len(results),
            len(reranked),
            duration,
        )
        return reranked


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    """Process-wide CrossEncoderReranker singleton -- the model loads once per process."""
    settings = get_settings()
    return CrossEncoderReranker(
        model_name=settings.rerank_model,
        top_k=settings.rerank_top_k,
        batch_size=settings.rerank_batch_size,
    )
