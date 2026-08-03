"""Elasticsearch client wrapper: index mapping/analyzer setup and BM25 search.

Satisfies `ingestion.pipeline.ChunkIndexer` so it plugs into the existing
ingestion pipeline as just another registered indexer (`docs/architecture.md`
§2.4).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from configs.settings import Settings, get_settings
from domain.chunk import Chunk

logger = logging.getLogger(__name__)

_ANALYZER_NAME = "legal_text_analyzer"

# Standard tokenizer + lowercase only, deliberately no stopword filter: legal
# text is full of exact-match-critical tokens (defined terms, clause numbers,
# statute citations) that a generic stopword list can damage, and the
# standard tokenizer already keeps digit.digit sequences like "8.3" as one
# token per Unicode UAX#29 rules (`docs/architecture.md` §2.4).
_INDEX_SETTINGS: dict[str, Any] = {
    "analysis": {
        "analyzer": {
            _ANALYZER_NAME: {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase"],
            }
        }
    }
}

_INDEX_MAPPINGS: dict[str, Any] = {
    "properties": {
        "chunk_id": {"type": "keyword"},
        "doc_id": {"type": "keyword"},
        "chunk_index": {"type": "integer"},
        "text": {"type": "text", "analyzer": _ANALYZER_NAME, "similarity": "BM25"},
        "token_count": {"type": "integer"},
        "page_number": {"type": "integer"},
        "section": {"type": "keyword"},
        "source_filename": {"type": "keyword"},
    }
}


@lru_cache
def get_elasticsearch_client() -> Elasticsearch:
    """Process-wide Elasticsearch client singleton, mirroring `get_mongo_client`."""
    settings = get_settings()
    auth = (
        (settings.elasticsearch_username, settings.elasticsearch_password)
        if settings.elasticsearch_username
        else None
    )
    return Elasticsearch(settings.elasticsearch_url, basic_auth=auth)


class ElasticsearchKeywordStore:
    """Lexical (BM25) storage and search over chunk text.

    Index creation is lazy and idempotent, mirroring `QdrantVectorStore`: the
    first `index_chunks` or `search` call ensures the index exists with the
    legal-text analyzer and BM25 mapping, so a fresh Elasticsearch instance
    needs no separate provisioning step.
    """

    def __init__(self, client: Elasticsearch, settings: Settings | None = None) -> None:
        self._client = client
        self._settings = settings or get_settings()
        self._index = self._settings.elasticsearch_index
        self._ensured = False

    def _ensure_index(self) -> None:
        if self._ensured:
            return
        if not self._client.indices.exists(index=self._index):
            self._client.indices.create(
                index=self._index, settings=_INDEX_SETTINGS, mappings=_INDEX_MAPPINGS
            )
            logger.info("elasticsearch index created index=%s", self._index)
        self._ensured = True

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Bulk-index `chunks`, keyed by `chunk_id`, preserving all chunk metadata."""
        if not chunks:
            return
        self._ensure_index()

        actions = [
            {
                "_index": self._index,
                "_id": chunk.chunk_id,
                "_source": chunk.model_dump(mode="json"),
            }
            for chunk in chunks
        ]
        bulk(self._client, actions)
        logger.info(
            "elasticsearch indexing complete index=%s chunk_count=%d",
            self._index,
            len(chunks),
        )

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        """Return up to `top_k` BM25 matches for `query` against chunk text, best first."""
        self._ensure_index()
        response = self._client.search(
            index=self._index,
            query={"match": {"text": query}},
            size=top_k,
        )
        return [
            (Chunk.model_validate(hit["_source"]), hit["_score"])
            for hit in response["hits"]["hits"]
        ]
