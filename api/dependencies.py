"""FastAPI dependency-injection wiring: constructs and caches the store clients,
retrieval/generation services, and pipelines that route handlers receive via
`Depends()` (CLAUDE.md "Dependency injection"). No business logic here -- this
module only wires together objects defined in `ingestion/`, `retrieval/`, and
`generation/`, mirroring the manual wiring in `scripts/seed_corpus.py`.
"""

from __future__ import annotations

from functools import lru_cache

from generation.generator import GenerationService
from generation.pipeline import QueryPipeline
from generation.prompts import PromptBuilder
from generation.providers import get_llm_provider
from ingestion.pipeline import IngestionPipeline
from ingestion.repository import DocumentRepository, get_mongo_client
from retrieval.dense import DenseRetriever
from retrieval.embedding import get_embedding_service
from retrieval.hybrid import HybridRetriever
from retrieval.keyword_store.elasticsearch_store import (
    ElasticsearchKeywordStore,
    get_elasticsearch_client,
)
from retrieval.reranker.cross_encoder import CrossEncoderReranker, get_reranker
from retrieval.sparse import SparseRetriever
from retrieval.vector_store.qdrant_store import QdrantVectorStore, get_qdrant_client


@lru_cache
def get_document_repository() -> DocumentRepository:
    """Process-wide `DocumentRepository` singleton over the shared Mongo client."""
    return DocumentRepository(get_mongo_client())


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    """Process-wide `QdrantVectorStore` singleton, sharing the cached embedding model."""
    return QdrantVectorStore(get_qdrant_client(), get_embedding_service())


@lru_cache
def get_keyword_store() -> ElasticsearchKeywordStore:
    """Process-wide `ElasticsearchKeywordStore` singleton."""
    return ElasticsearchKeywordStore(get_elasticsearch_client())


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    """`IngestionPipeline` dual-writing to Qdrant and Elasticsearch alongside Mongo."""
    return IngestionPipeline(
        get_document_repository(),
        indexers=[get_vector_store(), get_keyword_store()],
    )


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    """`HybridRetriever` combining dense (Qdrant) and sparse (Elasticsearch) search."""
    dense = DenseRetriever(get_vector_store(), get_embedding_service())
    sparse = SparseRetriever(get_keyword_store())
    return HybridRetriever(dense, sparse)


def get_cross_encoder_reranker() -> CrossEncoderReranker:
    """The process-wide `CrossEncoderReranker` singleton (`retrieval.reranker`)."""
    return get_reranker()


@lru_cache
def get_generation_service() -> GenerationService:
    """`GenerationService` wired to the configured LLM provider and prompt builder."""
    return GenerationService(get_llm_provider(), PromptBuilder())


@lru_cache
def get_query_pipeline() -> QueryPipeline:
    """`QueryPipeline` composing retrieval, reranking, and generation for `POST /query`."""
    return QueryPipeline(
        get_hybrid_retriever(), get_cross_encoder_reranker(), get_generation_service()
    )
