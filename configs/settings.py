"""Centralized, environment-driven application settings.

All configuration flows through this module (12-factor). Nothing outside
`configs/` should read `os.environ` directly, and no secrets are hardcoded --
values are supplied via `.env` (see `.env.example`) or real environment
variables in deployed environments.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- MongoDB (document + chunk metadata store) ---
    mongo_uri: str = Field(default="mongodb://localhost:27017", alias="MONGO_URI")
    mongo_db_name: str = Field(default="lexrag", alias="MONGO_DB_NAME")

    # --- Qdrant (vector store) ---
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="lexrag_chunks", alias="QDRANT_COLLECTION")

    # --- Elasticsearch (BM25 keyword store) ---
    elasticsearch_url: str = Field(default="http://localhost:9200", alias="ELASTICSEARCH_URL")
    elasticsearch_index: str = Field(default="lexrag_chunks", alias="ELASTICSEARCH_INDEX")

    # --- Embeddings ---
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=768, alias="EMBEDDING_DIMENSIONS")

    # --- Chunking ---
    chunk_size_tokens: int = Field(default=512, alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=64, alias="CHUNK_OVERLAP_TOKENS")

    # --- Hybrid retrieval ---
    rrf_k: int = Field(default=60, alias="RRF_K")
    retrieval_top_k: int = Field(default=50, alias="RETRIEVAL_TOP_K")
    rerank_top_k: int = Field(default=8, alias="RERANK_TOP_K")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )

    # --- Generation ---
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    generation_min_context_score: float = Field(
        default=0.35, alias="GENERATION_MIN_CONTEXT_SCORE"
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton -- Settings() is only constructed once per process."""
    return Settings()
