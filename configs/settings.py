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
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    # --- MongoDB (document + chunk metadata store) ---
    mongodb_uri: str = Field(default="mongodb://localhost:27017", alias="MONGODB_URI")
    mongodb_db_name: str = Field(default="lexrag", alias="MONGODB_DB_NAME")

    # --- Qdrant (vector store) ---
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="lexrag_chunks", alias="QDRANT_COLLECTION")
    # One of Qdrant's Distance enum values ("Cosine", "Dot", "Euclid"), matched
    # case-insensitively -- Cosine is the right default for normalized
    # sentence-transformers embeddings like bge-m3.
    qdrant_distance_metric: str = Field(default="Cosine", alias="QDRANT_DISTANCE_METRIC")

    # --- Elasticsearch (BM25 keyword store) ---
    elasticsearch_url: str = Field(default="http://localhost:9200", alias="ELASTICSEARCH_URL")
    elasticsearch_username: str = Field(default="", alias="ELASTICSEARCH_USERNAME")
    elasticsearch_password: str = Field(default="", alias="ELASTICSEARCH_PASSWORD")
    elasticsearch_index: str = Field(default="lexrag_chunks", alias="ELASTICSEARCH_INDEX")

    # --- Embeddings ---
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1024, alias="EMBEDDING_DIMENSIONS")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")

    # --- Chunking ---
    chunk_size_tokens: int = Field(default=512, alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=64, alias="CHUNK_OVERLAP_TOKENS")

    # --- Hybrid retrieval ---
    rrf_k: int = Field(default=60, alias="RRF_K")
    retrieval_top_k: int = Field(default=50, alias="RETRIEVAL_TOP_K")
    # Candidates taken from the RRF-fused ranking before cross-encoder
    # reranking -- distinct from rerank_top_k (which trims the reranker's
    # *output*). The cross-encoder is the dominant cost in query latency
    # (~2.4s/candidate observed on CPU), so this bounds how many candidates
    # pay that cost, independent of retrieval_top_k/RRF recall. Reduced from
    # 20 to 12 on Sprint 5 Day 6 after validating against the real golden set
    # (scripts/rerank_input_topk_validation.py): 0/22 positive cases lost
    # their expected document from the reranked top rerank_top_k at either
    # value, for a ~38% reranker latency cut -- see
    # docs/experiments/evaluation_notes_day6.md.
    rerank_input_top_k: int = Field(default=12, alias="RERANK_INPUT_TOP_K")
    rerank_top_k: int = Field(default=8, alias="RERANK_TOP_K")
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANK_MODEL")
    rerank_batch_size: int = Field(default=16, alias="RERANK_BATCH_SIZE")
    # Inference backend for the cross-encoder ("torch", "onnx", or "openvino") --
    # sentence-transformers loads the same model weights through whichever
    # backend is named here; changing this never changes rerank quality, only
    # execution speed. See docs/adr/001-reranker-onnx-backend.md for the
    # measured comparison behind the current default.
    rerank_backend: str = Field(default="torch", alias="RERANK_BACKEND")

    # --- Generation ---
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4.1-mini", alias="LLM_MODEL")
    # Alias tracks the OpenAI SDK's own env var convention (see .env.example) --
    # if LLM_PROVIDER ever moves to a non-OpenAI-compatible provider, this is
    # the credential generation/ will read regardless of the literal name.
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    generation_min_context_score: float = Field(default=0.35, alias="GENERATION_MIN_CONTEXT_SCORE")


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton -- Settings() is only constructed once per process."""
    return Settings()
