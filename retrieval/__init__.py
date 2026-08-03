"""Hybrid retrieval: parallel Qdrant (vector) + Elasticsearch (BM25) search, merged
via Reciprocal Rank Fusion (Sprint 5 Day 3) and refined by a cross-encoder
reranker (Day 4). `HybridRetriever` (`retrieval.hybrid`) is the package's public
interface -- callers outside `retrieval/` should depend on it and on
`domain.retrieval.RetrievalResult`, never on `vector_store`/`keyword_store`
internals or their storage clients directly (NFR-7).
"""
