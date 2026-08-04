"""Citation-grounded answer generation: prompt construction, LLM invocation, and
refusal logic for insufficient-evidence queries.

Consumes `domain.retrieval.RetrievalResult` produced by
`retrieval.hybrid.HybridRetriever` and `retrieval.reranker.CrossEncoderReranker`
-- this package never imports MongoDB, Qdrant, or Elasticsearch clients
directly (`docs/architecture.md` §2.9, NFR-7).
"""
