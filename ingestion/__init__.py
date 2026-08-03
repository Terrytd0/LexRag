"""Document ingestion pipeline: PDF loading, chunking, and metadata extraction
feeding MongoDB (raw text + chunks). Qdrant and Elasticsearch indexers plug into
`ingestion.pipeline.IngestionPipeline` as `ChunkIndexer` implementations starting
Sprint 5 Day 3, without changing the orchestration logic here.
"""
