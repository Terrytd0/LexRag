"""FastAPI application layer: routers, request/response schemas, and dependency wiring.

HTTP concerns only -- routes call into ingestion/retrieval/generation services and
translate results to/from Pydantic schemas. No business logic lives here. Populated
starting Sprint 5 Day 4.
"""
