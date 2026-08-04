"""FastAPI application layer: routers, request/response schemas, and dependency wiring.

HTTP concerns only -- routes call into ingestion/retrieval/generation services and
translate results to/from Pydantic schemas. No business logic lives here. Populated
Sprint 5 Day 4 (`POST /upload`, `POST /query`) and extended in the same day's
production-hardening pass (`GET /documents`, `DELETE /documents/{doc_id}`,
document-scoped queries, duplicate-upload detection).
"""
