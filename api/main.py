"""FastAPI application entrypoint.

Run locally with: uv run uvicorn api.main:app --reload

Only a health check is wired up today (Sprint 5 Day 1). POST /upload and
POST /query are added on Day 4 once ingestion, retrieval, and generation
exist to back them (see docs/architecture.md).
"""

from __future__ import annotations

from fastapi import FastAPI

from configs.settings import get_settings

settings = get_settings()

app = FastAPI(
    title="LexRAG",
    description="Contract & case-law retrieval-augmented generation API.",
    version="0.1.0",
)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Liveness probe. Returns 200 with app metadata once the process is up."""
    return {"status": "ok", "environment": settings.environment}
