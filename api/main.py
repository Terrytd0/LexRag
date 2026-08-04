"""FastAPI application entrypoint.

Run locally with: uv run uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes import documents, query, upload
from configs.logging import configure_logging
from configs.settings import get_settings

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="LexRAG",
    description="Contract & case-law retrieval-augmented generation API.",
    version="0.1.0",
)

app.include_router(upload.router)
app.include_router(documents.router)
app.include_router(query.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Backstop error handler: turns any unexpected failure into a structured 500
    response, never a raw stack trace (CLAUDE.md "Error handling").
    """
    logger.exception("unhandled exception path=%s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Liveness probe. Returns 200 with app metadata once the process is up."""
    return {"status": "ok", "environment": settings.environment}
