"""`POST /query`: `HybridRetriever` -> `CrossEncoderReranker` -> `PromptBuilder` ->
`GenerationService`, orchestrated by `generation.pipeline.QueryPipeline`. HTTP
concerns only -- the route parses the request, calls the pipeline, and shapes
the response (CLAUDE.md "Keep business logic out of API routes").
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, status

from api.dependencies import get_query_pipeline
from api.schemas.query import QueryRequest, QueryResponse
from generation.pipeline import QueryPipeline

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])

_REQUEST_EXAMPLES = {
    # Listed first (and shown by default in Swagger's request-body editor) so
    # document_ids is visible immediately, not just discoverable via the
    # "Examples" dropdown -- users kept missing it when global_search (no
    # document_ids at all) was the default.
    "single_document": {
        "summary": "Single-document search",
        "description": "Scope retrieval to one document's chunks only. Replace this "
        "placeholder id with a real one from GET /documents.",
        "value": {
            "question": "What is the termination notice period?",
            "document_ids": ["<doc_id_1>"],
        },
    },
    "multiple_documents": {
        "summary": "Multi-document search",
        "description": "Scope retrieval to several documents -- e.g. comparing a clause "
        "across contracts. Replace these placeholder ids with real ones from GET /documents.",
        "value": {
            "question": "What is the termination notice period?",
            "document_ids": ["<doc_id_1>", "<doc_id_2>"],
        },
    },
    "global_search": {
        "summary": "Global search (all documents)",
        "description": "Omit `document_ids` (or set it to `null`) to search the full corpus.",
        "value": {"question": "What is the termination notice period?"},
    },
}


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: Annotated[QueryRequest, Body(openapi_examples=_REQUEST_EXAMPLES)],
    pipeline: QueryPipeline = Depends(get_query_pipeline),
) -> QueryResponse:
    """Answer `request.question` from the indexed corpus, with citations or a refusal (FR-6).

    `request.document_ids`, if supplied, restricts retrieval to those
    documents (see the request examples in Swagger); omitted or `null`
    searches the full corpus.
    """
    logger.info(
        "query request received question_length=%d document_ids=%s",
        len(request.question),
        request.document_ids,
    )
    try:
        result = await pipeline.answer(request.question, document_ids=request.document_ids)
    except Exception as exc:
        logger.exception("query failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate an answer for this query.",
        ) from exc

    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        sources=result.sources,
        confidence=result.confidence,
        refused=result.refused,
    )
