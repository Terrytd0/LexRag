"""Pydantic request/response contracts for the API layer.

Planned (Day 4): UploadResponse, QueryRequest, QueryResponse (answer, citations,
sources), and shared error models. Internal domain objects are never returned
directly -- every boundary crosses through a schema defined here.
"""
