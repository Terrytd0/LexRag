"""Pydantic request/response contracts for the API layer: `UploadResponse`/
`DuplicateUploadResponse` (`upload.py`), `DocumentListResponse`/
`DeleteDocumentResponse` (`documents.py`), and `QueryRequest`/`QueryResponse`
(`query.py`). Internal domain objects are never returned directly -- every
boundary crosses through a schema defined here.
"""
