"""Token-aware, deterministic chunker producing provenance-tagged domain Chunks.

Uses `tiktoken` for both counting and splitting, so `chunk_size_tokens`/
`chunk_overlap_tokens` (`configs.settings`) are enforced in the same unit the
embedding model will consume later (FR-2), rather than an approximate
character count.
"""

from __future__ import annotations

import hashlib

import tiktoken

from configs.settings import Settings, get_settings
from domain.chunk import Chunk

_ENCODING_NAME = "cl100k_base"
_UNKNOWN_SECTION = "unspecified"


def chunk_document(
    *,
    doc_id: str,
    filename: str,
    pages: list[str],
    settings: Settings | None = None,
) -> list[Chunk]:
    """Split a document's page texts into overlapping, token-bounded chunks.

    `pages` is one string per page in reading order (1-indexed page numbers
    are assigned from that order). Chunking is a deterministic sliding
    window over the token stream: the same `pages` input always produces the
    same chunks, in the same order, and a window is only ever emitted once
    (guards against emitting an identical chunk twice, e.g. from a boundary
    miscalculation).
    """
    settings = settings or get_settings()
    encoding = tiktoken.get_encoding(_ENCODING_NAME)

    tokens: list[int] = []
    token_pages: list[int] = []
    for page_number, page_text in enumerate(pages, start=1):
        page_tokens = encoding.encode(page_text)
        tokens.extend(page_tokens)
        token_pages.extend([page_number] * len(page_tokens))

    if not tokens:
        return []

    chunk_size = settings.chunk_size_tokens
    overlap = settings.chunk_overlap_tokens
    stride = chunk_size - overlap
    if stride <= 0:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    chunks: list[Chunk] = []
    seen_hashes: set[str] = set()
    chunk_index = 0
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        text = encoding.decode(chunk_tokens)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{chunk_index}",
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    text=text,
                    token_count=len(chunk_tokens),
                    page_number=token_pages[start],
                    section=_UNKNOWN_SECTION,
                    source_filename=filename,
                )
            )
            chunk_index += 1

        if end == len(tokens):
            break
        start += stride

    return chunks
