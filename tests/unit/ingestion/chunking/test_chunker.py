from __future__ import annotations

import pytest
import tiktoken

from configs.settings import Settings
from domain.chunk import Chunk
from ingestion.chunking.chunker import chunk_document

_ENCODING = tiktoken.get_encoding("cl100k_base")
_SOURCE = (
    "In consideration of the mutual covenants and promises contained in this "
    "Agreement, the parties agree as follows: this Agreement shall commence on "
    "the Effective Date and continue until terminated in accordance with the "
    "provisions set forth herein, including but not limited to termination for "
    "convenience, termination for cause, indemnification obligations, and "
    "limitation of liability clauses that survive termination of this Agreement "
    "between the parties hereto."
) * 3


def _settings(chunk_size: int, overlap: int) -> Settings:
    # Explicit init kwargs take priority over any local .env/env vars, so this
    # is isolated from whatever a developer's own .env happens to contain.
    return Settings(CHUNK_SIZE_TOKENS=chunk_size, CHUNK_OVERLAP_TOKENS=overlap)


def _text_with_token_count(n: int) -> str:
    """Real English text truncated to exactly `n` tiktoken tokens."""
    tokens = _ENCODING.encode(_SOURCE)[:n]
    assert len(tokens) == n
    return _ENCODING.decode(tokens)


def test_chunk_count_matches_sliding_window_arithmetic() -> None:
    # 25 tokens, chunk_size=10, overlap=3 (stride=7) -> windows at 0, 7, 14, 21.
    settings = _settings(chunk_size=10, overlap=3)
    pages = [_text_with_token_count(25)]

    chunks = chunk_document(doc_id="doc-1", filename="a.pdf", pages=pages, settings=settings)

    assert len(chunks) == 4
    assert [c.token_count for c in chunks] == [10, 10, 10, 4]


def test_overlap_correctness_between_consecutive_chunks() -> None:
    settings = _settings(chunk_size=10, overlap=3)
    pages = [_text_with_token_count(25)]

    chunks = chunk_document(doc_id="doc-1", filename="a.pdf", pages=pages, settings=settings)

    for earlier, later in zip(chunks, chunks[1:], strict=False):
        earlier_tokens = _ENCODING.encode(earlier.text)
        later_tokens = _ENCODING.encode(later.text)
        assert earlier_tokens[-3:] == later_tokens[: len(earlier_tokens[-3:])]


def test_deterministic_output() -> None:
    settings = _settings(chunk_size=10, overlap=3)
    pages = [_text_with_token_count(25)]

    first_run = chunk_document(doc_id="doc-1", filename="a.pdf", pages=pages, settings=settings)
    second_run = chunk_document(doc_id="doc-1", filename="a.pdf", pages=pages, settings=settings)

    assert first_run == second_run


def test_metadata_correctness_across_pages() -> None:
    settings = _settings(chunk_size=50, overlap=5)
    page_one = _text_with_token_count(20)
    page_two = _text_with_token_count(20)

    chunks = chunk_document(
        doc_id="doc-42", filename="contract.pdf", pages=[page_one, page_two], settings=settings
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
    assert chunk.chunk_id == "doc-42:0"
    assert chunk.doc_id == "doc-42"
    assert chunk.chunk_index == 0
    assert chunk.token_count == 40
    assert chunk.page_number == 1
    assert chunk.section == "unspecified"
    assert chunk.source_filename == "contract.pdf"


def test_chunk_index_is_sequential_and_preserves_order() -> None:
    settings = _settings(chunk_size=10, overlap=3)
    pages = [_text_with_token_count(25)]

    chunks = chunk_document(doc_id="doc-1", filename="a.pdf", pages=pages, settings=settings)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_page_number_reflects_first_page_with_content() -> None:
    settings = _settings(chunk_size=100, overlap=10)
    page_one = _text_with_token_count(10)
    page_two = _text_with_token_count(10)

    chunks = chunk_document(
        doc_id="doc-1", filename="a.pdf", pages=[page_one, "", page_two], settings=settings
    )

    assert len(chunks) == 1
    assert chunks[0].page_number == 1


def test_no_duplicate_chunks_for_repetitive_content() -> None:
    # A single token repeated makes every same-length window byte-identical --
    # the chunker must collapse them instead of emitting near-duplicate chunks.
    settings = _settings(chunk_size=5, overlap=2)
    pages = [" the" * 20]

    chunks = chunk_document(doc_id="doc-1", filename="a.pdf", pages=pages, settings=settings)

    assert len(chunks) == 1
    texts = [c.text for c in chunks]
    assert len(texts) == len(set(texts))


def test_empty_pages_produce_no_chunks() -> None:
    settings = _settings(chunk_size=10, overlap=3)

    assert chunk_document(doc_id="doc-1", filename="a.pdf", pages=[], settings=settings) == []
    assert chunk_document(doc_id="doc-1", filename="a.pdf", pages=[""], settings=settings) == []


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    settings = _settings(chunk_size=10, overlap=10)

    with pytest.raises(ValueError, match="chunk_overlap_tokens"):
        chunk_document(
            doc_id="doc-1", filename="a.pdf", pages=[_text_with_token_count(25)], settings=settings
        )
