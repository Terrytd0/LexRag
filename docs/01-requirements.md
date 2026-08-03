# LexRAG — Requirements

**Status:** Approved for Sprint 5, Day 1
**Client:** Blackwell & Voss LLP (portfolio engagement)
**Author:** Terry Nyirenda

## 1. Project Overview

Blackwell & Voss's paralegals spend hours per matter manually searching
thousands of pages of contracts and prior case filings for relevant clauses
(indemnity, termination, force majeure, limitation of liability) and
precedent. Their current tooling is keyword-only full-text search, which
misses semantically relevant passages that don't share vocabulary with the
query. Pointing a generic LLM chat interface at the corpus is worse: it
answers confidently without evidence, and legal answers without traceable
citations are unusable and risky.

LexRAG is a retrieval-augmented generation platform purpose-built for this
problem: paralegals upload contracts and case-law PDFs, the system ingests
and indexes them, and a hybrid retrieval + reranking + citation-grounded
generation pipeline answers natural-language questions with answers that
are traceable to source passages and that **refuse to answer** when the
corpus does not contain sufficient evidence. Retrieval and generation
quality are treated as tested, measured properties of the system — gated in
CI — not vibes-based tuning.

This document defines what "done" means for the platform. Everything below
is written to be checkable: given a specific run of the system, a reader
should be able to determine pass/fail for each requirement without judgment
calls.

## 2. User Stories

1. **As a paralegal**, I want to upload a batch of contract PDFs and have
   them searchable within minutes, so that I don't have to manually read
   every document before starting a matter.

2. **As a paralegal**, I want to ask a natural-language question (e.g. "What
   are the termination notice periods across our vendor contracts?") and
   receive an answer with pinpoint citations (document, section, page), so
   that I can verify the answer against the source instead of trusting it
   blindly.

3. **As a paralegal**, I want the system to explicitly tell me when it
   cannot find sufficient evidence to answer, rather than guessing, so that
   I never mistake a hallucinated answer for a researched one.

4. **As a supervising attorney**, I want confidence that every generated
   answer is grounded only in retrieved passages (not the model's general
   knowledge), so that the tool cannot introduce fabricated legal claims
   into casework.

5. **As the engineering owner**, I want an automated evaluation gate that
   fails a pull request when retrieval or generation quality regresses, so
   that changes to chunking, retrieval, or prompts can't silently degrade
   answer quality in production.

## 3. MVP Scope

In scope for Sprint 5 (v1.0):

- PDF upload and ingestion (single-file and small-batch) with text
  extraction, configurable chunking, and provenance metadata
  (`doc_id`, `source`, `section`, `page`, `chunk_index`).
- Metadata and raw-document persistence in MongoDB.
- Dense vector indexing in Qdrant using a sentence-embedding model.
- Sparse/lexical indexing in Elasticsearch (BM25).
- Hybrid retrieval: parallel vector + BM25 search merged via Reciprocal
  Rank Fusion (RRF).
- Cross-encoder reranking of the fused candidate set.
- Citation-grounded answer generation with explicit refusal when retrieved
  context is insufficient.
- `POST /upload` and `POST /query` REST endpoints (FastAPI) with a Pydantic
  response contract (`answer`, `citations`, `sources`).
- A golden evaluation dataset (25–30 Q/A pairs, including negative/refusal
  cases) and an automated harness reporting recall@K, precision@K, and
  answer faithfulness.
- CI pipeline: lint, tests, Docker build, and an evaluation quality gate
  that fails the build below defined thresholds.
- Full local stack runnable via `docker compose up`.

## 4. Stretch Goals

Not required for v1.0; pursued only after MVP acceptance criteria pass:

- Pinecone vector-store adapter behind the same retrieval interface as
  Qdrant, to demonstrate a swappable-backend design and benchmark the two.
- Dedicated termination-clause detection (structured extraction beyond
  free-text Q/A).
- Multi-file upload with progress streaming.
- Answer streaming (token-by-token) over the `/query` endpoint.
- Query rewriting / decomposition for multi-part legal questions.
- Role-based access control per matter/client (multi-tenant document
  isolation).

## 5. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-1 | The system shall accept PDF file uploads via `POST /upload` and return a `doc_id` and ingestion status. |
| FR-2 | The system shall extract text from uploaded PDFs and split it into chunks using a configurable size (default 512 tokens) and overlap (default 64 tokens). |
| FR-3 | Every chunk shall be persisted with provenance metadata: source document ID, section/heading (when detectable), page number, and chunk index. |
| FR-4 | The system shall generate a dense embedding for every chunk and upsert it into Qdrant. |
| FR-5 | The system shall index every chunk's text into Elasticsearch for BM25 keyword search. |
| FR-6 | The system shall accept natural-language questions via `POST /query` and return an answer, a list of citations (document, section/page, chunk ID), and the underlying source snippets. |
| FR-7 | Retrieval shall run vector search and BM25 search in parallel and merge results using Reciprocal Rank Fusion with a configurable `k` constant. |
| FR-8 | The fused candidate set shall be reranked with a cross-encoder model before being passed to generation. |
| FR-9 | Generation shall be constrained to cite only retrieved passages; the prompt and post-generation validation shall reject/flag citations that don't map to a retrieved chunk. |
| FR-10 | The system shall refuse to answer (return a explicit "insufficient evidence" response, not a best-effort guess) when no retrieved chunk clears the minimum relevance threshold. |
| FR-11 | The evaluation harness shall run the full golden dataset end-to-end and report recall@K, precision@K, and faithfulness per query and in aggregate. |
| FR-12 | CI shall fail the build if aggregate evaluation metrics fall below the thresholds defined in Section 7. |

## 6. Non-Functional Requirements

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-1 | Performance | P95 query latency (retrieval + rerank + generation) ≤ 6 seconds for a corpus of ≤ 10,000 chunks, measured locally against the Docker Compose stack. |
| NFR-2 | Performance | P95 ingestion time ≤ 30 seconds per 50-page PDF, from upload to queryable (indexed in both Qdrant and Elasticsearch). |
| NFR-3 | Reliability | If either Qdrant or Elasticsearch is unreachable, `/query` shall degrade to the remaining retrieval source and return results with a `degraded: true` flag, rather than returning a 5xx. |
| NFR-4 | Reliability | Ingestion is atomic per document: a failure partway through (e.g. embedding failure) shall not leave partial chunks queryable in one store but not the other. |
| NFR-5 | Testability | The full test suite (unit) runs without any external service dependency and completes in under 60 seconds. |
| NFR-6 | Observability | Every request to `/upload` and `/query` is logged at INFO level with a correlation ID, latency, and (for `/query`) the number of chunks retrieved and reranked. |
| NFR-7 | Maintainability | Retrieval backends (vector store, keyword store) are accessed through an interface, not called directly from route handlers, so a backend (e.g. Qdrant → Pinecone) can be swapped without changing `retrieval/fusion` or `generation`. |
| NFR-8 | Security | No LLM or database credentials are hardcoded; all secrets are sourced from environment variables and `.env` is gitignored. |
| NFR-9 | Portability | The full stack (API + MongoDB + Qdrant + Elasticsearch) starts with a single `docker compose up` and requires no manual post-start configuration for the golden dataset to be queryable. |
| NFR-10 | Code quality | `ruff check`, `mypy`, and `pytest` all pass in CI on every pull request; no merge to `main` bypasses the quality gate. |

## 7. Measurable Acceptance Criteria

The platform is considered to meet its v1.0 bar when **all** of the
following hold against the golden dataset (25–30 Q/A pairs, defined Day 5):

1. **Citation accuracy ≥ 90%** — at least 90% of golden-set queries produce
   an answer citing the correct source document/section (exact match
   against the golden `expected_doc_id` / `expected_section`).
2. **Retrieval recall@10 ≥ 0.85** — for queries with a known relevant
   chunk, that chunk appears in the top 10 fused-and-reranked results at
   least 85% of the time.
3. **Retrieval precision@5 ≥ 0.70** — averaged across the golden set, at
   least 70% of the top 5 returned chunks are judged relevant.
4. **Faithfulness ≥ 0.90** — RAGAS/DeepEval faithfulness score (generated
   claims supported by retrieved context) averages ≥ 0.90 across the golden
   set's answerable queries.
5. **100% refusal on negative cases** — every adversarial/out-of-scope
   golden query (e.g. questions with no supporting evidence in the corpus)
   is refused, not answered.
6. **CI quality gate is provably enforced** — a deliberately degraded
   retrieval configuration (documented in `docs/experiments/`) causes CI to
   fail; reverting restores a passing build. This is demonstrated once,
   on the record, before v1.0 is tagged.
7. **`docker compose up` reproducibility** — a clean checkout starts the
   full stack and successfully answers a golden-set query with zero manual
   steps beyond providing an LLM API key.

## 8. Success Metrics

Tracked and reported in the final README (Day 6):

- **Answer citation accuracy** (Section 7.1) — headline quality metric.
- **Recall@K / Precision@K** (Section 7.2–7.3) — retrieval-layer quality,
  reported per retrieval strategy (vector-only, BM25-only, hybrid+RRF,
  hybrid+RRF+rerank) to make the value of each pipeline stage explicit.
- **Faithfulness / answer relevancy** (Section 7.4, via RAGAS/DeepEval) —
  generation-layer groundedness.
- **Refusal precision** — percentage of negative/adversarial golden
  queries correctly refused, and (as a guardrail against an over-cautious
  system) percentage of answerable golden queries that are *not*
  incorrectly refused.
- **P95 query latency** (NFR-1) — production-readiness signal.
- **CI gate reliability** — the degrade/restore demonstration (Section 7.6)
  as evidence the gate is load-bearing, not decorative.
