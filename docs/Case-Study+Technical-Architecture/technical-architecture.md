# LexRAG — Technical Architecture

A legal contract & case-law RAG platform — hybrid retrieval, cross-encoder reranking, citation-grounded refusal-aware generation, with quality enforced as a CI gate.

| | |
|---|---|
| **Project Type** | Self-directed portfolio project. Not a production deployment — the golden corpus is real, licensed contract text, not client data. |
| **Document** | Technical Architecture Reference |
| **Primary Source** | `docs/architecture.md` |
| **Prepared** | 2026 |
| **Status** | Sprint 5, Days 1–6 complete, plus Day 6 hardening |

`FastAPI` `Qdrant` `Elasticsearch` `RAGAS` `MongoDB` `CI Quality Gate`

> **How to read this document:** every capability is marked either *Implemented*, *Partially met*, or *Not met* against the project's own documented acceptance thresholds (`docs/01-requirements.md` §7) — including the two criteria the system currently fails, stated plainly rather than rounded up to "done."

## Contents

1. [System Architecture](#section-1--system-architecture)
2. [Component Responsibilities & Acceptance Status](#section-2--component-responsibilities--acceptance-status)
3. [Data Flow](#section-3--data-flow)
4. [Technology Choices](#section-4--technology-choices)
5. [Error Handling & Consistency](#section-5--error-handling--consistency)
6. [Testing & CI](#section-6--testing--ci)
7. [Known Limitations](#section-7--known-limitations)
8. [Resources](#resources)

## Section 1 · System Architecture

LexRAG ingests PDF contracts and case-law documents, chunks and embeds them, and fans the write out in parallel to three stores: MongoDB (document metadata and provenance), Qdrant (dense vector search), and Elasticsearch (BM25 keyword search). A query runs dense and keyword retrieval in parallel, fuses the results with Reciprocal Rank Fusion, reranks the fused candidates with a cross-encoder, then generates a citation-grounded answer — or refuses, if the retrieved evidence doesn't support one.

Retrieval and generation quality are measured against a 30-case golden dataset and enforced as a CI quality gate (FR-12), not tuned by eyeballing outputs.

**Pipeline Overview**

```
Ingestion            (ingestion/)   — PDF loading, chunking, provenance metadata; fan-out to
                                       Mongo + Qdrant + Elasticsearch; SHA-256 duplicate detection
        ↓
Hybrid Retrieval      (retrieval/)   — parallel dense (Qdrant) + BM25 (Elasticsearch); RRF fusion
        ↓
Reranking             (retrieval/reranker/) — cross-encoder (bge-reranker-v2-m3), capped by RERANK_INPUT_TOP_K
        ↓
Generation            (generation/)  — citation-grounded prompting; explicit refusal if evidence insufficient
        ↓
Evaluation            (evaluation/)  — golden-dataset harness; CI gate (FR-12) fails build on regression
```

**Architecture Diagram**

![LexRAG system architecture](assets/LexRAG.png)
*Fig. 1 — Ingestion pipeline, hybrid retrieval query pipeline, and evaluation pipeline gating CI. Editable source: `docs/architecture.mmd`.*

## Section 2 · Component Responsibilities & Acceptance Status

Status is measured against the project's own documented thresholds (`docs/01-requirements.md` §7), not a general impression of completeness.

`MET` clears the documented threshold · `PARTIAL` improved but not yet met · `NOT MET` below threshold

| Capability | Owns | Status |
|---|---|---|
| PDF ingestion + chunking + provenance | `ingestion/loaders/`, `ingestion/chunking/` | MET |
| Dual-write consistency (Mongo/Qdrant/ES) | `ingestion/pipeline.py` — not `status: ready` until every write is validated | MET |
| Duplicate detection (SHA-256) | `ingestion/repository.py` | MET |
| Hybrid retrieval (dense + BM25 + RRF) | `retrieval/hybrid.py`, `retrieval/fusion/` | MET — Recall@10 = 1.00 (≥ 0.85), Precision@5 = 0.88 (≥ 0.70) |
| Cross-encoder reranking | `retrieval/reranker/` — `bge-reranker-v2-m3` | MET |
| Citation-grounded generation | `generation/citations.py`, `generation/generator.py` | MET — Faithfulness = 0.98 (≥ 0.90) |
| Negative-case refusal (100% target) | `generation/prompts.py` — `LEGAL_RAG_V2` | **NOT MET** — 2 of 30 false acceptances remain |
| P95 query latency ≤ 6s (NFR-1) | Reranker is the dominant cost, CPU-bound | **NOT MET** — avg 26.3s, ~9x over budget |
| REST API (upload/query/documents) | `api/routes/` | MET |
| Golden-dataset evaluation harness | `evaluation/` — RAGAS + DeepEval | MET |
| CI evaluation quality gate (FR-12) | `.github/workflows/ci.yml` — `evaluation-gate` job | PARTIAL — real and wired, but manual-trigger + self-hosted only, not automatic on every push |
| Lint/type/test CI (`quality` job) | Ruff, mypy, pytest — every push/PR | MET |

> **The `confidence` field is not an answer-correctness score.** `domain.generation.GenerationResult.confidence` is the cross-encoder's relevance score for the single highest-ranked retrieved chunk — the same score refusal (FR-10) gates on — not a calibrated measure of whether the generated answer is correct. A follow-up measurement (`scripts/confidence_correlation.py`) found Pearson r ≈ 0 between `confidence` and answer correctness/faithfulness across the golden dataset, including one answer that scored 0.92 while being completely unfaithful. This is documented, current behavior — treat `confidence` as "how relevant was the best single piece of evidence," and rely on the `citations` list to judge whether an answer is trustworthy.

**Module Ownership Boundaries**

- `api/` — HTTP concerns only; parses requests, calls into ingestion/retrieval/generation, shapes responses
- `ingestion/` — PDF loading, chunking, the fan-out pipeline to all three stores
- `retrieval/` — vector store, keyword store, RRF fusion, reranker, each behind an interface
- `generation/` — citation-grounded prompting, provider-agnostic LLM interface, refusal logic
- `evaluation/` — golden-dataset runner + RAGAS/DeepEval metric wrappers
- `configs/` — all settings flow through here; nothing else reads `os.environ` directly

## Section 3 · Data Flow

**Ingestion**

1. `POST /upload` receives a PDF; its content hash is checked against existing documents — a byte-identical duplicate is skipped and the existing document returned.
2. The PDF is loaded, chunked with provenance metadata (document, section, page, chunk index), and embedded (`bge-m3`).
3. Chunks fan out to MongoDB (metadata), Qdrant (dense vectors), and Elasticsearch (BM25 index) in parallel.
4. The document is marked `status: ready` only once every store write is independently validated — a partial failure leaves it incomplete rather than falsely queryable.

**Query**

1. `POST /query` accepts a question and an optional `document_ids` scope.
2. Dense (Qdrant) and BM25 (Elasticsearch) retrieval run in parallel against the (optionally scoped) corpus.
3. Results merge via Reciprocal Rank Fusion, then the fused candidates (capped at `RERANK_INPUT_TOP_K`) are reranked by the cross-encoder.
4. The generator produces a citation-grounded answer from the top reranked passages, or refuses if they don't support one.
5. The response returns answer, citations, sources, and the reranker-relevance `confidence` field (see Section 2's callout on what it does and doesn't mean).

**Request/Response Shape**

```
POST /query → Hybrid Retrieval (Qdrant + Elasticsearch, parallel) → RRF Fusion → Cross-Encoder Rerank → Generation (cite or refuse) → JSON Response
```

## Section 4 · Technology Choices

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | Async-first; I/O-bound store queries and LLM calls use async clients directly |
| Metadata / document store | MongoDB | Flexible schema for document metadata and provenance, independent of vector/keyword indexes |
| Vector store | Qdrant | Dense semantic retrieval; swappable behind a DI interface (Pinecone is a documented stretch goal) |
| Keyword index | Elasticsearch | BM25 keyword search — catches exact-term matches hybrid dense search alone can miss |
| Fusion | Reciprocal Rank Fusion | Combines dense + keyword rankings without needing score normalization across incompatible scales |
| Reranker | bge-reranker-v2-m3 (cross-encoder) | Refines fused candidates before generation; the dominant latency cost, so capped via `RERANK_INPUT_TOP_K` |
| Generation | OpenAI-compatible, provider-agnostic interface | Citation-grounded prompting isn't tied to one vendor's API shape |
| Evaluation | RAGAS + DeepEval | RAGAS supplies LLM-as-judge metrics; DeepEval wraps them in threshold/pass-fail assertions for CI |
| Package/env | uv | Fast, lockfile-based dependency management; `uv.lock` mismatch fails CI before any other check runs |

**Why an ONNX reranker backend was rejected** — An ONNX Runtime backend for the same cross-encoder model was benchmarked as an alternative to the default PyTorch backend, specifically to address reranker latency (the pipeline's dominant cost). Measured result: 1.02x speedup — not a real improvement — so the added dependency and maintenance surface weren't adopted. Full write-up and numbers: `docs/adr/001-reranker-onnx-backend.md`. The latency win that did land came from a different lever: capping `RERANK_INPUT_TOP_K` (20 → 12), validated against the golden set with zero measured recall cost.

**Why refusal is a generation-stage concern, not just retrieval** — The two remaining negative-case false acceptances are not retrieval failures — both are cases where retrieval correctly returns real, topically-relevant evidence that doesn't actually contain the specific fact being asked about, and generation still answers instead of refusing. This is why the Day 6 fix (`LEGAL_RAG_V2`) targeted the generation prompt, not the retrieval or reranking stages.

## Section 5 · Error Handling & Consistency

**Dual-Write Consistency** — Ingestion writes to three independent stores (MongoDB, Qdrant, Elasticsearch) with no distributed transaction across them. A document is not marked `status: ready` in MongoDB until every store write is independently validated — so a partial failure (e.g. the Elasticsearch write fails after Qdrant succeeds) leaves the document in a non-ready state instead of silently becoming queryable with incomplete indexing.

**Refusal as an Explicit Error Path** — Generation is instructed to answer only from retrieved passages and to refuse explicitly — not answer with lower confidence — when the evidence is insufficient or the question requires synthesis the retrieved evidence doesn't support. Refusal is a first-class response, not an exception; it's measured in the same evaluation harness as recall, precision, and faithfulness.

> **Design principle:** a document is never queryable in a partially-indexed state, and an answer is never presented without the evidence that supports it — both are enforced structurally, not by convention.

## Section 6 · Testing & CI

Unit tests never hit a real MongoDB/Qdrant/Elasticsearch/LLM — the client is mocked or faked at the interface boundary. Real-service tests live in `tests/integration/` under the `integration` pytest marker and skip themselves when the dependency is unreachable. `pytest` with no flags must pass with zero external services running and finish in under 60 seconds (NFR-5).

| CI Job | Trigger | What it runs |
|---|---|---|
| `quality` | Every push / PR | `uv sync --locked`, `ruff check`, `ruff format --check`, `mypy`, `pytest` |
| `evaluation-gate` (FR-12) | Manual (`workflow_dispatch`), self-hosted runner | `make evaluate` + `make evaluate-gate` against the pre-seeded golden corpus |

The evaluation gate isn't automatic on every push because the golden corpus (`data/raw/sample_contracts/`) is real, licensed contract text, deliberately gitignored — a GitHub-hosted runner has no way to reproduce it from a checkout alone. The job itself is real, tested YAML, demonstrated both failing (against the pre-fix baseline and against deliberately emptied vector/keyword stores) and passing (against the real, correctly-configured system) — not aspirational configuration.

## Section 7 · Known Limitations

Honest, measured gaps against `docs/01-requirements.md` §7's acceptance bar — not resolved this sprint, tracked here rather than glossed over.

| Limitation | Detail |
|---|---|
| 100% negative-case refusal not met | 2 of 30 golden cases are still answered when they should be refused — both generation-stage judgment failures on topically-real-but-non-answering evidence |
| P95 query latency not met | Cut 53% (56.4s → 26.3s avg), but the CPU-only cross-encoder reranker remains ~9x over the 6s budget; no GPU/DirectML path on the dev machine, and ONNX was measured and rejected |
| Evaluation gate not automated on GitHub-hosted runners | Real, correctly wired, but only runs on a self-hosted runner with the (deliberately gitignored, licensed) corpus pre-seeded, triggered manually |
| Retrieval robustness to config mistakes untested at scale | Only an artificially emptied vector/keyword store cleanly failed the CI gate's checks on this 8-document corpus; a larger, more heterogeneous corpus would be needed to demonstrate a naturally occurring regression |
| Single-run measurements throughout | RAGAS/DeepEval LLM-judge scores aren't perfectly deterministic run-to-run — a repeat run of an identical config previously shifted refusal accuracy from 93.33% to 100% |

## Resources

Want to verify or explore this further? Here you go.

| | |
|---|---|
| **GitHub repository** — full source, commit history, CI configuration | [github.com/Terrytd0/LexRag](https://github.com/Terrytd0/LexRag) |
| **README** — setup, features, benchmark numbers, known limitations | [README.md](https://github.com/Terrytd0/LexRag/blob/main/README.md) |
| **Evaluation notes** — full Day 6 methodology, every intermediate measurement | [docs/experiments/evaluation_notes_day6.md](https://github.com/Terrytd0/LexRag/blob/main/docs/experiments/evaluation_notes_day6.md) |
| **Requirements** — full functional/non-functional requirements & §7 acceptance criteria | [docs/01-requirements.md](https://github.com/Terrytd0/LexRag/blob/main/docs/01-requirements.md) |
| **Case Study** — the companion document to this architecture reference | [case-study.md](case-study.md) |
| **Full docs folder** — architecture ADR, requirements, experiment notes | [docs/](https://github.com/Terrytd0/LexRag/tree/main/docs) |

---
*LexRAG · Technical Architecture · Self-directed portfolio project*
