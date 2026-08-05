# LexRAG

**Contract & case-law intelligence platform** — a legal retrieval-augmented
generation system that answers questions over contracts and case filings
with traceable citations, and refuses to answer when the evidence isn't
there.

> **Status:** Sprint 5, Days 1–5 complete, plus a Day 4 production-hardening
> pass (duplicate detection, document browser/delete, document-scoped
> queries, persistent model cache). Ingestion, hybrid retrieval, reranking,
> citation-grounded generation, the full REST API, and the golden-dataset
> evaluation harness are implemented and Dockerized end-to-end. CI quality
> gate wiring (Day 6) is next (see [Roadmap](#roadmap)).

## Overview

Paralegals burn hours per matter manually searching thousands of pages of
contracts and prior filings for relevant clauses and precedent. Keyword
search misses paraphrased queries; a generic LLM chat interface answers
confidently without evidence, which is unusable — and risky — for legal
work.

LexRAG ingests PDF contracts and case-law documents, indexes them for
**hybrid retrieval** (dense vector search + BM25 keyword search, merged via
Reciprocal Rank Fusion and refined by a cross-encoder reranker), and
generates answers that cite only retrieved passages — refusing to answer
when the corpus doesn't contain sufficient evidence. Retrieval and
generation quality are measured against a golden dataset and enforced as a
CI quality gate, not tuned by eyeballing outputs.

Full requirements: [`docs/01-requirements.md`](docs/01-requirements.md).
Architecture decisions and trade-offs: [`docs/architecture.md`](docs/architecture.md).

## Architecture

![LexRAG system architecture: PDF ingestion pipeline feeding MongoDB/Qdrant/Elasticsearch, a hybrid retrieval query pipeline with RRF fusion and cross-encoder reranking, and an evaluation pipeline gating CI](docs/screenshots/LexRAG.png)

Editable source: [`docs/architecture.mmd`](docs/architecture.mmd). Render
changes at [mermaid.live](https://mermaid.live) or a Mermaid-aware editor,
then re-export the PNG to `docs/screenshots/LexRAG.png` to keep this in
sync.

## Features

- PDF ingestion with configurable chunking and full provenance metadata
  (document, section, page, chunk index).
- Hybrid retrieval: parallel dense vector (Qdrant) + BM25 (Elasticsearch)
  search, merged with Reciprocal Rank Fusion.
- Cross-encoder reranking of fused candidates, with a configurable
  pre-rerank candidate cap (`RERANK_INPUT_TOP_K`) to bound the dominant cost
  in query latency.
- Citation-grounded generation that cites only retrieved passages, with
  explicit refusal when evidence is insufficient or the question requires
  synthesis the retrieved evidence doesn't support.
- `POST /upload`, `POST /query`, `GET /documents`, and
  `DELETE /documents/{doc_id}` REST API with a typed response contract
  (answer, citations, sources, confidence).
- Duplicate-upload detection (SHA-256 content hash) — a byte-identical
  re-upload is skipped entirely and returns the existing document instead
  of re-ingesting.
- Document-scoped queries — `POST /query`'s optional `document_ids` filters
  retrieval (via native Qdrant/Elasticsearch filters, not post-filtering) to
  one or more specific documents instead of the whole corpus.
- Golden-dataset evaluation harness (RAGAS/DeepEval) reporting recall@K,
  precision@K, faithfulness, refusal accuracy, and per-stage latency, with
  automatic failure-stage classification. Not yet wired into CI as a
  quality gate (Day 6 — see [Roadmap](#roadmap)).
- Fully Dockerized local stack (API + MongoDB + Qdrant + Elasticsearch),
  including a persistent volume for downloaded embedding/reranker model
  weights so rebuilding the API image doesn't re-download them.

## Tech Stack

| Layer | Choice |
|---|---|
| API | FastAPI |
| Metadata / document store | MongoDB |
| Vector store | Qdrant |
| Keyword index | Elasticsearch (BM25) |
| Embeddings | BAAI/bge-m3 (sentence-transformers) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Generation | OpenAI-compatible LLM behind a provider-agnostic interface |
| Evaluation | RAGAS + DeepEval |
| Package/env management | uv |
| Testing | Pytest |
| Lint/format/type-check | Ruff, mypy |
| CI | GitHub Actions |
| Containerization | Docker Compose |

Full rationale and alternatives considered for each choice:
[`docs/architecture.md`](docs/architecture.md).

## Installation

Requires **Python 3.13** and [**uv**](https://docs.astral.sh/uv/).

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux
# irm https://astral.sh/uv/install.ps1 | iex        # Windows PowerShell

# Clone, configure, and install (creates .venv, installs deps + git hooks)
git clone <repo-url> lexrag && cd lexrag
cp .env.example .env   # fill in OPENAI_API_KEY etc.
make install
```

`make install` runs `uv sync --extra dev` + `uv run pre-commit install`. No
`make` on your system (plain Windows cmd/PowerShell)? Run those two commands
directly — see [Development](#development) for the full command reference.

## Quick Start

```bash
# Run the full stack (API + MongoDB + Qdrant + Elasticsearch)
cp .env.example .env   # fill in OPENAI_API_KEY etc.
docker compose up

# Health check
curl http://localhost:8000/health

# Upload a PDF, then ask a question against the indexed corpus
curl -X POST http://localhost:8000/upload -F "file=@contract.pdf;type=application/pdf"
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What is the termination notice period?"}'

# Browse ingested documents
curl http://localhost:8000/documents

# Run tests
make test
```

Interactive API docs (Swagger UI) are at `http://localhost:8000/docs` once
the stack is up.

## Development

Everyday commands are wrapped in the `Makefile` so CI and local development
run the exact same checks. `make help` lists all targets; the ones you'll
use most:

| Command | What it does |
|---|---|
| `make install` | Create/refresh `.venv`, install deps (incl. dev extras), install pre-commit hooks |
| `make run` | Run the API locally with auto-reload (`uv run python -m api`) |
| `make test` | Run the unit test suite (`uv run pytest`) |
| `make test-cov` | Run tests with a coverage report |
| `make lint` | Lint with Ruff |
| `make format` | Auto-format with Ruff (rewrites files) |
| `make check` | Full quality gate: format check + lint + mypy + tests — what CI runs |

Every target just wraps a `uv run ...` command, so if `make` isn't available
(there's no native `make` on plain Windows cmd/PowerShell — use Git Bash,
WSL, or install one), open the `Makefile` and run the underlying command
directly.

CI (`.github/workflows/ci.yml`) runs on every push and pull request:
`uv sync --locked`, then `ruff check`, `ruff format --check`, `mypy`, and
`pytest`, in that order — identical to `make check`. A `uv.lock` mismatch
with `pyproject.toml` fails the build immediately, before any other check
runs.

## Project Structure

```
lexrag/
├── api/                 # FastAPI app, routes, request/response schemas
├── configs/             # Environment-driven settings, logging setup
├── data/                # Local working storage: raw/, processed/, golden/
├── docs/                # Requirements, architecture (ADR), diagrams, experiment notes
├── evaluation/          # Golden-dataset harness, RAGAS/DeepEval metric wrappers
├── generation/          # Citation-grounded prompting, refusal logic, LLM interface
├── ingestion/           # PDF loaders, chunking, embedding, store fan-out
├── retrieval/           # Vector store, keyword store, RRF fusion, reranker
├── scripts/             # One-off operational scripts (seeding, eval runs, resets)
└── tests/               # unit/, integration/, fixtures/ mirroring the layout above
```

Each package's `__init__.py` documents what belongs there and which sprint
day populates it — see [`CLAUDE.md`](CLAUDE.md) for the full repository
layout and engineering conventions.

## Roadmap

| Day | Focus | Status |
|---|---|---|
| 1 | Discovery & solution design — requirements, architecture, scaffold | ✅ Done |
| 2 | Data & ingestion foundation — Docker Compose, chunker, MongoDB schema | ✅ Done |
| 3 | Hybrid retrieval — Qdrant + Elasticsearch, RRF merge | ✅ Done |
| 4 | Cited generation & API — reranker, `/query`, `/upload`, plus a production-hardening pass (dedup, document browser/delete, document-scoped queries, persistent model cache) | ✅ Done |
| 5 | Evaluation & quality — golden dataset (7 real SEC-filed contracts, 30 Q/A cases), retrieval/generation/refusal metrics, error analysis | ✅ Done |
| 6 | Hardening & portfolio — CI quality gate, full Docker stack, walkthrough | ⬜ Planned |

Lint/type/test CI (`.github/workflows/ci.yml`) and the `Makefile` command
surface landed early, as infrastructure hardening ahead of Day 2. Day 6 adds
the remaining piece: wiring the metrics this evaluation harness already
produces into an actual CI quality gate (FR-12) that fails a build when
retrieval/generation metrics regress.

## Evaluation

The evaluation harness (`evaluation/`, Day 5) runs a 30-case golden Q/A
dataset (`data/golden/golden_qa.jsonl` — 22 positive cases across all 10
required clause topics, 8 adversarial/out-of-scope negative cases) through
the same retrieval → rerank → generation pipeline `POST /query` uses in
production, and reports:

- **Retrieval** — Recall@5/10 and Precision@5/10, computed directly against
  the golden set's expected documents (not delegated to RAGAS), for each of
  dense-only, sparse-only, and hybrid+RRF — so the value each retrieval
  stage adds is visible independently, not just the combined pipeline.
- **Generation** — Faithfulness, Context Precision, Context Recall, and
  Answer Relevancy via RAGAS's LLM-as-judge metrics, for answerable
  (non-refused) cases only. DeepEval wraps each RAGAS score in a
  threshold/pass-fail assertion (`docs/architecture.md` §2.10) — the
  CI/test-runner integration layer around RAGAS's scores, not a second,
  redundant LLM-judge pass.
- **Refusal behaviour** — accuracy across *all* 30 cases, plus false
  refusals (an answerable case incorrectly refused) and false acceptances
  (a case with no supporting evidence incorrectly answered) reported as
  separate counts, not averaged into one number — for a legal tool, an
  unflagged hallucination is a materially worse failure than an overly
  cautious refusal.
- **Latency** — average retrieval, reranker, generation, and end-to-end
  time across all 30 cases.
- **Error analysis** — every failing case is automatically attributed to
  the first pipeline stage that couldn't have produced a correct answer
  (retrieval / reranker / refusal / generation), so failures point at where
  to look, not just that something failed.

Acceptance thresholds are defined in
[`docs/01-requirements.md` §7](docs/01-requirements.md#7-measurable-acceptance-criteria).
Reproduce locally with:

```bash
docker compose up -d mongo qdrant elasticsearch
uv run python scripts/seed_corpus.py     # once, to seed the golden dataset's corpus
make evaluate                            # or: uv run python scripts/run_evaluation.py
```

Full methodology, dataset design rationale, and baseline results:
[`docs/experiments/evaluation_notes.md`](docs/experiments/evaluation_notes.md).

### Benchmark numbers

Baseline run 2026-08-05 against the 7-document real-contract corpus
described in `docs/experiments/evaluation_notes.md` (30 golden cases; LLM:
`gpt-5.6-luna`, a one-off substitution for this run — see that doc for why):

| Metric | Value | vs. `docs/01-requirements.md` §7 |
|---|---|---|
| Recall@10 (hybrid) | 1.00 | ✅ ≥ 0.85 |
| Precision@5 (hybrid) | 0.88 | ✅ ≥ 0.70 |
| Faithfulness | 0.95 | ✅ ≥ 0.90 |
| Context Precision | 0.65 | no documented threshold |
| Context Recall | 0.96 | no documented threshold |
| Answer Relevancy | 0.80 | no documented threshold |
| Refusal accuracy | 93.33% (28/30) | — |
| Negative-case refusal | 75% (6/8, 2 false acceptances) | ❌ target is 100% |
| Avg end-to-end latency | 56.4s | ❌ NFR-1 target ≤ 6s (P95) — reranker-bound, 94.9% of total |

Two real gaps, neither tuned this session per the optimization policy
(measure first, change later with its own before/after) — see
`docs/experiments/evaluation_notes.md`'s Observations for root-cause analysis
of both: negative-case refusal misses 100% (generation-stage judgment issue,
not retrieval or threshold), and end-to-end latency is dominated almost
entirely by the CPU-only cross-encoder reranker (a known, previously-profiled
bottleneck, confirmed here at full golden-dataset scale).

Full retrieval-strategy comparison, per-case detail, and failure list:
`docs/experiments/evaluation_notes.md` and `evaluation/reports/latest.md`
(regenerate with `make evaluate` — reports are generated output, gitignored).

A follow-up comparison against `gpt-5.4-nano` (same dataset/corpus/config,
only the LLM changed) is in
[`docs/experiments/evaluation_notes_gpt54nano.md`](docs/experiments/evaluation_notes_gpt54nano.md)
— it also includes a repeat-run finding worth reading before trusting any
single-run model comparison from this harness: re-running gpt-5.6-luna with
nothing else changed shifted its own refusal accuracy from 93.33% to 100%.

### What the `confidence` field means (and doesn't)

`POST /query`'s response includes a `confidence` field
(`domain.generation.GenerationResult.confidence`). **This is the
cross-encoder reranker's relevance score for the single highest-ranked
retrieved chunk — not a calibrated measure of whether the generated answer
is correct.** Concretely:

- It reflects how well *one* chunk matches the question, not whether the
  full answer (which may synthesize several chunks) is accurate.
- A multi-part question whose evidence is spread across several chunks can
  report a *lower* `confidence` than a single-fact lookup, even when the
  synthesized answer is entirely accurate and well-cited — no chunk alone
  is a tight match for the whole question.
- It's the same score `Settings.generation_min_context_score` gates
  refusal on (FR-10) — a retrieval-relevance threshold, not an
  answer-correctness threshold.

This is documented, current behaviour, not a bug — and per this sprint's
scope, it is **not** being redesigned into a calibrated answer-confidence
score today. Treat `confidence` as "how relevant was the best single piece
of evidence," and rely on the `citations` list (which passages actually
support the answer) to judge whether an answer is trustworthy, not on this
field alone.

**Measured, not just asserted:** a follow-up analysis
(`scripts/confidence_correlation.py`) computed the actual correlation between
`confidence` and answer correctness/faithfulness across the golden dataset —
Pearson r ≈ 0 for both (see
[`docs/experiments/evaluation_notes.md`](docs/experiments/evaluation_notes.md#confidence-correlation-follow-up-measurement)
for the full results, caveats, and concrete examples, including one answer
that scored `confidence=0.92` while being completely unfaithful). Confidence
reliably flags "nothing relevant was retrieved at all," but carries no
further signal once something relevant is found.

## Future Improvements

Tracked as stretch goals in [`docs/01-requirements.md` §4](docs/01-requirements.md#4-stretch-goals):

- Pinecone vector-store adapter behind the existing retrieval interface,
  benchmarked against Qdrant.
- Structured termination-clause detection (beyond free-text Q/A).
- Streamed (token-by-token) `/query` responses.
- Query rewriting/decomposition for multi-part legal questions.
- Per-matter/client access control on top of the existing `document_ids`
  query filter (today it's an opt-in scoping parameter, not an
  authentication/authorization boundary).

## License

MIT — see [`LICENSE`](LICENSE).
