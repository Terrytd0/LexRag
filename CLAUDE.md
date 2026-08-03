# CLAUDE.md

Engineering playbook for AI-assisted development (Claude Code and similar
tools) in this repository. This is the long-term source of truth for *how*
we build LexRAG — read it before making structural changes, not just before
writing code.

## Project Summary

LexRAG is a legal RAG platform: PDF contracts/case-law → chunk → embed →
hybrid retrieval (Qdrant vector + Elasticsearch BM25, merged via RRF,
refined by a cross-encoder reranker) → citation-grounded, refusal-aware
generation → FastAPI. See [`docs/architecture.md`](docs/architecture.md)
for why each component was chosen and [`docs/01-requirements.md`](docs/01-requirements.md)
for the measurable bar every requirement is held to.

This repository is currently a **scaffold** (Sprint 5 Day 1). Package
`__init__.py` files document what belongs in each module and which sprint
day populates it — check those before assuming a module is dead code.

## Repository Layout

- `api/` — FastAPI app, routers, dependency wiring. HTTP concerns only:
  parse request → call into `ingestion`/`retrieval`/`generation` → shape
  response via `api/schemas/`. No business logic.
- `configs/` — `settings.py` (env-driven `pydantic-settings`) and
  `logging.py`. All configuration flows through here; nothing else reads
  `os.environ` directly.
- `ingestion/` — PDF loading (`loaders/`), chunking (`chunking/`), and the
  pipeline that fans out writes to MongoDB, Qdrant, and Elasticsearch.
- `retrieval/` — `vector_store/` (Qdrant), `keyword_store/` (Elasticsearch),
  `fusion/` (RRF), `reranker/` (cross-encoder). Each store is accessed
  through an interface, never called directly from `api/` or `generation/`.
- `generation/` — Citation-grounded prompting, the provider-agnostic LLM
  interface, and refusal logic.
- `evaluation/` — Golden-dataset runner and `metrics/` (RAGAS/DeepEval
  wrappers for recall@K, precision@K, faithfulness).
- `scripts/` — Standalone entrypoints (`uv run python scripts/x.py`), not
  imported as a package.
- `data/` — Local working storage (`raw/`, `processed/` gitignored;
  `golden/` committed — see `data/README.md`).
- `tests/` — `unit/`, `integration/`, `fixtures/`, mirroring the source
  layout above. No `__init__.py` files (pytest doesn't need them; see
  `explicit_package_bases` in `pyproject.toml` for why mypy needs the
  matching flag).
- `docs/` — Requirements, architecture ADR, diagrams, and `experiments/`
  (working notes, not polished docs) and `adr/` (narrower decisions beyond
  the foundational ones in `architecture.md`).

## Engineering Conventions

### Python style & typing

- **Type hints are required** on every function/method signature —
  parameters and return type. `mypy` runs with `disallow_untyped_defs =
  true`; an untyped `def` is a CI failure, not a style nit.
- Use `from __future__ import annotations` in every module (already the
  convention in the modules that exist today) so annotations stay cheap
  and forward references just work.
- Prefer built-in generics (`list[str]`, `dict[str, int]`) and `X | None`
  over `Optional`/`List`/`Dict` from `typing` — this is a Python 3.13
  codebase, not one carrying 3.8 compatibility baggage.
- Line length 100, enforced by `ruff format` (see
  [Deviation: Ruff instead of Ruff+Black](docs/architecture.md#4-deviations-from-the-suggested-tooling-list)).
  Run `uv run ruff check . && uv run ruff format .` before committing;
  `pre-commit` runs both automatically.

### Docstrings

- Every public module, class, and function gets a **one-line docstring**
  stating what it does or why it exists — not a restatement of the
  signature. Module-level docstrings (see any existing `__init__.py`) also
  say *which sprint day* fills the module in, while the scaffold is
  incomplete.
- Multi-line docstrings are for genuine non-obvious behavior (an
  algorithm's shape, a caller contract, a gotcha) — not padding. If a
  one-liner fully covers it, stop at one line.
- Don't restate the type hints in prose ("param x: an int") — the
  signature already says that.

### Logging

- Use the standard `logging` module via `logging.getLogger(__name__)` —
  never `print`. Configure once at process start via
  `configs.logging.configure_logging()`.
- `INFO` for request-level events (ingestion started/completed, query
  received/answered) with a correlation ID and latency, per NFR-6 in
  `docs/01-requirements.md`. `DEBUG` for retrieval internals (candidate
  counts, fusion scores) — verbose enough to debug locally, not enabled by
  default.
- Never log secrets, full document text, or raw LLM prompts/responses at
  `INFO` — log identifiers (`doc_id`, `chunk_id`, query hash) and
  aggregate stats instead.

### Error handling

- Validate at boundaries: `api/schemas/` validates everything crossing the
  HTTP boundary; nothing downstream re-validates the same shape defensively.
- Internal code trusts its own guarantees. Don't add a `None` check for a
  value the type system already says can't be `None`.
- Catch specific exceptions, not bare `except Exception`, except at the
  top-level API error handler that turns unexpected failures into a
  structured 500 response — that's the one deliberate backstop.
- Store-write failures during ingestion (Mongo/Qdrant/Elasticsearch) must
  not leave a document `status: ready` if any write failed — see
  `docs/architecture.md §3` (dual-write consistency) before touching
  `ingestion/`.

### Testing

- Every new module under `x/y.py` gets a test at `tests/unit/x/test_y.py`.
- Unit tests never hit a real MongoDB/Qdrant/Elasticsearch/LLM — mock or
  fake the client at the interface boundary. Real-service tests go in
  `tests/integration/` under the `integration` pytest marker and must skip
  themselves (not fail) when the dependency is unreachable — see
  `tests/integration/README.md`.
- `uv run pytest` (no flags) must pass with zero external services running
  and finish in under 60 seconds (NFR-5). If a test needs Docker Compose
  up, it belongs in `integration/`.
- A bug fix gets a regression test in the same commit. A new endpoint or
  retrieval/generation behavior ships with at least one test proving the
  acceptance criterion it's meant to satisfy (`docs/01-requirements.md §7`).

### Naming conventions

- Modules and packages: `snake_case`. Classes: `PascalCase`. Functions,
  variables, module-level constants that vary: `snake_case`. True
  constants: `UPPER_SNAKE_CASE`.
- Pydantic schemas in `api/schemas/`: suffix by role — `UploadRequest`,
  `UploadResponse`, `QueryRequest`, `QueryResponse`, not generic names like
  `Payload` or `Data`.
- Settings fields in `configs/settings.py` mirror their environment
  variable name in lowercase (`qdrant_url` ↔ `QDRANT_URL`) — no renaming
  between the two.

### Dependency injection

- FastAPI's `Depends()` is the DI mechanism for request-scoped
  dependencies (settings, store clients, the retrieval/generation
  pipeline). Route handlers receive dependencies as parameters; they don't
  reach for module-level singletons or construct clients inline.
- Store and LLM clients are constructed once (see `configs.settings.get_settings`'s
  `lru_cache` pattern) and injected, not re-instantiated per request.
- This is also what makes backend swaps (Qdrant → Pinecone, per the
  stretch goal) a DI wiring change, not a call-site rewrite — code depends
  on the interface in `retrieval/vector_store/`, never on `QdrantClient`
  directly outside that module.

### Async-first

- `api/` route handlers are `async def`. I/O-bound calls (store queries,
  HTTP calls to an LLM provider) use async clients where available.
- A genuinely blocking, CPU-bound call (local embedding/reranker
  inference) is pushed to a thread pool (`fastapi.concurrency.run_in_threadpool`
  or `asyncio.to_thread`) rather than left to block the event loop —
  called out specifically because `sentence-transformers` inference is
  synchronous.
- Don't reach for `async def` in `ingestion/`, `retrieval/`, or
  `generation/` internals just for consistency — use it where there's
  real concurrent I/O to overlap (e.g. the parallel vector + BM25 query in
  FR-7), not as a blanket style rule.

### Keep business logic out of API routes

- A route handler's body is: parse/validate input (via the schema),
  call one service-layer function, return its result via a response
  schema. If a route body is doing retrieval logic, prompt construction,
  or store writes inline, that logic belongs in `retrieval/`,
  `generation/`, or `ingestion/` instead, with the route calling into it.
- This is what keeps `evaluation/` able to run the same pipeline the API
  uses, rather than re-implementing query logic against the API's HTTP
  surface.

### Modularity & maintainability

- Prefer the smallest change that satisfies the current requirement. Don't
  add abstraction layers, config flags, or provider adapters for
  hypothetical future needs — the Pinecone adapter, for example, is an
  explicit stretch goal, not something to half-build now "in case."
  See the root-level guidance on avoiding speculative abstraction.
- Three similar lines beat a premature shared helper. Extract only once a
  real third use case shows up.

### Explain reasoning before major architectural changes

- `docs/architecture.md` is an ADR — treat it as binding until explicitly
  revised. If implementation reveals a foundational decision there should
  change (a store, the fusion strategy, the embedding model), **explain
  the reasoning and trade-off before making the change**, then update the
  relevant ADR section (or add a new record under `docs/adr/` for a
  narrower decision) so the record stays accurate.
- Smaller, local decisions (a helper's internal structure, a variable
  name) don't need this ceremony — use judgment about what's
  "architectural": does it affect another module's interface, a stored
  data shape, or a documented requirement/acceptance criterion?

## AI Engineering Principles

- Performance before cleverness.
- Measure before optimizing.
- Prefer explicit pipelines over hidden framework magic.
- Every retrieval improvement should be measurable.
- Every generation improvement should improve evaluation metrics.
- Don't increase architectural complexity unless it improves measurable retrieval or generation quality.

## Version Control

- **Never run `git commit` or `git push` unless explicitly instructed in
  that same turn.** Making local edits, running tests, or fixing a bug does
  not imply permission to commit it — wait to be asked.
- "Explicit" means an instruction to commit/push, not agreement with an
  unrelated question. A "yes" answering "should I use approach A or B?"
  is not authorization to also commit. When genuinely ambiguous, ask
  rather than assume.
- A request to commit does not carry forward to later changes in the same
  session. Each round of edits needs its own explicit go-ahead.
- This applies even when a task (e.g. a sprint-day deliverable) describes
  a commit as one of its steps — confirm before running it rather than
  treating the task description as standing authorization.