# LexRAG — Architecture Decision Record

**Status:** Approved for Sprint 5, Day 1
**Scope:** Foundational, system-wide decisions. Narrower decisions made later
in the sprint get their own record under `docs/adr/` and are linked from
here as they're written.

See `docs/architecture.mmd` for the full system diagram (render at
[mermaid.live](https://mermaid.live) or any Mermaid-aware Markdown viewer).

## 1. System Overview

LexRAG has three pipelines sharing one storage layer:

- **Ingestion** (Day 2): PDF → text extraction → chunking → embedding →
  fan-out write to MongoDB (raw text + metadata), Qdrant (vectors), and
  Elasticsearch (BM25 index).
- **Query** (Day 3–4): question → parallel vector + BM25 retrieval → RRF
  merge → cross-encoder rerank → citation-grounded generation (or refusal)
  → response.
- **Evaluation** (Day 5–6): golden dataset → same query pipeline → RAGAS/
  DeepEval metrics → CI quality gate.

Every architectural decision below is driven by the same constraint: this
is a legal-domain RAG system, so **retrieval quality and answer groundedness
are the product**, not implementation details. Where a simpler alternative
existed, it was rejected specifically when it would have made retrieval
quality harder to measure or improve.

## 2. Decisions

### 2.1 API Framework: FastAPI

**Decision:** FastAPI for the HTTP layer.

**Why:** Native Pydantic integration gives us request/response validation
"for free," which matters directly for a requirement like FR-6/FR-9
(citations must map to real retrieved chunks — a typed `Citation` model
makes that a schema-level guarantee, not just a prompt instruction).
Async-native, so the I/O-bound fan-out to Mongo/Qdrant/Elasticsearch during
ingestion and the parallel vector+BM25 calls during query (FR-7) can run
concurrently without a separate async framework bolted on. Auto-generated
OpenAPI docs give us a Swagger UI for manual end-to-end testing (Day 4 DoD)
with no extra tooling.

**Alternatives considered:**
- *Flask* — mature and simple, but synchronous by default; parallel
  retrieval would need explicit threading, and request validation would be
  hand-rolled or bolted on via an extension.
- *Django REST Framework* — far more than this project needs (ORM, admin
  panel, auth scaffolding we don't want); heavier startup and mental
  overhead for a focused RAG API.

**Consequences:** We take on FastAPI/Starlette's async discipline
project-wide — blocking calls (e.g. a synchronous embedding model call)
must be pushed to a thread pool (`run_in_threadpool`) or the event loop
stalls under load. This is called out explicitly in `CLAUDE.md`.

### 2.2 Metadata & Document Store: MongoDB

**Decision:** MongoDB for raw document text and chunk metadata (`doc_id`,
`source`, `section`, `page`, `chunk_index`, timestamps, ingestion status).

**Why:** Chunk metadata is naturally document-shaped and schema-variable —
a case-law filing and a vendor contract don't share the same section
structure, and legal documents routinely need new metadata fields added
without a migration. MongoDB's flexible schema fits that better than a
relational table, and its document model maps 1:1 onto "one document has
many chunks, each chunk carries a provenance blob." We don't need
relational joins or multi-table transactions here — the two-store
consistency problem lives between Mongo/Qdrant/Elasticsearch, not inside
Mongo.

**Alternatives considered:**
- *PostgreSQL + JSONB* — would work, and is what the SupportOps AI sibling
  project uses for its structured domain. Rejected here because chunk
  metadata has no relational structure worth normalizing (no foreign-key
  heavy joins), so JSONB columns would just be MongoDB with extra
  ceremony. If LexRAG later needs relational features (e.g. per-firm
  billing, user accounts), Postgres becomes the better default again —
  this decision is scoped to document/chunk metadata specifically.
- *Storing metadata directly in Qdrant payloads only* — rejected as the
  sole store because Qdrant payload filtering is not a substitute for a
  queryable document store, and it would couple metadata lifecycle to the
  vector store's lifecycle (e.g. reindexing Qdrant would risk losing the
  only copy of provenance data).

**Consequences:** MongoDB is the source of truth for provenance; Qdrant and
Elasticsearch are derived indexes over it. If either index is dropped and
rebuilt, it's rebuilt *from* MongoDB, never the reverse (see NFR-4,
atomic ingestion).

### 2.3 Vector Store: Qdrant

**Decision:** Qdrant for dense vector storage and similarity search.

**Why:** Open-source, self-hostable via Docker Compose (no external
account/API key needed for local dev or CI), with a mature Python client,
payload filtering (needed to scope search by document/matter later), and
solid recall/latency at the corpus sizes this project targets (≤10k
chunks, NFR-1). It's also one of the roadmap tools this sprint is meant to
cover.

**Alternatives considered:**
- *Pinecone* — managed, less ops overhead, but requires an external
  account and network dependency for local dev and CI, and a paid tier for
  meaningful capacity. Planned as a Day 6 **stretch** adapter specifically
  *because* comparing it against Qdrant behind the same interface (NFR-7)
  is more valuable as a documented, lived trade-off than a single default
  choice would be.
- *pgvector* — attractive if we were already committed to Postgres for
  metadata (2.2), but we're not, and pgvector's ANN performance/tooling is
  less mature than a purpose-built vector database at the scale/latency
  targets here.
- *FAISS (in-process)* — no server, no persistence/replication story out
  of the box, no filtering; would mean building storage/persistence
  ourselves. Wrong trade for a system meant to run as a real service.

**Consequences:** Retrieval code depends on Qdrant only through the
interface in `retrieval/vector_store/` (NFR-7), which is what makes the
Pinecone stretch goal a config change plus an adapter implementation,
not a rewrite.

### 2.4 Keyword Store: Elasticsearch (BM25)

**Decision:** Elasticsearch for lexical/keyword search using its default
BM25 scoring.

**Why:** Legal text is full of exact-match-critical tokens — defined terms,
clause numbers, statute citations, party names — where lexical search
reliably beats embeddings (a vector model may consider "Section 8.3" and
"Section 8.4" nearly identical; BM25 won't). Elasticsearch is the
industry-standard way to get production-grade BM25 with minimal custom
code, and its analyzer configuration gives us control over tokenization
for legal-specific patterns (e.g. not splitting "§8.3").

**Alternatives considered:**
- *Postgres full-text search (`tsvector`)* — sufficient for small corpora,
  but weaker relevance tuning and analyzer control than Elasticsearch, and
  we're not otherwise using Postgres (2.2).
- *Whoosh / pure-Python BM25* — fine for a prototype, not something we'd
  represent as production-grade, and this project is explicitly meant to
  demonstrate production tooling.

**Consequences:** This is the second store in the dual-write ingestion
path (alongside Qdrant), which is the sprint's called-out top risk (see
Section 3). Ingestion validates both writes independently before a
document is marked queryable.

### 2.5 Hybrid Retrieval (Vector + BM25 in Parallel)

**Decision:** Every query runs both a Qdrant vector search and an
Elasticsearch BM25 search, concurrently, over the same candidate pool.

**Why:** Vector and lexical search fail in different, complementary ways.
Vector search generalizes across phrasing ("end the agreement early" ≈
"termination for convenience") but can miss exact-term queries; BM25 nails
exact terms but can't generalize across phrasing. A legal Q/A system needs
both — a paralegal might ask about "indemnification" in one query and
"who bears the loss if X happens" in the next. Running single-strategy
retrieval and hoping one mode covers both cases isn't acceptable when
recall is a measured, gated metric (FR-11, Section 7 of
`01-requirements.md`).

**Alternatives considered:**
- *Vector-only* — simplest, but demonstrably worse recall on exact-term
  legal queries (defined terms, clause numbers); the golden dataset's
  precision/recall thresholds are chosen specifically to make this
  failure visible rather than hidden.
- *BM25-only* — simplest of all, but fails on paraphrased/semantic
  queries, which is most of how non-lawyers (and even lawyers, under time
  pressure) actually ask questions.

**Consequences:** Two network calls per query instead of one, run
concurrently to keep latency within NFR-1. Two ranked lists need merging —
see 2.6.

### 2.6 Reciprocal Rank Fusion (RRF)

**Decision:** Merge the vector and BM25 ranked lists using Reciprocal Rank
Fusion, `score(d) = Σ 1 / (k + rank_i(d))` across retrieval strategies `i`,
with `k = 60` as the default (configurable, `RRF_K` in `configs/settings.py`).

**Why:** RRF requires no score normalization between fundamentally
different scoring functions (cosine similarity vs. BM25's unbounded
scores) — it only needs each list's *rank order*, which sidesteps the
classic "how do I compare a 0.82 cosine score to a 14.3 BM25 score"
problem. It's simple, has no learned parameters (nothing to train or
overfit on a 25–30 example golden set), and is well-established in
information retrieval literature and in production hybrid-search systems
(e.g. Elasticsearch's own RRF support, OpenSearch's hybrid query). `k=60`
is the commonly cited default in the source RRF paper (Cormack et al.) and
is our documented starting point, tuned from there using
`docs/experiments/retrieval_debugging.md` (Day 3).

**Alternatives considered:**
- *Weighted score fusion (normalize + linear combination)* — requires
  choosing/tuning a normalization scheme and a weight, both of which are
  corpus- and query-distribution-dependent; more moving parts for a
  25–30-example golden set to validate against.
- *Learned re-ranking as the only fusion step (skip RRF, concatenate +
  rerank)* — pushes all the fusion work onto the cross-encoder over a
  much larger, noisier candidate set, which is slower and doesn't
  cleanly separate "did the retrieval layer surface the right chunk"
  from "did reranking pick it," which we want measured separately
  (Section 8 of `01-requirements.md` reports metrics per pipeline stage).

**Consequences:** RRF is a fusion step, not a relevance filter — its output
still needs reranking (2.7) before generation, since RRF only reorders by
combined rank, not by fine-grained semantic relevance to the specific
query.

### 2.7 Cross-Encoder Reranking

**Decision:** Rerank the RRF-fused top-k (default top 50, `RETRIEVAL_TOP_K`)
with a cross-encoder model, then take the top-N (default 8, `RERANK_TOP_K`)
into generation.

**Why:** Bi-encoders (used for the initial Qdrant vector search) embed the
query and each chunk independently, trading accuracy for speed so we can
search thousands of chunks. A cross-encoder jointly attends over
`(query, chunk)` pairs, which is far more accurate but too slow to run
over an entire corpus — so it's applied only to the already-narrowed
RRF output. This two-stage "retrieve cheap, rerank precisely" pattern is
standard practice specifically because it gets both properties: an
initial pass over the full corpus, and a precise final ranking of a small
candidate set. It's also the direct lever for FR-8 and the precision@5
acceptance threshold (Section 7.3 of `01-requirements.md`).

**Alternatives considered:**
- *Skip reranking, pass RRF output straight to generation* — cheaper and
  faster, but pushes noisier context into the LLM prompt, which directly
  hurts faithfulness (Section 7.4) — the model is more likely to cite or
  rely on a marginally-relevant chunk if nothing better distinguishes it.
- *LLM-as-reranker (ask the LLM to rank the RRF candidates)* — more
  expensive per query (extra LLM calls) and slower than a small local
  cross-encoder, for comparable or worse ranking quality at this scale.

**Consequences:** Reranking is an extra model load + inference step in the
query path, budgeted for in the P95 latency target (NFR-1). The model
(`cross-encoder/ms-marco-MiniLM-L-6-v2` by default) runs locally — no
external API dependency or per-call cost.

### 2.8 Embedding Model: BGE (BAAI/bge-base-en-v1.5)

**Decision:** `BAAI/bge-base-en-v1.5` as the default embedding model
(`EMBEDDING_MODEL`, swappable via config), served through
`sentence-transformers`.

**Why:** Open-source, runs locally (no per-embedding API cost or external
dependency during ingestion — relevant given the risk called out in the
sprint plan about LLM/embedding provider dependency), strong performance
on the MTEB retrieval benchmark for its size class, and 768 dimensions —
a reasonable storage/accuracy trade-off for a ≤10k-chunk corpus. It's also
directly swappable: `retrieval/vector_store/` and the embedding step in
`ingestion/` depend on a model name and dimension from `configs/settings.py`,
not a hardcoded provider.

**Alternatives considered:**
- *OpenAI `text-embedding-3-small/large`* — strong quality, but a paid,
  network-dependent API for every chunk embedded and every query — adds
  cost and an external dependency to both ingestion and the query hot
  path, which conflicts with the sprint's explicit risk mitigation
  ("keep models small and cheap behind a provider-agnostic interface").
- *`all-MiniLM-L6-v2`* — smaller and faster, but noticeably lower
  retrieval quality than BGE-base on benchmark leaderboards; not the
  right trade when recall@K is a gated metric.

**Consequences:** Local embedding inference means ingestion throughput is
bounded by local CPU/GPU rather than an API rate limit — a deliberate
trade favoring reproducible offline evaluation over raw ingestion speed.

### 2.9 Generation LLM: Provider-Agnostic Interface, OpenAI-Compatible Default

**Decision:** Generation goes through a small provider interface in
`generation/`, defaulting to an OpenAI-compatible chat completion API
(`LLM_PROVIDER` / `LLM_MODEL` in `configs/settings.py`), rather than a
hardcoded SDK call.

**Why:** The sprint plan explicitly flags LLM dependency as a risk; keeping
the boundary thin means swapping models (or providers) is a config change.
Legal-domain generation also needs the same provider-agnostic discipline
`retrieval/` uses for its stores (NFR-7) — this is a consistent pattern
project-wide, not a one-off.

**Alternatives considered:**
- *LangChain's full chat-model abstraction* — would give provider
  swapping "for free," but pulls in a large dependency surface for what
  is, here, one call site with a structured prompt and a citation-parsing
  step; a thin custom interface is easier to reason about and keeps the
  citation-validation logic (FR-9) explicit rather than hidden in a
  framework callback.

**Consequences:** We own the (small) interface and its error handling;
in exchange we avoid a framework dependency whose abstractions don't buy
us much for a single-call, structured-output use case.

### 2.10 Evaluation: RAGAS + DeepEval

**Decision:** Use RAGAS for retrieval/generation metrics (faithfulness,
context precision/recall, answer relevancy) and DeepEval for
assertion-style test integration of those metrics into `pytest`/CI.

**Why:** RAGAS is purpose-built for exactly the metrics this project is
gated on (Section 7 of `01-requirements.md`): faithfulness (are generated
claims supported by retrieved context) and context precision/recall,
computed without requiring hand-labeled relevance judgments for every
query, using an LLM-as-judge approach. DeepEval complements it by giving
those metrics a native `pytest` assertion interface
(`assert_test`/custom metrics), which is what makes "CI fails below
threshold" (FR-12) a normal test failure instead of a bespoke script with
its own exit-code convention.

**Alternatives considered:**
- *Hand-rolled metrics only (manual recall@K/precision@K, no
  faithfulness)* — retrieval metrics are straightforward to compute
  directly against the golden set's known-relevant chunks and are in fact
  still computed directly (not delegated to RAGAS) for that reason.
  Faithfulness, though, requires judging whether free-text generated
  claims are *supported* by context — that's a semantic judgment
  hand-rolled string matching can't do reliably, which is exactly what
  RAGAS's LLM-judge metrics are for.
- *DeepEval's own retrieval/generation metrics only (skip RAGAS)* —
  DeepEval's metric set overlaps RAGAS's; RAGAS is used here specifically
  for its RAG-focused, widely-benchmarked faithfulness/context metrics,
  while DeepEval is used for the CI/test-runner integration layer around
  them — each is used for what it's best known for rather than picking
  one and reimplementing the other's strengths.

**Consequences:** Both tools call an LLM as a judge for some metrics,
which means evaluation runs have a real cost and are not fully
deterministic run-to-run — mitigated by pinning judge-model temperature to
0 and treating small metric fluctuations near a threshold as a signal to
investigate, not just re-run.

### 2.11 Package & Environment Management: uv

**Decision:** `uv` manages the virtual environment, dependency resolution,
and lockfile (`uv.lock`), with `pyproject.toml` as the single source of
dependency truth.

**Why:** Single tool replacing `pip` + `venv` + (optionally) `pip-tools`,
with dramatically faster installs/resolution — meaningful for CI runtime
(NFR-10's gate runs on every PR) and for local iteration speed when
`sentence-transformers`/`torch` are in the dependency tree. Produces a
committed lockfile for reproducible installs, matching the "clean checkout
just works" bar in NFR-9.

**Alternatives considered:**
- *Poetry* — mature and also gives a lockfile, but slower dependency
  resolution and a separate `[tool.poetry]` metadata format historically
  at odds with standard `pyproject.toml` (now converging, but uv's native
  PEP 621 `[project]` support is simpler); no meaningful capability uv
  lacks for this project's needs.
- *Plain `pip` + `requirements.txt` only* — no lockfile-grade
  reproducibility (hash pinning, transitive pin consistency) without
  extra tooling. We still publish a `requirements.txt` (exported from the
  `uv`-managed dependency set) for anyone who wants a plain-pip install
  path without adopting uv themselves — `pyproject.toml`/`uv.lock` remain
  the source of truth.

**Consequences:** Contributors need `uv` installed locally
(`pip install uv` or the official installer); documented in the README's
Installation section.

## 3. Cross-Cutting Risk: Dual-Write Consistency

Ingestion writes to three stores (MongoDB, Qdrant, Elasticsearch) with no
distributed transaction across them — the sprint plan calls this out as
the top technical risk. Mitigation, reflected in NFR-4: each store write
is validated independently (a document isn't marked `status: ready` in
MongoDB until both the Qdrant upsert and the Elasticsearch index call
succeed), and a partial-failure path is a retryable ingestion job, not a
silently half-indexed document. This is validated explicitly on Day 3
("verify each store independently answers sample queries" per the sprint
plan) before hybrid retrieval is built on top of it.

## 4. Deviations From the Suggested Tooling List

The sprint brief's suggested dev-tooling list includes both Ruff and
Black. This repo uses **Ruff alone**, including `ruff format` (Ruff's
Black-compatible formatter), instead of Ruff + Black side by side. Running
two formatters against the same files is a common source of CI flapping
(each can nudge formatting the other then "fixes," or they simply disagree
on an edge case) for zero quality benefit — `ruff format` targets the same
output style as Black and is maintained by the same project as the linter
already in use here, so lint and format share one config block and one
tool invocation. If a future need arises that `ruff format` genuinely
can't cover, that's a one-line `pyproject.toml` change, not a rearchitect.
