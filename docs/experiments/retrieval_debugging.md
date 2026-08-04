# Retrieval Debugging

Engineering notebook for the hybrid retrieval layer (`retrieval/`, Sprint 5
Day 3). Unlike the dated one-off experiments elsewhere in this directory,
this file accumulates across retrieval-tuning sessions -- append new dated
runs below rather than replacing this one. See `docs/architecture.md` §2.5-
§2.6 for why hybrid retrieval and RRF were chosen; this doc is where that
choice gets checked against real (if small) evidence.

## Methodology

Runs use `scripts/retrieval_debug_run.py`, which:

1. Indexes a small, hand-written set of 13 representative contract-clause
   chunks (indemnification, termination, confidentiality, non-compete,
   limitation of liability, force majeure, payment terms, IP assignment,
   dispute resolution, ...) directly into Qdrant + Elasticsearch via
   `QdrantVectorStore`/`ElasticsearchKeywordStore.index_chunks`, bypassing
   PDF loading/chunking (already covered by Day 2 tests).
2. Runs a fixed set of natural-language queries -- mostly paraphrased, not
   copied from the clause text, since that's the realistic query shape --
   through `DenseRetriever`, `SparseRetriever`, and `HybridRetriever`.
3. Prints per-query results, dense/sparse overlap, and latency for each
   retrieval mode.

Per `data/raw/sample_contracts/README.md`, synthetic contracts must never be
added to that corpus directory, so this fixture data stays in-process in the
script rather than on disk. **This is a mechanism check, not a quality
benchmark** -- 13 hand-written chunks can't produce a valid recall@10 or
precision@5 number against the FR-11/NFR thresholds in
`docs/01-requirements.md` §7; that requires the real golden dataset in
Day 5-6's evaluation harness. What this notebook *can* validate: that dense,
sparse, and RRF fusion behave the way `docs/architecture.md` §2.5-§2.6
predicts, on real Qdrant/Elasticsearch, with real (if modest) latency.

Reproduce with:

```bash
docker compose up -d mongo qdrant elasticsearch
uv run python scripts/retrieval_debug_run.py
```

---

## Run: 2026-08-03 (Day 3 implementation validation)

**Environment:** local Docker Compose stack (Qdrant v1.11.0, Elasticsearch
8.15.0), `BAAI/bge-m3` embeddings via `sentence-transformers` on CPU (no
GPU), `RRF_K=60` (config default), `top_k=5` per retriever (small on
purpose, to keep the six queries below readable -- production default is
`RETRIEVAL_TOP_K=50`).

### Sample queries and results

Each block shows the top-5 `chunk_id`s per mode, ranked best first.
`chunk_id` encodes `<doc_id>:<chunk_index>` -- see the clause table in
`scripts/retrieval_debug_run.py` for what each index means (e.g.
`msa-vendorco:1` = Termination, `msa-vendorco:0` = Indemnification).

**Query: "How can either party end the agreement early without cause?"**
(paraphrase of the Termination clause -- no shared vocabulary with
"terminate")

| Mode | Top-5 (best first) | Latency |
|---|---|---|
| Dense | `msa-vendorco:1`, `saas-agreement:9`, `msa-vendorco:0`, `msa-vendorco:2`, `licensing-agreement:12` | 343.7ms |
| Sparse | `msa-vendorco:1`, `msa-vendorco:2`, `nda-partnerco:4`, `msa-vendorco:3`, `msa-vendorco:0` | 237.0ms |
| Hybrid | `msa-vendorco:1`, `msa-vendorco:2`, `msa-vendorco:0`, `saas-agreement:9`, `nda-partnerco:4`, `msa-vendorco:3`, `licensing-agreement:12` (7) | 335.6ms |

Both modes correctly rank the Termination clause (`msa-vendorco:1`) first.
Overlap: 3/5.

**Query: "Who is responsible for losses if something goes wrong during the project?"**
(paraphrase of Indemnification -- no shared vocabulary with "indemnify")

| Mode | Top-5 (best first) | Latency |
|---|---|---|
| Dense | `msa-vendorco:0`, `saas-agreement:9`, `saas-agreement:8`, `licensing-agreement:11`, `msa-vendorco:1` | 606.7ms |
| Sparse | `employment-agreement:6`, `nda-partnerco:4`, `msa-vendorco:0`, `nda-partnerco:5`, `saas-agreement:8` | 63.6ms |
| Hybrid | `msa-vendorco:0`, `saas-agreement:8`, `employment-agreement:6`, `nda-partnerco:4`, `saas-agreement:9`, `licensing-agreement:11`, `nda-partnerco:5`, `msa-vendorco:1` (8) | 595.9ms |

Dense ranks the correct clause (`msa-vendorco:0`, Indemnification) #1;
sparse buries it at #3 behind two unrelated clauses that happen to share
generic words ("project", "during") with the query. This is the exact
failure mode `docs/architecture.md` §2.5 predicts for BM25 on paraphrased
queries. Overlap: 2/5.

**Query: "What happens if an employee starts working for a competitor?"**
(paraphrase of Non-Compete)

| Mode | Top-5 (best first) | Latency |
|---|---|---|
| Dense | `employment-agreement:6`, `licensing-agreement:11`, `employment-agreement:7`, `saas-agreement:8`, `msa-vendorco:0` | 352.1ms |
| Sparse | `employment-agreement:7`, `nda-partnerco:5`, `employment-agreement:6`, `saas-agreement:9`, `msa-vendorco:1` | 59.8ms |
| Hybrid | `employment-agreement:6`, `employment-agreement:7`, `licensing-agreement:11`, `nda-partnerco:5`, `saas-agreement:8`, `saas-agreement:9`, `msa-vendorco:0`, `msa-vendorco:1` (8) | 296.2ms |

Dense ranks the correct clause (`employment-agreement:6`, Non-Compete) #1;
sparse ranks the wrong sibling clause (`:7`, Compensation) #1 and the right
one #3 -- both employment-agreement chunks surface, but dense orders them
correctly. Overlap: 2/5.

**Query: "Section 8.3"** (exact clause-number lookup, not a paraphrase)

| Mode | Top-5 (best first) | Latency |
|---|---|---|
| Dense | `msa-vendorco:1`, `saas-agreement:8`, `msa-vendorco:0`, `employment-agreement:7`, `saas-agreement:10` | 179.6ms |
| Sparse | `msa-vendorco:1`, `saas-agreement:8` (2 results only) | 56.9ms |
| Hybrid | `msa-vendorco:1`, `saas-agreement:8`, `msa-vendorco:0`, `employment-agreement:7`, `saas-agreement:10` (5) | 217.2ms |

The inverse of the previous two: only two chunks in the corpus literally
contain "8.3" (`msa-vendorco:1` and `saas-agreement:8`, both of which
cross-reference "Section 8.3" from a different clause), and sparse finds
*exactly* those two with no noise. Dense finds the same two but ranked
#1/#2 among three unrelated chunks pulled in by general semantic
similarity -- the exact-match precision BM25 is good at, and embeddings
aren't (`docs/architecture.md` §2.4). Overlap: 2/5, but a *precise* 2/5 on
the sparse side.

**Query: "What is the interest rate on late invoice payments?"**
(close paraphrase of Payment Terms -- shares "interest"/"payment"/"late"
with the source text)

| Mode | Top-5 (best first) | Latency |
|---|---|---|
| Dense | `saas-agreement:10`, `saas-agreement:8`, `msa-vendorco:1`, `nda-partnerco:5`, `employment-agreement:7` | 325.4ms |
| Sparse | `saas-agreement:10`, `nda-partnerco:4`, `employment-agreement:6`, `employment-agreement:7`, `msa-vendorco:0` | 71.7ms |
| Hybrid | `saas-agreement:10`, `employment-agreement:7`, `nda-partnerco:4`, `saas-agreement:8`, `employment-agreement:6`, `msa-vendorco:1`, `nda-partnerco:5`, `msa-vendorco:0` (8) | 336.3ms |

Both modes agree on the #1 result (`saas-agreement:10`, Payment Terms) --
the query shares enough literal vocabulary with the clause that BM25 wins
without needing semantic generalization. Overlap: 2/5.

**Query: "How are disagreements between the parties resolved?"**
(paraphrase of Dispute Resolution -- "disagreements" vs. "dispute")

| Mode | Top-5 (best first) | Latency |
|---|---|---|
| Dense | `msa-vendorco:2`, `licensing-agreement:12`, `msa-vendorco:0`, `msa-vendorco:1`, `msa-vendorco:3` | 273.7ms |
| Sparse | `licensing-agreement:12`, `employment-agreement:6`, `employment-agreement:7`, `msa-vendorco:0`, `saas-agreement:8` | 53.1ms |
| **Hybrid** | **`licensing-agreement:12`**, `msa-vendorco:0`, `msa-vendorco:2`, `employment-agreement:6`, `employment-agreement:7`, `msa-vendorco:1`, `msa-vendorco:3`, `saas-agreement:8` (8) | 276.0ms |

**The interesting one.** Dense mis-ranks the correct clause
(`licensing-agreement:12`, Dispute Resolution) to #2, behind Confidentiality
-- "disagreements between the parties" apparently sits closer to
Confidentiality's general contract-relationship phrasing in embedding space
than expected. Sparse ranks it correctly at #1. RRF's combined score (rank 1
in sparse + rank 2 in dense) pulls it back to #1 in the hybrid output,
*correcting* the dense mis-ranking. Overlap: 2/5.

This is the clearest evidence in this run for why `docs/architecture.md`
§2.5 rejects vector-only retrieval: dense got this one wrong on its own, and
fusion fixed it without any query-specific tuning.

### Overlap analysis

| Query | Dense/Sparse overlap (of top-5) |
|---|---|
| "end the agreement early" | 3/5 |
| "responsible for losses" | 2/5 |
| "employee...competitor" | 2/5 |
| "Section 8.3" | 2/5 |
| "interest rate...late...payments" | 2/5 |
| "disagreements...resolved" | 2/5 |

Average overlap: **2.2/5 (44%)**. Dense and sparse agree on roughly less
than half their top candidates even in this small corpus -- consistent with
`docs/architecture.md` §2.5's premise that the two modes fail in different,
complementary ways, and validates running both rather than picking one.

Also note: `HybridRetriever` does **not** truncate the RRF-merged output
back down to `top_k` -- it returns the full union of both lists (5-8 items
here from two 5-item lists, since RRF's job is to merge and rank, not
filter). At production defaults (`RETRIEVAL_TOP_K=50` per list), that's a
correctly-sized candidate pool of up to 100 chunks for the Day 4
cross-encoder reranker to narrow down to `RERANK_TOP_K=8` --
`docs/architecture.md` §2.6/§2.7's two-stage design, confirmed working as
specified rather than assumed.

### Latency observations

| Stage | Avg (6 queries) |
|---|---|
| Dense (embed query + Qdrant search) | 346.9ms |
| Sparse (Elasticsearch BM25) | 90.4ms |
| Hybrid (dense + sparse, concurrent + RRF) | 342.9ms |

Two things worth calling out:

1. **Hybrid latency (342.9ms) ≈ dense latency alone (346.9ms), not
   dense + sparse (437.3ms).** This confirms `HybridRetriever.retrieve`'s
   `asyncio.gather`/`asyncio.to_thread` concurrency (`docs/architecture.md`
   §2.5, FR-7) is actually overlapping the two calls rather than running
   them sequentially -- the slower of the two (dense) sets the floor, sparse
   is "free" in wall-clock terms.
2. **Dense latency is dominated by query embedding, not the Qdrant search
   itself** -- `bge-m3` inference on CPU for a single short query is
   ~250-600ms here; Elasticsearch's BM25 search alone is consistently
   <100ms. At the ≤10k-chunk target corpus (NFR-1), Qdrant's HNSW search
   time is expected to stay near-constant as the corpus grows; the
   embedding cost per query won't. This still leaves ample headroom against
   the 6s P95 budget (NFR-1 covers retrieval + rerank + generation
   combined), but it's the retrieval-side cost most worth watching once the
   reranker (also CPU-bound, also loaded once) is added in Day 4.

Indexing 13 chunks into both stores (including one-time lazy collection/
index creation) took 10.41s total -- dominated by the first-call
`ensure_collection`/`ensure_index` round trip and embedding generation, not
by per-chunk overhead.

### Tuning ideas

- **RRF `k` sensitivity**: unit tests (`tests/unit/retrieval/fusion/test_rrf.py`)
  already confirm a smaller `k` weights top ranks more heavily. Worth an
  actual sweep (`k=10` vs. the `k=60` default) against the real golden
  dataset once it exists (Day 5-6), rather than this toy corpus.
- **Elasticsearch analyzer**: the current `legal_text_analyzer` (standard
  tokenizer + lowercase, no stopwords -- `docs/architecture.md` §2.4) has no
  synonym handling. Query 1 ("end the agreement early") only matched the
  Termination clause because both share incidental tokens ("party",
  "agreement"), not because of a semantic link to "terminate" -- a legal
  synonym filter (terminate/end, indemnify/hold harmless, etc.) would make
  sparse retrieval more robust to this kind of paraphrase without relying
  on dense to carry the whole burden.
- **The "disagreements" example (query 6) is worth keeping as a regression
  case** once the golden dataset exists -- it's a concrete instance of RRF
  correcting a dense-only ranking error, which is exactly the property
  hybrid retrieval is supposed to buy.

### Operational note: embedding model download

`BAAI/bge-m3` (~2.3GB) downloaded very slowly in this environment over
HuggingFace's `xet` CDN protocol (repeatedly stalling at <100KB/s as
"unauthenticated"; recovered somewhat after adding an `HF_TOKEN`, but still
took the better part of an hour end-to-end). This is a one-time cost per
machine (the model is cached under `~/.cache/huggingface` afterward), but
worth flagging for CI/onboarding: either ensure `HF_TOKEN` is set wherever
this runs, or consider pre-baking the model into a Docker image layer /
named volume so a flaky HF CDN connection doesn't become a recurring
CI-flake source. Not an architecture concern -- purely an operational one.

### Decision

**Iterate.** Hybrid retrieval and RRF behave as `docs/architecture.md`
§2.5-§2.6 predict on real Qdrant/Elasticsearch: dense and sparse disagree
often enough (44% overlap) to justify running both, each wins on the query
shape the ADR said it would (dense on paraphrase, sparse on exact terms),
and RRF measurably corrects at least one dense mis-ranking. No config
default changes are warranted from a 13-chunk toy corpus -- `RRF_K=60` and
`RETRIEVAL_TOP_K=50` stay as-is until the real golden dataset (Day 5-6)
gives a statistically meaningful signal to tune against.

### Next steps

- Day 4: cross-encoder reranker (`retrieval/reranker/`) consuming the
  RRF-merged candidate pool this run confirmed is correctly un-truncated.
- Day 4: wire `HybridRetriever` into `api/` via `Depends()` for `POST /query`.
- Day 5-6: replace this toy corpus with the real golden dataset and compute
  actual recall@10/precision@5 against the FR-11 thresholds -- this run
  validates *mechanism*, not *quality*.

---

## Run: 2026-08-04 (Day 4 query-latency profiling & RERANK_INPUT_TOP_K)

Follow-up to the Day 3 run above's first next-step item, once the Day 4
cross-encoder reranker (`retrieval/reranker/cross_encoder.py`) and `POST
/query` existed to profile end-to-end. Unlike the Day 3 run's 13-chunk toy
corpus, this one used the real live corpus (`msa-vendorco.pdf`,
`saas-agreement.pdf`, `nda-partnerco.pdf`, and a real 64-page document,
`Legal-Aid-Manual_VERSION-7...pdf`, 58 chunks -- 63 unique RRF-merged
candidates for the test query below) via a live server manually exercised
through Swagger, which surfaced a real operational problem: a 64-page
upload took 3+ minutes and pegged CPU at 95%, and a subsequent query hung
long enough that it was killed manually from Task Manager.

**Environment:** local Docker Compose stack, `BAAI/bge-m3` embeddings +
`BAAI/bge-reranker-v2-m3` reranking via `sentence-transformers` on CPU only
(`torch==2.13.0+cpu`, no CUDA; confirmed no GPU acceleration path is active
on this machine's Intel Iris Xe iGPU either -- would need `torch-directml`
or OpenVINO, neither installed).

### Stage-by-stage profile (one real query, models pre-warmed)

Query: *"Can either party terminate the agreement for convenience, and how
much notice is required?"*

| Stage | Time | % of total |
|---|---|---|
| Hybrid retrieval (dense+sparse+RRF, sequential sum) | 0.502s | 0.2% |
| -- Dense retrieval | 0.425s | 0.19% |
| -- Sparse retrieval | 0.077s | 0.03% |
| -- RRF fusion | <0.001s | ~0% |
| **Cross-encoder reranking** | **216.434s** | **97.6%** |
| Prompt construction | <0.001s | ~0% |
| LLM generation | 4.806s | 2.2% |
| Citation generation | <0.001s | ~0% |
| **TOTAL** | **221.743s** | 100% |

Candidate counts: 50 dense + 50 sparse -> 63 unique after RRF dedup -> all
63 passed to the reranker -> 8 kept (`RERANK_TOP_K`) -> 8 sent to the LLM.
Reranker cost: **~3,435ms/candidate** on CPU. `CrossEncoderReranker.rerank()`
scores every candidate handed to it before truncating to `RERANK_TOP_K` --
the top-k cutoff only trims the *output*, not the *input* -- so all 63
candidates paid the full cross-encoder cost even though only 8 were ever
used downstream. This, not embedding, dense/sparse search, RRF, or the LLM
call, is overwhelmingly the bottleneck.

(Separately, ingesting the 64-page/58-chunk PDF took 220.9s, also CPU-bound
embedding -- consistent with the same root cause: CPU-only transformer
inference with no GPU acceleration path configured.)

### RERANK_INPUT_TOP_K: bounding reranker cost

Introduced a new setting, `RERANK_INPUT_TOP_K` (`configs/settings.py`,
default `20`), applied in `generation/pipeline.py::QueryPipeline.answer()`
as `retrieved[: settings.rerank_input_top_k]` between `HybridRetriever
.retrieve` and `CrossEncoderReranker.rerank`. `HybridRetriever.retrieve`
itself is unmodified -- `RETRIEVAL_TOP_K`, dense/sparse search, and RRF
fusion are untouched, so retrieval-only metrics stay measurable
independently of this trim, per this doc's own §2.6 "measured separately"
goal referenced in the Day 3 run above.

No golden dataset exists yet (Day 5), so quality here is a single-query
self-consistency check: each candidate-count's reranked top-10 compared
against the untrimmed baseline's (63-candidate) reranked top-10 stands in
for "recall@10," not true labeled relevance. Take these numbers as directional.

**Baseline (63) vs. the chosen default (20), same shared RRF output:**

| Metric | Baseline | RERANK_INPUT_TOP_K=20 | Δ |
|---|---|---|---|
| Reranker latency | 192.5s | 59.8s | -132.7s |
| Total latency | 196.5s | 63.4s | -133.1s |
| Speedup | -- | **3.10x** | -- |
| Citations identical to baseline | -- | 7/8 | the one swap was noise-for-noise (an off-topic chunk for another); the one citation the answer actually cites, `msa-vendorco:1`, was unchanged and ranked [1] in both |
| Answer | "...60 days'... without cause and without penalty, subject to specific wind-down obligations... [1]" | "...60 days'... without cause and without penalty, subject to any specified wind-down obligations... [1]" | trivial paraphrase only |

Per-candidate reranker cost was ~3.0s in both runs (192.499s/63 = 3055.5ms,
59.778s/20 = 2988.9ms) -- confirms the linear-cost assumption behind this
lever.

**Sweep across 10/13/15/18/20** (same query, same shared RRF output):

| RERANK_INPUT_TOP_K | Reranker Latency | Total Latency | Recall@10 vs baseline | Citation Overlap (/8) | Answer Materially Changed |
|---|---|---|---|---|---|
| baseline (63) | 214.5s | 219.2s | 10/10 (ref) | 8/8 (ref) | N/A (reference) |
| 10 | 36.9s | 38.9s | 4/10 | 4/8 | False |
| 13 | 47.4s | 49.3s | 5/10 | 5/8 | False |
| 15 | 53.1s | 54.8s | 7/10 | 6/8 | False |
| 18 | 59.6s | 61.9s | 8/10 | 6/8 | False |
| 20 | 59.8s | 61.9s | 9/10 | 7/8 | False |

All six raw answers (baseline + 5 variants) state the same core facts (60
days' notice, without cause, without penalty, wind-down obligations, citing
`[1]` correctly) with only cosmetic wording differences -- the heuristic
"materially changed" flag (key-fact presence + `[1]` usage) was `False` at
every tested value, even at `RERANK_INPUT_TOP_K=10` where only 4/10 top
chunks and 4/8 citations matched baseline. That gap between "citation set
churned a lot" and "answer content didn't change" is itself notable: for
this query/corpus, only one candidate chunk (`msa-vendorco:1`) actually
mattered to the answer, and it survived RRF's ranking comfortably inside
the top 10 at every tested value -- the churn in citations 2-8 is mostly
noise from an otherwise-unrelated multi-document corpus, not evidence of
losing the relevant chunk. Recall@10 rises roughly monotonically with
`RERANK_INPUT_TOP_K`, and reranker latency rises roughly linearly with it
-- a fairly clean latency/recall-proxy trade-off along one dimension.

### Decision

**Iterate -- measurement only, no further change made this session beyond
the `RERANK_INPUT_TOP_K=20` default already set.** It held answer quality
and 7/8 citations for the one tested query at 3.10x lower latency. The
sweep shows real degradation in the recall proxy and raw citation overlap
as the value drops toward 10, even though the single-query "materially
changed" heuristic didn't catch it -- which is exactly why that heuristic
alone shouldn't be trusted to pick a final production value.

### Next steps

- **Do not tune `RERANK_INPUT_TOP_K` further from a single query.** Rerun
  this sweep (or a proper precision@K/recall@K + faithfulness eval) against
  the real golden dataset once it exists (Day 5-6) before treating any
  value here as validated.
- The golden set should specifically include queries where the relevant
  chunk sits outside the top ~15-20 of the RRF ranking, if any exist in the
  real corpus -- this run's corpus happened to rank the one relevant chunk
  very highly for this query, so the recall-proxy degradation at low
  `RERANK_INPUT_TOP_K` values may understate real-world risk for harder
  queries.
- Consider replacing the hand-picked key-fact "materially changed" check
  with a real reference-answer comparison (e.g. a RAGAS/DeepEval
  faithfulness or answer-similarity metric) once the golden dataset's
  expected answers exist.
- GPU/DirectML or OpenVINO acceleration for the reranker (and embedding
  model) is a separate, larger lever, orthogonal to this candidate-count
  trim -- both could stack. Not pursued this session; would need real
  validation against `bge-reranker-v2-m3`/`bge-m3` outputs before trusting
  it, per this machine's confirmed lack of an active GPU acceleration path.
