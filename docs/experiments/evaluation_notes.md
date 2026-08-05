# Evaluation Notes

**Date:** 2026-08-05
**Author:** Terry Nyirenda
**Status:** Complete (Sprint 5 Day 5)

Engineering notebook for the golden-dataset evaluation harness (`evaluation/`,
Sprint 5 Day 5). Unlike `retrieval_debugging.md`'s toy-corpus mechanism checks,
this is the real recall@K/precision@K/faithfulness/refusal measurement
`docs/01-requirements.md` §7 is gated on. See `docs/architecture.md` §2.10 for
the RAGAS+DeepEval ADR this harness implements.

## Dataset design

`data/golden/golden_qa.jsonl` — 30 hand-authored cases (22 positive, 8 negative),
one JSON object per line, extendable without touching any Python.

**Corpus.** Sprint planning left `data/raw/sample_contracts/` empty (gitignored,
per its README — no synthetic contracts allowed). For this evaluation to mean
anything, the corpus had to be real: 7 contracts sourced from SEC EDGAR exhibit
filings (real, publicly filed agreements — an employment agreement, an NDA, a
SaaS/master subscription agreement, a commercial lease, a consulting agreement, a
supply agreement, and a patent license agreement), spanning all 10 required clause
topics. EDGAR exhibits are filed as HTML, not PDF; each was converted to PDF with
`reportlab`, preserving the real filed text unaltered — only the container format
changed. See the corpus list below.

| File | Type | Source |
|---|---|---|
| `AtriCure_Employment_Agreement.pdf` | Employment Agreement | AtriCure, Inc. 8-K (2007-01-09), Ex-10.1 |
| `Benson_Hill_Consulting_Agreement.pdf` | Consulting Agreement | Benson Hill, Inc. 8-K (2023-06-16), Ex-10.2 |
| `BioShaft_Water_Technology_NDA.pdf` | NDA | BioShaft Water Technology, Inc. 10-Q (2012-09-14), Ex-10.16 |
| `LPL_Financial_SaaS_Services_Agreement.pdf` | SaaS/Services Agreement | LPL Financial Holdings Inc. 10-K (2021-02-23), Ex-10.24 |
| `Natural_Alternatives_Patent_License_Agreement.pdf` | Licensing Agreement | Natural Alternatives Int'l Inc. 8-K (2005-12-09), Ex-10.6 |
| `Unified_Western_Grocers_Supply_Agreement.pdf` | Vendor/Supply Agreement | Unified Western Grocers Inc. 10-Q (2004-02-10), Ex-10.62 |
| `Vista_International_Commercial_Lease.pdf` | Commercial Lease | Vista International Technologies Inc. 10-Q (2011-08-15), Ex-10.1 |

A pre-existing 8th document, `Legal-Aid-Manual_VERSION-7...pdf` (uploaded during
earlier sprint days), stays in the corpus but is never referenced by the golden
set — its procedural, non-contract content makes it a useful real distractor
document for negative cases, rather than a source of golden answers.

**Positive cases** (22): 2 per topic (termination, confidentiality,
indemnification, payment obligations, force majeure, governing law, dispute
resolution, liability, assignment, notices), each with `expected_answer`,
`expected_documents` (by filename — see "Why filenames, not `doc_id`" below), and
a human-readable `expected_citations` hint (e.g. `"Section 8(c) - Termination by
the Company for Cause"`). Every fact in every `expected_answer` was checked
directly against the extracted PDF text before being written, not inferred or
paraphrased from memory of "what a typical contract says."

**Negative cases** (8, 2 each of 4 subtypes, tagged `negative_subtype`):
- `nonexistent_clause` — asks about a clause type genuinely absent from the named
  document (e.g. a non-compete period in an NDA that has no non-compete clause).
- `misleading` — asserts a false premise borrowed from a *different* real document
  in the corpus (e.g. attributing the lease's per-acre rent structure to the SaaS
  agreement), so a model that pattern-matches on contract-ish language without
  checking the premise is likely to answer anyway.
- `hallucination_trap` — asks for a specific number/fact (a dollar cap, a per-diem
  late fee) that plausibly *could* exist in a contract like this one, but doesn't
  in this specific document, probing whether the model invents a plausible-sounding
  figure by analogy to other contracts it's seen chunks of.
- `unrelated` — completely outside the legal domain (boiling point of water, 1998
  World Cup winner), the baseline refusal case.

**Why filenames, not `doc_id`, as ground truth.** `docs/01-requirements.md` §7.1
originally envisioned exact-match against `expected_doc_id`. In practice,
`scripts/seed_corpus.py` assigns a fresh random UUID per ingestion run — there is
no stable `doc_id` to pin the dataset to across re-ingestion. The harness resolves
`expected_documents` (filenames) to the *live* `doc_id`s once per run via
`evaluation.dataset.resolve_filename_doc_ids`, which queries `DocumentRepository`
and raises immediately if a referenced filename isn't currently ingested — a
loud failure instead of a silently-wrong recall number. This is a deliberate,
documented deviation from the Day 1 planning language, not an oversight.

## Evaluation methodology

`evaluation/harness.py::run_evaluation` runs every case through the same
`HybridRetriever` → `CrossEncoderReranker` → `GenerationService` chain
`generation.pipeline.QueryPipeline` uses in production (`docs/architecture.md`
§2.1) — reimplemented stage-by-stage in `evaluation/runner.py::run_case`, not
called via `QueryPipeline.answer()` directly, solely so per-stage latency can be
captured without adding evaluation-only instrumentation to the production
pipeline. Retrieval-strategy comparison (dense/sparse/hybrid) additionally calls
`DenseRetriever`/`SparseRetriever` directly, after the timed hybrid call, so those
extra calls never inflate the reported retrieval latency.

For each case: retrieve (hybrid, timed; dense/sparse, untimed) → rerank → generate
→ score refusal (always) → score retrieval recall/precision (positive cases only)
→ score RAGAS generation metrics (positive, non-refused cases with an
`expected_answer` only) → classify failure stage if any. `scripts/run_evaluation.py`
is the reproducible entrypoint (`make evaluate`), wiring real dependencies exactly
like `scripts/seed_corpus.py` does, and writing both a timestamped and a `latest`
report (JSON + Markdown) to `evaluation/reports/` (gitignored — generated output,
not source).

Reproduce with:

```bash
docker compose up -d mongo qdrant elasticsearch
uv run python scripts/seed_corpus.py     # once, to seed the 7 contracts + 1 pre-existing doc
uv run python scripts/run_evaluation.py  # or: make evaluate
```

Requires a valid `OPENAI_API_KEY` — both generation and the RAGAS judge call the
configured `LLM_MODEL`.

## Metric definitions

**Retrieval — Recall@5/10, Precision@5/10** (`evaluation/metrics/retrieval.py`),
computed directly rather than via RAGAS, per `docs/architecture.md` §2.10 ("straightforward
to compute directly against the golden set's known-relevant chunks"). Relevance is
judged at the **document level**: a retrieved chunk counts as relevant if its
`doc_id` resolves to one of the case's `expected_documents`. This is a deliberate
simplification — chunk IDs, like `doc_id`s, aren't stable across re-ingestion, so
there's no stable chunk-level ground truth to hand-label against (see "Why
filenames" above). Recall@K = (expected documents found in top K) / (expected
documents); Precision@K = (top-K chunks from an expected document) / min(K,
chunks retrieved). Computed for dense-only, sparse-only, and hybrid, per task
requirement to compare strategies rather than only the production hybrid path.

**Generation — Faithfulness, Context Precision, Context Recall, Answer Relevancy**
(`evaluation/metrics/generation.py`), via RAGAS's `ragas.metrics.collections`
LLM-as-judge classes (`Faithfulness`, `ContextPrecisionWithReference`,
`ContextRecall`, `AnswerRelevancy`), scored only for positive, non-refused cases
with an `expected_answer` (refusals and negative cases have no meaningful
"faithfulness to context" to measure). `retrieved_contexts` is the exact evidence
text handed to the LLM in production — `reranked[:RERANK_TOP_K]` — not the full
reranked candidate list. **Assumption, per the sprint brief's ask to document
library requirements:** the RAGAS judge and Answer Relevancy's embedding model
both call OpenAI directly (`AsyncOpenAI`, not the sync client
`generation.providers.get_openai_client()` uses — RAGAS's `.ascore()` requires an
async-capable client) — a real cost and dependency distinct from the system's own
local `bge-m3` dense-retrieval embeddings, which never leave the machine. Judge
temperature is whatever `llm_factory`'s defaults resolve to; RAGAS's LLM-judge
scores are therefore not perfectly deterministic run-to-run, per
`docs/architecture.md` §2.10's stated consequence.

**DeepEval threshold pass rate** (`PrecomputedScoreMetric` in the same file):
per the ADR, DeepEval's role here is the "CI/test-runner integration layer" around
RAGAS's already-computed scores, not a second LLM-judge pass over the same four
metrics. Each RAGAS score is wrapped in a `deepeval.metrics.BaseMetric` subclass
exposing `.is_successful()` against a threshold — faithfulness's threshold (0.90)
mirrors `docs/01-requirements.md` §7.4's acceptance criterion; the other three
(0.70 each) have no documented contractual bar yet and are evaluation-only
placeholders pending Day 6 calibration against real runs. This produces per-metric
pass rates in the report without a second, redundant LLM call per metric.

**Refusal — accuracy, false refusals, false acceptances**
(`evaluation/metrics/refusal.py`), computed for every case (not just negatives).
A **false acceptance** (answering a case with no supporting evidence) is reported
separately from accuracy, not averaged into it — for a legal tool, an unflagged
hallucination is a materially worse failure than an overly cautious refusal, and
collapsing both into one accuracy number would hide that asymmetry.

**Latency**: average retrieval (hybrid call alone), reranker, generation, and
end-to-end (sum of the three) across all 30 cases.

## Error analysis / failure classification

`evaluation/error_analysis.py::classify_case` cascades through the pipeline in
execution order for every case that isn't a clean success:

1. **`retrieval_failure`** — the expected document never appears in the hybrid
   top 10.
2. **`reranker_failure`** — it was in the hybrid top 10, but the cross-encoder
   dropped it before generation.
3. **`refusal_failure`** — relevant evidence survived reranking, but the pipeline
   refused anyway (positive case) or answered anyway (negative case — the more
   dangerous direction).
4. **`generation_failure`** — the pipeline answered, cited a document other than
   the expected one, or scored below `FAITHFULNESS_REVIEW_THRESHOLD` (0.5 — a
   looser, "worth a human look" bar distinct from the 0.90 ship/no-ship threshold).

The first applicable stage is blamed even if a later stage also looks imperfect —
citation accuracy is moot if the reranker never surfaced the right chunk.

## Baseline results

**Run date:** 2026-08-05. **Environment:** local Docker Compose (Mongo,
Qdrant v1.11.0, Elasticsearch 8.15.0), `BAAI/bge-m3` dense embeddings +
`BAAI/bge-reranker-v2-m3` reranking on CPU (consistent with
`retrieval_debugging.md`'s Day 3-4 runs — same machine, no GPU acceleration path).
**LLM:** `gpt-5.6-luna` for both generation and the RAGAS judge (see "RAGAS
`gpt-5.6` compatibility bug" below for why that needed a workaround) — a
deliberate substitution for this run, not a change to the project's configured
default (`gpt-4.1-mini`, unchanged in `.env.example`/`configs/settings.py`).

### Retrieval

| Strategy | Recall@5 | Recall@10 | Precision@5 | Precision@10 | Cases |
|---|---|---|---|---|---|
| Dense | 1.00 | 1.00 | 0.81 | 0.71 | 22 |
| Sparse | 1.00 | 1.00 | 0.89 | 0.73 | 22 |
| Hybrid | 1.00 | 1.00 | 0.88 | 0.77 | 22 |

### Generation (RAGAS)

| Metric | Score | DeepEval pass rate |
|---|---|---|
| Faithfulness | 0.95 | 91% |
| Context Precision | 0.65 | 41% |
| Context Recall | 0.96 | 91% |
| Answer Relevancy | 0.80 | 73% |

Scored cases: 22 (positive, non-refused).

### Refusal

- Accuracy: 93.33% (28/30 cases)
- False refusals: 0
- False acceptances: 2 — `negative-nonexistent-01`, `negative-misleading-01`

### Latency (avg across 30 cases)

| Stage | Time |
|---|---|
| Retrieval (hybrid) | 0.330s |
| Reranker | 53.468s |
| Generation | 2.571s |
| End-to-end | 56.370s |

Full report: `evaluation/reports/latest.md` / `latest.json` (generated, gitignored
— regenerate with `make evaluate`).

## Observations

**Against `docs/01-requirements.md` §7's acceptance criteria:**

| # | Criterion | Result | Verdict |
|---|---|---|---|
| 7.1 | Citation accuracy ≥ 90% | No positive case cited a wrong source document (22/22); one (`payment-02`) had a low-faithfulness answer despite correct citations | **Pass** (informally — no dedicated "citation accuracy" field was computed as a named metric; derived from the failure list and per-case citation doc_ids) |
| 7.2 | Retrieval recall@10 ≥ 0.85 | Hybrid recall@10 = 1.00 | **Pass** |
| 7.3 | Retrieval precision@5 ≥ 0.70 | Hybrid precision@5 = 0.88 | **Pass** |
| 7.4 | Faithfulness ≥ 0.90 | 0.95 | **Pass** |
| 7.5 | 100% refusal on negative cases | 6/8 = 75% (2 false acceptances) | **Fail** |
| 7.6 | CI gate provably enforced | Not attempted — Day 6 scope | N/A this session |
| 7.7 | `docker compose up` reproducibility | Not re-verified this session (last confirmed Day 4) | N/A this session |

**1. Retrieval recall is saturated at this corpus size, precision is where
strategies actually differ.** Recall@5 and Recall@10 are a perfect 1.00 for
*all three* strategies — with 8 documents (~189 chunks total) and
`RETRIEVAL_TOP_K=50` per source, the expected document essentially always
appears somewhere in the fused candidate pool. That makes recall@K
uninformative at this corpus size; precision@K is the metric actually
discriminating between strategies here. On precision, **sparse (BM25) alone
(0.89 @5) slightly edges out hybrid (0.88 @5)**, both ahead of dense alone
(0.81 @5) — plausible given each contract's clause vocabulary is fairly
distinctive (different companies, different defined terms), a setting where
literal term matching does most of the work and semantic generalization
mostly doesn't need to correct it. This doesn't contradict
`docs/architecture.md` §2.5's case for hybrid retrieval (Day 3's toy-corpus
run showed the opposite pattern, dense correcting sparse's misses via RRF on
paraphrased queries) — it says the two corpora exercise different failure
modes, and a larger/more heterogeneous corpus would be needed before drawing
a general conclusion about hybrid's precision edge over sparse-alone.

**2. Context Precision (0.65, 41% pass rate) is the weakest generation
metric, while Context Recall (0.96) and Faithfulness (0.95) are strong.**
This combination is the signature of the pipeline handing the LLM *more*
context than most questions need: `RERANK_TOP_K=8` puts up to 8 reranked
chunks in front of the model regardless of whether the question needs 1
chunk or all 8, so RAGAS's context-precision judge (which penalizes
irrelevant chunks in the provided context set) marks the surplus chunks
down, even though the model itself mostly ignores them when answering
(hence faithfulness stays high) and the *relevant* chunk is essentially
always somewhere in the 8 (hence context recall stays high). This is a
genuine, measured finding — but per the sprint's optimization policy, it is
**not** grounds to lower `RERANK_TOP_K` today; it's a candidate for a future
measured experiment (`docs/experiments/_template.md`) that would need to
check the precision gain against faithfulness/recall on harder,
multi-chunk-synthesis questions before changing the default.

**3. The refusal gate has a real gap on two of four negative-case subtypes.**
Both `hallucination_trap` (fabricated specific numbers) and `unrelated`
(fully out-of-domain) cases were refused 4/4 — those are the "easy" negative
cases, where either nothing relevant retrieves at all (top score near zero,
e.g. `negative-unrelated-02`'s `top_score=0.0066`) or the model has no
plausible number to reach for. The two false acceptances were both cases
where retrieval returned *topically real* content — `negative-nonexistent-01`
retrieved genuine BioShaft NDA chunks (just none of them about a non-compete,
since the NDA has no such clause) with a top rerank score of 0.87; the model
answered anyway rather than recognizing the specific fact wasn't present.
`negative-misleading-01` retrieved a mix of real Vista Lease and LPL SaaS
chunks (the false premise mixed the two), and the model apparently accepted
the premise rather than flagging the cross-document mismatch. In both cases
`generation_min_context_score` (0.35) wasn't the binding constraint — evidence
scored well above it — so this is a **generation-stage** gap (the LLM judging
plausible-looking retrieved context as "sufficient" when it doesn't actually
answer the specific question asked), not a retrieval or threshold-tuning
problem. Fixing it credibly needs prompt-level changes (e.g. more explicit
instruction to check whether the retrieved evidence answers the *specific*
question, not just whether it's topically related) — a prompt change is
exactly the kind of thing the sprint's optimization policy says needs a
measured before/after, not a same-session tweak.

**4. Latency is ~9.4x over the NFR-1 budget, and it's the reranker, full
stop.** 53.468s of the 56.370s average end-to-end latency (94.9%) is the
cross-encoder reranker — consistent with `retrieval_debugging.md`'s Day 4
profiling run, which found the same component at 97.6% of a single query's
latency. This is not a new finding, just a confirmation at full
golden-dataset scale that the already-documented CPU-only reranker bottleneck
(no GPU/DirectML/OpenVINO acceleration path active on this machine) is the
dominant cost by a wide margin, and that `RERANK_INPUT_TOP_K=20` (chosen in
that Day 4 run) does not by itself get latency anywhere close to the 6s NFR-1
target — a further reduction, GPU acceleration, or both would be needed, and
either warrants its own measured experiment against this baseline before
changing.

## Confidence correlation (follow-up measurement)

Follow-up to the "What the `confidence` field means" section (README) and
`domain.generation.GenerationResult.confidence`'s own docstring, both of which
describe `confidence` (the top reranked chunk's `rerank_score`) as a
retrieval-relevance proxy, not a calibrated answer-correctness signal —
asserted from first principles (how the value is computed), not previously
measured against outcomes. This section measures it, without redesigning the
metric (out of scope for this sprint).

**Method** (`scripts/confidence_correlation.py`): re-ran retrieval → rerank →
generation for all 30 golden cases (reusing `evaluation.runner.run_case`, so
`confidence` is defined identically to production — `reranked[0].rerank_score`,
`None` if nothing retrieved), scored Faithfulness only (not the other three
RAGAS metrics, to limit cost) for the 22 positive, non-refused cases, and
derived a binary `correct` label per case via the same
`evaluation.error_analysis.classify_case` cascade the main harness uses (a
case is "correct" iff it isn't classified as any kind of failure). LLM:
`gpt-5.6-luna`, same one-off substitution as the main baseline run.

### Results

| Comparison | Pearson r | n |
|---|---|---|
| Confidence vs. correct (all 30 cases) | −0.091 | 30 |
| Confidence vs. correct (22 positive cases only) | −0.031 | 22 |
| Confidence vs. faithfulness (scored cases only) | −0.052 | 22 |

**All three correlations are approximately zero** — none are even weakly
positive. Higher reported `confidence` does not predict a more correct or more
faithful answer in this dataset, in either direction.

### What the per-case data shows

Full per-case table (case_id, confidence, faithfulness, correct) is in the
script's own output; the pattern worth naming directly:

- **`payment-02`** — the one generation-stage failure from the main baseline
  run — had `confidence=0.917` (near the top of the distribution) and
  `faithfulness=0.000` (completely unfaithful). A caller trusting `confidence`
  here would have been maximally misled.
- **`negative-nonexistent-01`** and **`negative-misleading-01`** — the two
  false acceptances — had `confidence=0.984` and `0.795` respectively, both
  high. `confidence` gave no warning that these answers should have been
  refusals.
- **`negative-unrelated-01`/`02`** — the two fully out-of-domain questions —
  had `confidence≈0.004–0.007`, correctly near zero, and were correctly
  refused. This is `confidence` behaving exactly as designed: a strong signal
  *only* for "is there any topically relevant chunk at all," which is a real
  and useful signal, just a different one than "is this specific answer
  correct."

Put together: `confidence` reliably distinguishes "nothing relevant retrieved"
(near-zero, `negative-unrelated-*`) from "something relevant retrieved"
(everything else, roughly 0.6–1.0) — consistent with it gating the refusal
decision correctly in the "easy" negative cases. But once *something*
relevant is retrieved, its value carries no further information about whether
the resulting answer is actually correct or faithful — the two false
acceptances and the one unfaithful answer all had confidence in the same high
range as the 27 fully correct cases.

### Statistical power caveat

**This is a directional signal, not a statistically powered result.** N=22–30,
and critically, the "incorrect" class is tiny — only 1 of 22 positive cases
and 2 of 8 negative cases were incorrect. A Pearson/point-biserial correlation
estimated from 3 "incorrect" points against 27 "correct" points has enormous
sampling variance; a differently-composed set of 30 questions could plausibly
produce a noticeably different r purely from which few cases happen to land
on which side. The finding that should be trusted is the qualitative one
(confidence doesn't discriminate correct from incorrect once evidence is
found at all, illustrated concretely by `payment-02` and the two false
acceptances above), not the precise r values, which should not be quoted as
if they were a hypothesis-tested effect size.

### Conclusion

This does not change the `confidence` field's implementation or its
documented semantics (both already correctly describe it as a
retrieval-relevance proxy) — it empirically supports keeping that framing
rather than treating `confidence` as an answer-quality signal in any caller
code, UI display, or future automated gating logic. If a calibrated
answer-confidence score is ever wanted, this measurement is evidence that
`confidence` as currently computed would not serve that purpose without
redesign (explicitly out of scope here, per the sprint brief).

## RAGAS `gpt-5.6` compatibility bug

Not a project bug, but worth recording since it blocked this run and the fix is
now load-bearing code (`evaluation/_ragas_compat.py`). `ragas.llms.base
.InstructorLLM._map_openai_params` detects "reasoning models" that require
`max_completion_tokens` instead of `max_tokens` by parsing the model's version
number with `int()`. For `gpt-5.6-luna`, `int("5.6")` raises `ValueError`
(silently swallowed), so the remapping never happens and OpenAI rejects the
request outright (`Unsupported parameter: 'max_tokens'`). Confirmed as a live,
currently-open upstream issue —
[`vibrantlabsai/ragas#2708`](https://github.com/vibrantlabsai/ragas/issues/2708),
filed 2026-05-11 — with an unmerged fix,
[PR #2725](https://github.com/vibrantlabsai/ragas/pull/2725), whose core change is
exactly `int(version_str)` → `float(version_str)`.

`evaluation/_ragas_compat.py::ensure_ragas_dotted_version_support()` applies that
same one-line fix locally, but only after a live functional probe confirms the
installed ragas still exhibits the bug (constructs a minimal `InstructorLLM` with
a dotted-version model name and checks whether `max_tokens` survives unmapped) —
not a hardcoded `ragas.__version__` comparison, so the patch self-disables
automatically the moment ragas ships the real fix rather than needing maintenance
to detect that. `GenerationJudge.__init__` calls it once, idempotently. See that
module's docstring for the full rationale and the `TODO(ragas#2708)` marking what
to delete once upstream ships the fix.

Separately, RAGAS's `.ascore()` async methods require an async-capable OpenAI
client (`openai.AsyncOpenAI`) — handing `llm_factory` the project's shared *sync*
client (`generation.providers.get_openai_client()`) raises `TypeError` at the
first call. `GenerationJudge` constructs its own `AsyncOpenAI` client rather than
reusing that singleton.

## Known limitations

- **Document-level, not chunk-level, retrieval relevance** (see "Why filenames"
  and the retrieval metric definition above) — a real but deliberate
  simplification given `doc_id`/`chunk_id` instability across re-ingestion.
  Section labels in `expected_citations` are informational, not part of the
  automated pass/fail signal.
- **30 cases, not thousands** — this is a golden *spot-check* set, not a
  statistically powered benchmark. A single case flipping changes recall@10 by
  ~4.5 percentage points (1/22 positive cases). Read the numbers as directional,
  consistent with `docs/01-requirements.md` §7's own framing ("25-30 Q/A pairs").
- **RAGAS/DeepEval judge cost and non-determinism** — every faithfulness/context
  precision/context recall/answer relevancy score is an LLM call; re-running the
  harness will not reproduce bit-identical scores, only scores within noise of
  each other (`docs/architecture.md` §2.10's documented consequence). Treat
  small deltas near a threshold as "investigate," not "regression."
- **This run used `gpt-5.6-luna`, not the project's configured default
  (`gpt-4.1-mini`)** — a one-off substitution for this baseline run (`LLM_MODEL`
  environment override, not a `.env`/`configs/settings.py` change). A rerun with
  the default model would very likely produce different absolute numbers,
  particularly for latency and possibly faithfulness (different model, different
  judge). The `evaluation/_ragas_compat.py` workaround only matters when running
  against a dotted-minor-version model like this one — it's a no-op (after one
  cheap functional probe) against `gpt-4.1-mini` or any other unaffected model.
- **No CI quality gate wired yet** — `docs/01-requirements.md` §7.6 calls for
  demonstrating a degrade/restore CI gate "once, on the record, before v1.0 is
  tagged"; that's Day 6 scope per the requirements doc's own Day 6 success-metrics
  framing, not attempted here. This harness produces the numbers such a gate
  would assert against, but nothing in `.github/workflows/ci.yml` calls it yet.
- **Single-run baseline** — one run, one point in time, one model
  (`gpt-5.6-luna`). No variance/repeat-run analysis was performed at the time
  this baseline was written; the ADR's "small metric fluctuations near a
  threshold" guidance was a documented expectation, not something empirically
  characterized here. **Update:** a later repeat run of gpt-5.6-luna (done for
  `docs/experiments/evaluation_notes_gpt54nano.md`'s cost measurement)
  measured this directly — refusal accuracy alone moved from 93.33% to 100%
  between two runs of the identical configuration. See that document's "A
  repeat gpt-5.6-luna run changed the refusal outcome" section. The baseline
  numbers in this document are left as originally measured, not updated to
  match the repeat run, since this remains the canonical first-measurement
  record; treat both runs together as the actual evidence of this system's
  measurement noise.

## Next steps

- Day 6: wire an evaluation quality gate into CI (FR-12, §7.6's degrade/restore
  demonstration) using the thresholds in `evaluation/metrics/generation.py`'s
  `GENERATION_THRESHOLDS` and `docs/01-requirements.md` §7's other criteria.
- Re-run this baseline against the project's actual configured default
  (`gpt-4.1-mini`) for an apples-to-apples number against future changes, since
  this run's `gpt-5.6-luna` substitution was a one-off.
- Delete `evaluation/_ragas_compat.py` once
  [ragas#2708](https://github.com/vibrantlabsai/ragas/issues/2708) ships upstream
  (tracked via the module's own `TODO`).
- **Two real gaps found this run — neither tuned today, per the sprint's
  optimization policy (measure first, change later with its own before/after):**
  - §7.5's 100% negative-case refusal bar is currently missed (75%, 2 false
    acceptances) — Observation 3 above localizes this to generation-stage
    judgment on topically-real-but-non-answering context, not retrieval or the
    `generation_min_context_score` threshold. A prompt-level fix is the
    likely lever, but needs its own measured before/after
    (`docs/experiments/_template.md`) against this baseline, not a
    same-session tweak.
  - NFR-1's 6s P95 latency budget is missed by ~9.4x, entirely reranker-bound
    (Observation 4) — already known from Day 4's profiling, confirmed here at
    full golden-dataset scale. `RERANK_INPUT_TOP_K` reduction and/or GPU
    acceleration are the candidate levers; both need their own measured
    experiment against this baseline before changing the default.
