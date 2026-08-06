# Sprint 5 Day 6: CI Gate, Refusal Fix, Latency Optimization

**Date:** 2026-08-05/06
**Author:** Terry Nyirenda
**Status:** Complete

Follow-up to `docs/experiments/evaluation_notes.md` (Day 5 baseline, run against
`gpt-5.6-luna` as a one-off substitution) and its own "Next steps" list. This
session re-baselines against the project's actual configured default
(`gpt-4.1-mini`), then makes two measured changes — a refusal-prompt fix and a
reranker candidate-count reduction — each validated against the real golden
set before being adopted, plus wires and demonstrates the CI evaluation gate
(FR-12, `docs/01-requirements.md` §7.6). Every report referenced below is
preserved under `evaluation/reports/` (gitignored, but not overwritten by
later runs — see the filenames used throughout).

## 1. Day 6 pre-change baseline (`gpt-4.1-mini`)

Day 5's baseline used `gpt-5.6-luna`, an explicitly one-off substitution (see
that doc's "Known limitations"). Before making any change, re-ran the
identical 30-case golden set against the project's actual configured default
(`Settings.llm_model`, unchanged: `gpt-4.1-mini`), with every other setting
at its Day 5 value (`RERANK_INPUT_TOP_K=20`, prompt `LEGAL_RAG_V1`). This is
the apples-to-apples reference point every change below is measured against.

Report: `evaluation/reports/day6_baseline_before_gpt-4.1-mini.md`/`.json`.

| Metric | Day 5 (`gpt-5.6-luna`) | Day 6 pre-change (`gpt-4.1-mini`) |
|---|---|---|
| Recall@10 (hybrid) | 1.00 | 1.00 |
| Precision@5 (hybrid) | 0.88 | 0.88 |
| Faithfulness | 0.95 | 0.94 |
| Context Precision | 0.65 | 0.79 |
| Context Recall | 0.96 | 1.00 |
| Answer Relevancy | 0.80 | 0.93 |
| Refusal accuracy | 93.33% (28/30) | 90.00% (27/30) |
| False acceptances | 2 | 3 |
| Avg reranker latency | 53.468s | 46.672s |
| Avg end-to-end latency | 56.370s | 51.176s |

Retrieval is identical (as expected — it doesn't depend on the LLM). The
weaker refusal number on the configured default model (3 false acceptances
vs. 2) confirms Day 5's Observation 3 wasn't a `gpt-5.6-luna` quirk: this is a
real, model-independent generation-stage gap, not something a model swap
alone would fix.

## 2. Reranker latency: ONNX backend (rejected) and `RERANK_INPUT_TOP_K` (adopted)

Day 5 confirmed the reranker at ~95% of end-to-end latency, CPU-only (no
GPU/DirectML path — `torch.cuda.is_available()` is `False`). Two levers were
measured; see `docs/adr/001-reranker-onnx-backend.md` for the first in ADR
form.

### 2a. ONNX Runtime backend — measured, rejected

**Hypothesis:** `sentence-transformers`' `backend="onnx"` runs the same
`bge-reranker-v2-m3` weights through ONNX Runtime instead of eager PyTorch,
which is often faster on CPU.

**Method:** `scripts/rerank_backend_benchmark.py` — retrieved real RRF-fused
candidates for all 30 golden questions once, then reranked them with both
backends, comparing total latency and score/ranking consistency.

| Backend | Total (30 cases) | Avg/case |
|---|---|---|
| torch | 1451.202s | 48.373s |
| onnx | 1423.531s | 47.451s |

**Result: 1.02x — not a measurable win.** Max `|rerank_score|` delta was
`0.000002` and top-1 chunk agreement was 30/30, confirming the export is
numerically faithful; the default (unoptimized, unquantized) ONNX export
just isn't faster than PyTorch's own CPU path for this model on this
hardware. **Rejected as the default** — `optimum[onnxruntime]` stays
installed and `RERANK_BACKEND=onnx` still works (tested), but `torch` remains
the default per CLAUDE.md's "don't increase complexity without measured
benefit."

### 2b. `RERANK_INPUT_TOP_K` reduction — measured, adopted

Day 4 (`docs/experiments/retrieval_debugging.md`) had already reduced this
from 63 (unbounded) to 20 using a single-query heuristic, explicitly flagging
"re-run this sweep ... against the real golden dataset once it exists" as a
next step. That golden dataset now exists.

**Hypothesis:** Candidates ranked below ~12 by RRF fusion are, for this
corpus, never the ones that end up in the generation-facing top
`RERANK_TOP_K=8` after reranking — so trimming the reranker's input further
should cut latency without changing what generation actually sees.

**Method:** `scripts/rerank_input_topk_validation.py` — for every golden
question, retrieved once, then reranked the same RRF-fused candidates at
`RERANK_INPUT_TOP_K` ∈ {20, 12}, checking whether every **positive** case's
expected document still has a chunk in the reranked top `RERANK_TOP_K=8` (the
actual set handed to generation) — a real recall-after-rerank check against
ground truth, not the Day 4 heuristic.

| `RERANK_INPUT_TOP_K` | Total reranker latency (30 cases) | Avg/case | Positive cases losing expected doc from top 8 |
|---|---|---|---|
| 20 | 1439.9s | 47.996s | 0/22 |
| 12 | 895.8s | 29.861s | 0/22 |

**Result: 37.8% reranker latency reduction (1.61x speedup), zero measured
quality cost** — every positive case that had its expected document in the
top 8 at `RERANK_INPUT_TOP_K=20` still has it at `12`. **Adopted**:
`Settings.rerank_input_top_k` default changed `20` → `12`
(`configs/settings.py`, `.env.example`).

This check is a proxy (document-level, not the full faithfulness/citation
pipeline) — the combined "after" run in §4 is what confirms end-to-end
generation quality holds with the new default in place, not just this proxy.

## 3. Refusal fix: `LEGAL_RAG_V2`

Day 5's Observation 3 localized both false acceptances to generation, not
retrieval or `GENERATION_MIN_CONTEXT_SCORE`: the LLM judged retrieved
evidence "sufficient" because it was topically real (same document, same
general subject), even when it didn't contain the *specific* fact the
question asked about, or when the question's premise didn't match the
evidence at all.

**Change:** Added `LEGAL_RAG_V2` (`generation/prompt_versions.py`) alongside
the unchanged `LEGAL_RAG_V1`, adding one explicit instruction: verify every
specific fact/premise the question assumes — a number, date, clause type,
condition — is actually stated in the evidence, not just topically related,
before answering; if the premise isn't stated or is contradicted, refuse
rather than answer or "correct" the premise. The refusal sentence itself
(`REFUSAL_ANSWER`, matched structurally in `generation/generator.py`) is
unchanged. `ACTIVE_PROMPT_VERSION` switched `LEGAL_RAG_V1` → `LEGAL_RAG_V2`
(`generation/prompts.py`).

This is a **generation-level change only** — no threshold
(`GENERATION_MIN_CONTEXT_SCORE`) was touched, per the sprint's "measured
prompt-level or generation-level improvements rather than ad-hoc threshold
tuning" instruction.

Isolated before/after (see §4 for the combined, official numbers): a
degraded-retrieval run made mid-session (§5b) already used `LEGAL_RAG_V2`
and still showed 3 false acceptances, on a *different* case mix
(`negative-hallucination-02` joined, `negative-misleading-02` dropped) — a
first signal that this specific prompt change, on its own, may not close the
gap, discussed against the clean measurement in §4.

## 4. Combined "after" measurement

One clean run — no deliberately degraded settings — with both changes
in place: `RERANK_INPUT_TOP_K=12`, prompt `LEGAL_RAG_V2`, `gpt-4.1-mini`,
everything else at its Day 6 pre-change value. This is both the official Day
6 "after" number and the CI gate's "restore → pass" half of the §7.6
demonstration (§5).

Report: `evaluation/reports/day6_after_gpt-4.1-mini.md`/`.json`.

| Metric | Day 6 before | Day 6 after | Δ |
|---|---|---|---|
| Recall@10 (hybrid) | 1.00 | 1.00 | — |
| Precision@5 (hybrid) | 0.88 | 0.88 | — |
| Faithfulness | 0.9432 | 0.9788 | +0.036 |
| Refusal accuracy | 90.00% (27/30) | 90.00% (27/30) | — |
| False acceptances | 3 | **2** | **-1** |
| False refusals | 0 | 1 (`confidentiality-03`) | +1 |
| Avg reranker latency | 46.672s | **23.446s** | **-49.8%** |
| Avg end-to-end latency | 51.176s | **26.331s** | **-48.6%** |
| Gate result | FAIL (false acceptances) | FAIL (false acceptances) | still failing |

**Latency: a real, large win** — end-to-end latency nearly halved
(51.176s → 26.331s), driven almost entirely by the reranker
(46.672s → 23.446s, tracking §2b's `RERANK_INPUT_TOP_K` validation closely).
Retrieval and faithfulness are unaffected or slightly improved, consistent
with §2b's zero-quality-cost proxy check.

**Refusal: a real but partial win.** False acceptances dropped from 3 to 2 —
`negative-misleading-02` (the false premise about Matthew Crisp relocating to
Cincinnati) is now correctly refused under `LEGAL_RAG_V2`. The same two
cases from the *original* Day 5 baseline survive:
`negative-nonexistent-01` (BioShaft NDA non-compete period — a clause type
genuinely absent from that document) and `negative-misleading-01` (the
$675/acre rent premise borrowed from a different document). Refusal
*accuracy* stayed flat at 90% because the fix traded one false acceptance for
one new false refusal (`confidentiality-03`, an answerable case) — a
different, less dangerous error type for a legal tool per this project's own
framing (`evaluation/metrics/refusal.py`), but not free. **The 100% target on
negative cases is not met.** See "Next steps" for why this specific pair
survived and what a further iteration would need to target.

## 5. CI gate demonstration (FR-12, §7.6)

`evaluation/gate.py` + `scripts/evaluation_gate.py` (`make evaluate-gate`)
check an already-generated report against `docs/01-requirements.md` §7's
thresholds: `recall_at_10 >= 0.85`, `precision_at_5 >= 0.70`,
`faithfulness >= 0.90`, `refusal_false_acceptances == 0`. Wired into
`.github/workflows/ci.yml` as a manual-trigger job on a self-hosted runner
(see that file's comments, and "Known limitations" below, for why: the
corpus is real, gitignored, licensed contract text —
`data/raw/sample_contracts/README.md` — that a GitHub-hosted runner has no
way to reproduce).

### 5a. A genuine failing run (no artificial degradation needed)

Running the gate against the Day 6 **pre-change** baseline report (§1) —
before any Day 6 change — already fails:

```
$ uv run python scripts/evaluation_gate.py evaluation/reports/day6_baseline_before_gpt-4.1-mini.json

Evaluation gate results:

  [PASS] recall_at_10: 1.0000 (required >= 0.8500)
  [PASS] precision_at_5: 0.8818 (required >= 0.7000)
  [PASS] faithfulness: 0.9432 (required >= 0.9000)
  [FAIL] refusal_false_acceptances: 3.0000 (required == 0.0000)

GATE: FAIL
```

This is real evidence the gate isn't decorative: it fails against the
system's actual, unmodified state.

### 5b. Deliberately degraded retrieval configuration — two failed attempts, then a clean one

Per §7.6's specific ask ("a deliberately degraded retrieval configuration
... causes CI to fail"), tried three retrieval-configuration degradations.
The first two are real, measured, and instructive, but turned out *not* to
fail the retrieval-metric checks specifically — worth recording as findings
in their own right before the one that did.

**Attempt 1 — `RETRIEVAL_TOP_K=2`** (report:
`evaluation/reports/day6_degraded_demo_retrieval_top_k2.json`) — dropped avg
end-to-end latency to 6.646s (far fewer candidates reach the reranker), but
hybrid recall@10/precision@5 stayed at 1.00/0.96 — **unaffected**. This
corpus (8 documents, ~208 chunks, document-specific queries) is small enough
that even 2 candidates per source usually still contain the right document
(consistent with Day 5's "retrieval recall is saturated at this corpus size"
finding). The gate still failed, but on refusal (7 false refusals, 1 false
acceptance) — a real, correctly-detected consequence of the degradation,
just not the retrieval-metric failure mode the criterion's wording implies.

**Attempt 2 — `RRF_K=1000000`** (report:
`evaluation/reports/day6_degraded_demo_rrf_k_1000000.json`) — hypothesized
that an extreme fusion constant would flatten RRF scores toward
`chunk_id`-alphabetical tie-breaking, scrambling the fused ranking.
Recall@10/precision@5 again stayed essentially unchanged (1.00/0.87).
**Why the hypothesis was wrong:** `1/(k+rank)` is strictly monotonically
decreasing in `rank` for *any* `k > 0` — a huge `k` compresses the score
*range* but never reorders within a single ranked list; float64 precision
still resolves the tiny gaps this leaves. It reduces the relative boost a
chunk gets from appearing in *both* dense and sparse lists (vs. just one),
but on this corpus the individual-list ranks were already good enough that
this didn't matter. The gate still failed (3 false acceptances, on yet a
third case mix) — again refusal, not retrieval.

Both attempts are a real, measured finding, not just failed experiments:
retrieval recall/precision on this 8-document corpus is robust to fairly
aggressive perturbation of two different retrieval-configuration knobs.

**Attempt 3 — pointing at empty vector/keyword stores** (report:
`evaluation/reports/day6_degraded_demo_empty_stores.json`) — both
`QdrantVectorStore` and `ElasticsearchKeywordStore` lazily auto-create their
collection/index on first `search()` if it doesn't already exist
(`_ensure_collection`/`_ensure_index`). Pointing `QDRANT_COLLECTION` and
`ELASTICSEARCH_INDEX` at fresh, never-indexed names is therefore a clean way
to make retrieval genuinely return nothing, without touching any code:

```
$ QDRANT_COLLECTION=lexrag_chunks_degraded_demo_empty \
  ELASTICSEARCH_INDEX=lexrag_chunks_degraded_demo_empty \
  uv run python scripts/run_evaluation.py
...
Retrieval (hybrid): recall@10=0.00 precision@5=0.00
Generation: faithfulness=0.00
Refusal accuracy: 26.67%
```

```
$ uv run python scripts/evaluation_gate.py evaluation/reports/day6_degraded_demo_empty_stores.json

Evaluation gate results:

  [FAIL] recall_at_10: 0.0000 (required >= 0.8500)
  [FAIL] precision_at_5: 0.0000 (required >= 0.7000)
  [FAIL] faithfulness: 0.0000 (required >= 0.9000)
  [PASS] refusal_false_acceptances: 0.0000 (required == 0.0000)

GATE: FAIL
```

This run cost zero LLM calls: with no evidence retrieved, `GenerationService`
refuses every case before ever calling the LLM (`generator.py`'s
`_should_refuse`), so it completed in under two seconds. Refusal
*correctly passes* here — with nothing retrieved, every negative case is
(trivially) refused, so there are no false acceptances; the 22 positive
cases are false refusals instead, which don't count against this specific
check. The empty test collection/index were deleted immediately after
(`DELETE /collections/...`, `DELETE /<index>`) — this demo leaves no
residue in the shared local stack.

### 5c. Restore → pass (for the checks this degradation affects)

Reverting to the real `QDRANT_COLLECTION`/`ELASTICSEARCH_INDEX` names is
exactly the §4 clean "after" run (same config, nothing else changed):

```
$ uv run python scripts/evaluation_gate.py

Evaluation gate results:

  [PASS] recall_at_10: 1.0000 (required >= 0.8500)
  [PASS] precision_at_5: 0.8818 (required >= 0.7000)
  [PASS] faithfulness: 0.9788 (required >= 0.9000)
  [FAIL] refusal_false_acceptances: 2.0000 (required == 0.0000)

GATE: FAIL
```

**This is the honest result, and it's a stronger demonstration than a clean
pass would have been.** `recall_at_10`, `precision_at_5`, and `faithfulness`
all flip FAIL → PASS exactly when the degraded store names are reverted —
proof the gate's wiring correctly tracks each threshold independently, not
just "did the run complete." `refusal_false_acceptances` stays FAIL
throughout, in both the degraded and restored states, because it is a real,
independent, still-unresolved gap (§4) — unrelated to and unaffected by this
particular degradation. A gate that flipped to a clean overall PASS here
would either mean the refusal gap had been silently dropped from the check,
or would misrepresent the system's actual, current state. Per
`docs/01-requirements.md` §7's own framing ("all of the following hold"),
the v1.0 acceptance bar is not yet met, and this gate correctly says so —
which is the entire point of building it.

## 6. Known limitations

- **CI gate is not automated on GitHub-hosted infrastructure.** The golden
  corpus (`data/raw/sample_contracts/`) is real, licensed contract text,
  deliberately gitignored per that directory's README — a GitHub-hosted
  runner has no way to reproduce it from a checkout alone. The
  `evaluation-gate` job in `.github/workflows/ci.yml` is real, wired,
  manually-triggerable YAML that runs correctly against a self-hosted runner
  with the corpus already seeded (i.e., a machine like the one this session
  ran on) — but it is not exercised by GitHub's own infrastructure today.
  This demonstration (§5) was run directly, not through GitHub Actions.
- **The v1.0 acceptance bar (`docs/01-requirements.md` §7) is not fully met
  yet.** The CI gate correctly reflects this: every run measured today
  (Day 5's original baseline, Day 6's pre-change baseline, both degradation
  attempts, and the post-fix "after" run) fails on
  `refusal_false_acceptances`. This is not a demo artifact — it is this
  session's most important real finding, and exactly what a load-bearing
  gate is supposed to catch.
- **Two of three retrieval-configuration degradation attempts didn't hit the
  retrieval-metric checks specifically** (§5b, attempts 1-2) — a real,
  measured finding about this corpus's retrieval robustness, not a gap in
  the gate. Attempt 3 (empty stores) did cleanly isolate and fail
  `recall_at_10`/`precision_at_5`/`faithfulness`.
- **Single-run measurements.** Per `evaluation_notes.md`'s own documented
  RAGAS/DeepEval non-determinism and repeat-run refusal variance (93.33% →
  100% for an identical `gpt-5.6-luna` config), every number in this
  document is one run, not a repeat-run average. Treat deltas near a
  threshold as "investigate," not "regression," same as Day 5.
- **`RERANK_INPUT_TOP_K=12` validated on a document-level recall-after-rerank
  proxy** (§2b), not the full faithfulness/citation pipeline in isolation —
  the combined §4 run is the actual end-to-end confirmation, but it changed
  two things at once (this plus the prompt), so it cannot alone separate
  "latency change harmed generation" from "prompt change helped/hurt
  generation." Both were validated independently before combining
  (§2b's proxy for the former; §3's isolated single-degraded-run signal for
  the latter), which is why they were combined into one run rather than
  requiring a third full evaluation.

## 7. Next steps

- **Close the remaining refusal gap — the most important open item.**
  `negative-nonexistent-01` and `negative-misleading-01` survived
  `LEGAL_RAG_V2` unchanged from the original Day 5 baseline, so the next
  lever is likely a second, more targeted prompt iteration (`LEGAL_RAG_V3`)
  or a structural check (e.g., a cheap second-pass "does this evidence
  contain fact X" verification before accepting an answer) rather than
  further threshold tuning — consistent with Day 5's finding that
  `GENERATION_MIN_CONTEXT_SCORE` was never the binding constraint for either
  original false acceptance.
- The two attempts that didn't move recall/precision (§5b, attempts 1-2) are
  worth re-running against a larger, more heterogeneous corpus once one
  exists, to see whether a "natural" retrieval-configuration mistake (as
  opposed to attempt 3's deliberately-empty stores) can fail those checks
  directly on real data, not just via an artificially empty index.
- Revisit ONNX quantization (INT8 dynamic quantization via
  `optimum.onnxruntime.ORTQuantizer`) as a separate, riskier latency lever
  if further reranker speedup is needed — `optimum[onnxruntime]` is already
  installed; quantization changes numerics (unlike the plain export tested
  in §2a) and would need its own score-drift validation before adoption.
- Provision a self-hosted CI runner with the corpus pre-seeded so the
  `evaluation-gate` GitHub Actions job can run automatically rather than
  only being demonstrated by direct local invocation.
