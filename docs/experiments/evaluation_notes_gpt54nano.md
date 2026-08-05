# Model Comparison: gpt-5.6-luna vs. gpt-5.4-nano

**Date:** 2026-08-05
**Author:** Terry Nyirenda
**Status:** Complete

Follow-up to `docs/experiments/evaluation_notes.md` (the main Sprint 5 Day 5
evaluation baseline). Re-runs the identical golden-dataset evaluation with only
`LLM_MODEL` changed, to isolate the model's effect on retrieval-independent
metrics from everything else the pipeline does.

This is a **separate experiment file**, not an edit to `evaluation_notes.md` —
the original gpt-5.6-luna baseline (dataset, corpus, report files, and prose)
is untouched. `evaluation/reports/latest.md`/`latest.json` still point at the
original gpt-5.6-luna baseline run; this comparison's reports live under
`evaluation/reports/model_comparison/<model>/`.

## Held identical between both runs

- **Golden dataset:** `data/golden/golden_qa.jsonl` (30 cases, unchanged).
- **Corpus:** the same 7 SEC-filed contracts + 1 pre-existing document,
  already ingested — no re-ingestion between runs.
- **Embeddings:** `BAAI/bge-m3` (`Settings.embedding_model`, unchanged).
- **Reranker:** `BAAI/bge-reranker-v2-m3` (`Settings.rerank_model`, unchanged).
- **Retrieval configuration:** `RETRIEVAL_TOP_K`, `RRF_K`, `RERANK_INPUT_TOP_K`,
  `RERANK_TOP_K` — all `configs/settings.py` defaults, unchanged.
- **Prompt template:** `generation/prompts.py`'s `LEGAL_RAG_V1`, unchanged.
- **Refusal threshold:** `GENERATION_MIN_CONTEXT_SCORE=0.35`, unchanged.
- **RAGAS metrics:** the same four (Faithfulness, Context Precision, Context
  Recall, Answer Relevancy), same thresholds
  (`evaluation/metrics/generation.py::GENERATION_THRESHOLDS`), unchanged.
- **Evaluation methodology:** the same `evaluation.harness.run_evaluation`
  orchestration, the same `evaluation.error_analysis.classify_case` failure
  cascade, the same latency instrumentation.

**The only variable changed:** `LLM_MODEL`, used for both the system's
generation calls and the RAGAS judge (same as the original gpt-5.6-luna
baseline's setup — see `evaluation_notes.md`'s "RAGAS `gpt-5.6` compatibility
bug" section for why the judge uses the same model rather than a separate
one).

No production configuration was changed to run this comparison —
`scripts/run_evaluation.py`, `evaluation/harness.py`, `configs/settings.py`,
and `.env`/`.env.example` are all untouched. A new, separate script
(`scripts/run_model_comparison.py`) wires its own OpenAI clients (needed for
token/cost instrumentation, absent from the regular harness) and calls the
same, unmodified `run_evaluation()`.

## Methodology addendum: cost tracking

The regular evaluation harness tracks no cost (see `evaluation_notes.md`) —
building that out was specific to this comparison. `evaluation/cost_tracking.py`
wraps each OpenAI client's `chat.completions.create`/`embeddings.create` to
record `response.usage`, applied to a **freshly constructed, unwrapped**
client before `ragas.llms.llm_factory` hands it to `instructor` for structured
output — instrumenting after that point sees only instructor's wrapper, whose
return type has no `.usage`. Verified working via a smoke test before either
paid run (confirmed usage was captured through the instructor wrapping layer
for the judge, and directly for generation).

**Pricing used** (standard, non-batch tier; USD per 1M tokens) — fetched
directly from OpenAI's published pricing page
(`https://developers.openai.com/api/docs/pricing`, verified via two
independent fetches on 2026-08-05) rather than estimated or recalled from
training data, since both models postdate this assistant's knowledge cutoff:

| Model | Input /1M | Output /1M |
|---|---|---|
| `gpt-5.4-nano` | $0.20 | $1.25 |
| `gpt-5.6-luna` (short context) | $0.20 | $1.20 |
| `text-embedding-3-small` (judge's Answer Relevancy embeddings) | $0.02 | — |

Both models' actual prompts in this evaluation run to a few thousand tokens
(system + up to 8 reranked chunks + question) — nowhere near any documented
long-context threshold for either model family — so the short-context/standard
rate applies to both.

**Cost is broken out by purpose**, not just totaled: `generation` (the
system's own answer-generation calls), `judge` (RAGAS's Faithfulness/Context
Precision/Context Recall/Answer Relevancy LLM calls), and `judge_embedding`
(Answer Relevancy's embedding calls, billed at `text-embedding-3-small`'s rate
regardless of `LLM_MODEL`, since that embedding model is fixed).

## Results

| Metric | gpt-5.6-luna (baseline) | gpt-5.4-nano | Δ (nano − luna) |
|---|---|---|---|
| Recall@10 (hybrid) | 1.00 | 1.00 | 0.00 |
| Precision@5 (hybrid) | 0.88 | 0.88 | 0.00 |
| Faithfulness | 0.95 | 0.94 | −0.01 |
| Context Precision | 0.65 | 0.62 | −0.03 |
| Context Recall | 0.96 | 0.95 | −0.01 |
| Answer Relevancy | 0.80 | 0.84 | +0.04 |
| Refusal accuracy | 93.33% (28/30) | 83.33% (25/30) | −10.0 pp |
| False acceptances | 2 | 5 | +3 |
| False refusals | 0 | 0 | 0 |
| Avg generation latency | 2.571s | 2.62s | +0.05s |
| Avg end-to-end latency | 56.370s | 55.93s | −0.44s |
| Total API cost (30 cases) | $0.2051 | $0.1927 | −$0.0124 |
| Cost per evaluated question | $0.0068 | $0.0064 | −$0.0004 |

Retrieval, faithfulness, context recall, and both latency figures for
gpt-5.6-luna are the original baseline's documented numbers
(`evaluation_notes.md`) — reused as-is per the decision not to re-derive them
from a third run. **Cost required a fresh gpt-5.6-luna run** (no tracking
existed for the original baseline), and that fresh run is the direct source of
the next section's most important finding.

## A repeat gpt-5.6-luna run changed the refusal outcome — read this before the comparison below

The gpt-5.6-luna run made specifically to measure cost (identical config, same
model, same code, same corpus, same dataset) produced **100% refusal accuracy
(0/8 false acceptances)** — not the original baseline's 93.33% (2/8 false
acceptances). Faithfulness (0.92 vs. 0.95), Context Precision (0.67 vs. 0.65),
Context Recall (0.98 vs. 0.96), and Answer Relevancy (0.79 vs. 0.80) also
shifted by small amounts. Full numbers, this repeat run:

| Metric | gpt-5.6-luna (original baseline) | gpt-5.6-luna (repeat run) |
|---|---|---|
| Faithfulness | 0.95 | 0.92 |
| Context Precision | 0.65 | 0.67 |
| Context Recall | 0.96 | 0.98 |
| Answer Relevancy | 0.80 | 0.79 |
| Refusal accuracy | 93.33% (28/30) | 100.00% (30/30) |
| False acceptances | 2 | 0 |
| Avg end-to-end latency | 56.370s | 53.79s |

This is the single most important number in this document, and it isn't a
model-comparison finding at all — it's **direct, measured evidence of how much
this harness's own numbers move between two runs of the identical
configuration**, something the main baseline could only assert from RAGAS's
documented non-determinism (`docs/architecture.md` §2.10), never previously
demonstrate. It changes how the nano-vs-luna comparison below should be read.

## Summary of differences

**Revised in light of the repeat-run finding above.** The first draft of this
document (written before the gpt-5.6-luna cost run returned) called the
refusal-accuracy gap "the one difference large enough to call meaningful."
That conclusion doesn't survive the repeat run: gpt-5.6-luna itself produced
0 false acceptances in one run and 2 in another, with nothing at all changed
between them. A same-model swing of that size means gpt-5.4-nano's 5 false
acceptances — worse than *both* observed gpt-5.6-luna runs, but only by 3 and
5 cases respectively out of 8 — is no longer confidently attributable to a
real model difference rather than the same kind of run-to-run noise just
demonstrated. It remains the most *suggestive* signal in this comparison
(nano was worse in the one run measured, and worse than luna's own range),
just not the settled finding the pre-repeat-run draft claimed.

**Retrieval is identical, as expected** — Recall@10 and Precision@5 are
bit-for-bit the same in both models' runs, since retrieval and reranking
don't depend on `LLM_MODEL` at all. This is a sanity check on the
comparison's isolation (confirms nothing else leaked between runs), not
itself a finding, and it's the one row in this comparison not subject to the
non-determinism problem above.

**Every RAGAS score and the refusal count moved between the two gpt-5.6-luna
runs by roughly the same magnitude as the nano-vs-luna deltas themselves** —
Faithfulness moved 0.03 between luna's two runs (0.95→0.92) vs. a 0.01
nano-vs-luna delta; Context Recall moved 0.02 (0.96→0.98) vs. a 0.01 delta;
Answer Relevancy moved 0.01 (0.80→0.79) vs. a 0.04 nano-vs-luna delta (the one
case where the model-vs-model gap exceeds observed same-model noise, though
still on a single measurement of each). None of the RAGAS score comparisons in
this document should be read as demonstrating a real quality difference
between the two models — the repeat run shows the measurement noise floor is
comparable to or larger than most of the deltas being compared.

## Statistical power caveat

**None of these deltas are hypothesis-tested, and the repeat gpt-5.6-luna run
demonstrates why that matters concretely, not just in the abstract.** n=30
cases (22 scored for RAGAS metrics, 8 for refusal) was always a small sample;
what changed is that this document no longer has to argue about that
abstractly — it has one real, measured instance of the *same* model
"disagreeing with itself" by 2 false acceptances and up to 0.03 on a RAGAS
score, from nothing but re-running. Given that, a 3-5 false-acceptance gap
and ≤0.04 RAGAS score gaps between two *different* models, each measured
exactly once, cannot be treated as reliable model differences. Establishing
one would need repeated runs of *each* model (not just gpt-5.6-luna) to
estimate a per-model variance, then a comparison that accounts for it — out of
scope for this document, and flagged here as unfinished rather than glossed
over.

## Cost conclusion

At near-identical published pricing (both $0.20/1M input; $1.25 vs. $1.20/1M
output), the measured dollar cost difference is small and in the direction of
gpt-5.4-nano being marginally *cheaper* ($0.1927 vs. $0.2051 total, $0.0064
vs. $0.0068 per question — about 6% less) — driven by gpt-5.6-luna's judge
calls producing more completion tokens (50,645 vs. 39,245) on this run, not by
a list-price difference. Given this document's own finding that gpt-5.6-luna's
token usage isn't stable run-to-run either (the repeat run's own numbers would
need comparing against a third luna run to know how much of *this* 6% gap is
noise vs. real), cost should be treated as a minor, uncertain factor, not a
deciding one, in either direction.

## Recommendation

**Do not switch the project's configured default away from `gpt-4.1-mini`
based on this comparison** — neither `gpt-5.6-luna` nor `gpt-5.4-nano` was
benchmarked against the actual configured default (`gpt-4.1-mini` has not been
measured at all yet, per `evaluation_notes.md`'s Next Steps), and this
comparison's own repeat-run finding means even the two-model comparison here
shouldn't be treated as settled. **The most important output of this exercise
is not "which model wins" — it's that a single run of this evaluation harness,
for any model, is not by itself sufficient evidence to change a model
default.** Before making any model decision from this harness's output:
1. Run each candidate model multiple times to establish its own variance
   (this document accidentally measured gpt-5.6-luna's variance once, from 2
   runs — that's a start, not a sample).
2. Only then compare models, against the estimated noise floor from (1), not
   against single-run point estimates as this document initially did.
3. Include `gpt-4.1-mini` in that comparison, since it's the actual production
   default and neither run here touched it.
