# Experiments

Working notes from iterating on the pipeline -- not polished docs, a lab
notebook. Referenced from `docs/01-requirements.md` milestones.

Starting a new experiment? Copy `_template.md` rather than writing one from
scratch:

```bash
cp docs/experiments/_template.md docs/experiments/YYYY-MM-DD-short-title.md
```

The template has one section per thing every experiment here needs to
answer: **Hypothesis, Configuration, Dataset, Metrics, Results, Decision,
Next Steps.** A logged experiment with no stated hypothesis or decision
isn't useful later -- it's just an unlabeled table of numbers.

- `retrieval_debugging.md` (Day 3, appended Day 4) -- top-k BM25 vs. vector
  result comparisons, overlap analysis, and latency measurements that
  informed the RRF `k` and `retrieval_top_k` defaults in
  `configs/settings.py`; Day 4 appended live query-latency profiling and the
  `RERANK_INPUT_TOP_K` benchmark.
- `evaluation_notes.md` (Day 5) -- golden-dataset design, evaluation
  methodology, baseline retrieval/generation/refusal/latency metrics, a
  confidence-vs-correctness correlation follow-up, and observed failure
  modes. No chunking/retrieval/prompt defaults were changed this session
  (measure-first optimization policy) -- two real gaps found (refusal
  accuracy, reranker latency) are documented but left for a future measured
  experiment, not tuned same-session.
- `evaluation_notes_gpt54nano.md` (Day 5 follow-up) -- gpt-5.6-luna vs.
  gpt-5.4-nano model comparison on the identical golden dataset/corpus/config,
  including per-token cost measurement and a repeat-run finding that
  materially affects how the comparison should be read (same-model
  run-to-run variance measured directly, not just asserted).
