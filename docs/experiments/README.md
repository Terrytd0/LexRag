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

- `retrieval_debugging.md` (Day 3) -- top-k BM25 vs. vector result
  comparisons, overlap analysis, and latency measurements that informed the
  RRF `k` and `retrieval_top_k` defaults in `configs/settings.py`.
- `evaluation_notes.md` (Day 5) -- baseline metric scores and the chunking /
  retrieval / prompt changes made in response to them.
