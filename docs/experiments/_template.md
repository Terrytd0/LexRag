# Experiment: <short descriptive title>

**Date:** YYYY-MM-DD
**Author:**
**Status:** Draft | Running | Complete

## Hypothesis

What change are you testing, and what specific, falsifiable outcome do you
expect? "If we switch RRF `k` from 60 to 30, recall@10 will improve because
BM25 ranks get relatively more weight" -- not "let's see if this helps."

## Configuration

What changed vs. the current baseline -- config values, model, prompt,
chunk size, `RRF_K`, `RERANK_TOP_K`, etc. List only the delta, not the
whole pipeline. Link the commit/branch if code changed.

- **Baseline:**
- **Variant:**

## Dataset

Which dataset this ran against (usually `data/golden/golden_qa.jsonl`),
how many examples, and any subset/filtering applied.

## Metrics

Which metrics were measured and why they're the right ones for this
hypothesis (recall@K, precision@K, faithfulness, refusal accuracy,
latency, ...). Reference the thresholds in
[`docs/01-requirements.md` §7](../01-requirements.md#7-measurable-acceptance-criteria)
where relevant.

## Results

| Metric | Baseline | Variant | Δ |
|---|---|---|---|
| | | | |

Link the raw report if one was generated, e.g. `evaluation/reports/<file>`.

## Decision

**Ship / Reject / Iterate** -- one sentence, tied explicitly back to the
hypothesis and the acceptance criteria in `docs/01-requirements.md`, not
vibes. If shipped, note whether a default in `configs/settings.py` or an
ADR in `docs/architecture.md` needs updating to match.

## Next Steps

Follow-up experiments, open questions, or a link to the PR/ADR if this
becomes a permanent change.
