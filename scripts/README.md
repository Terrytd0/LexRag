# Scripts

One-off operational scripts, run via `uv run python scripts/<name>.py`. Not
imported as a package -- each script is a standalone entrypoint reusing the
application's existing settings, clients, and pipeline code.

- `seed_corpus.py` -- ingest every PDF in `data/raw/sample_contracts/` through
  `ingestion.pipeline.IngestionPipeline`.
- `run_evaluation.py` (Day 5, also `make evaluate`) -- execute the golden
  dataset (`data/golden/golden_qa.jsonl`) through the full retrieval +
  generation pipeline and write a metrics report to `evaluation/reports/`.
  See `docs/experiments/evaluation_notes.md`.
- `confidence_correlation.py` -- one-off analysis correlating the API's
  `confidence` field against measured answer correctness/faithfulness across
  the golden dataset; does not modify the harness. See
  `docs/experiments/evaluation_notes.md`'s "Confidence correlation" section.
- `run_model_comparison.py` -- one-off, cost-instrumented re-run of the full
  evaluation with only `LLM_MODEL` changed, for comparing candidate models;
  writes to `evaluation/reports/model_comparison/<model>/`, never touching
  `evaluation/reports/latest.*`. See
  `docs/experiments/evaluation_notes_gpt54nano.md`.
- `evaluation_gate.py` (Day 6, also `make evaluate-gate`) -- check an
  already-generated evaluation report (default: `evaluation/reports/latest.json`)
  against `docs/01-requirements.md` §7's thresholds; exits non-zero on
  failure. The CI mechanism behind FR-12. See
  `docs/experiments/evaluation_notes_day6.md`.
- `rerank_backend_benchmark.py` (Day 6, one-off) -- compares the
  cross-encoder reranker's `torch` vs. `onnx` inference backends for
  latency and score/ranking consistency against the golden set. Backing
  `docs/adr/001-reranker-onnx-backend.md`.
- `rerank_input_topk_validation.py` (Day 6, one-off) -- validates candidate
  `RERANK_INPUT_TOP_K` values against real golden-set ground truth (does
  every positive case's expected document survive reranking?). See
  `docs/experiments/evaluation_notes_day6.md` §2b.

Planned:

- `reset_stores.py` -- drop and recreate the Qdrant collection and
  Elasticsearch index from their current mappings, for local dev resets.
