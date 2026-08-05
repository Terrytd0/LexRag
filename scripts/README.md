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

Planned:

- `reset_stores.py` -- drop and recreate the Qdrant collection and
  Elasticsearch index from their current mappings, for local dev resets.
