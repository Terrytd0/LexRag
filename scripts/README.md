# Scripts

One-off operational scripts, run via `uv run python scripts/<name>.py`. Not
imported as a package -- each script is a standalone entrypoint reusing the
application's existing settings, clients, and pipeline code.

Planned:

- `seed_corpus.py` (Day 2) -- load the sample contract corpus into
  `data/raw/` and run it through ingestion.
- `run_eval.py` (Day 5) -- execute the golden dataset through the full
  retrieval + generation pipeline and write a metrics report.
- `reset_stores.py` -- drop and recreate the Qdrant collection and
  Elasticsearch index from their current mappings, for local dev resets.
