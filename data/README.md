# Data

Not a database -- this is local working storage for the ingestion pipeline and
evaluation harness.

- `raw/` -- uploaded source PDFs as received. Gitignored; populate locally or
  via `scripts/seed_corpus.py` (Day 2). A small sample contract corpus
  (5-10 documents) lives here during development.
- `processed/` -- intermediate ingestion artifacts (extracted text, chunk
  manifests) useful for debugging the pipeline without re-parsing PDFs.
  Gitignored.
- `golden/` -- the golden Q/A evaluation dataset (`golden_qa.jsonl`, Day 5),
  including negative/refusal cases. Committed -- this is a versioned test
  asset, not a data dump.
