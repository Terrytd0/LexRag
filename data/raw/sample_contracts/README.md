# Sample Contracts

Local working corpus for developing and testing the ingestion pipeline. Place
PDF files directly in this directory -- `scripts/seed_corpus.py` walks it and
runs every `.pdf` it finds through `ingestion.pipeline.IngestionPipeline`.

This directory (and its contents, besides this file) is gitignored: no
contracts are committed to the repository. Populate it locally with:

- Publicly available or permissively licensed legal agreements (e.g. SEC
  EDGAR contract exhibits, government contract templates, Creative
  Commons-licensed agreement samples), or
- Your own documents, if you're experimenting with real casework.

Do not add synthetic/generated contracts here -- the point of this corpus is
to exercise the pipeline against real document structure (headers, page
breaks, defined-term formatting, exhibits) that a generated document won't
reproduce faithfully.

LexRAG is intended to ingest contract-style legal documents such as:

- Employment Agreements
- NDAs
- SaaS Agreements
- Vendor Agreements
- Lease Agreements
- Service Agreements
- Consulting Agreements
- Licensing Agreements

A handful of documents (5-10) spanning a few of these categories is enough to
exercise chunking, provenance metadata, and (from Day 3) hybrid retrieval
across genuinely different document structures.
