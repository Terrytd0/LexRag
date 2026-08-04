"""Shared raw-upload storage path, used by both `api/routes/upload.py` (writes)
and `api/routes/documents.py` (deletes) so the two agree on where a document's
original PDF lives on disk (`data/raw/{doc_id}/{filename}`).
"""

from __future__ import annotations

from pathlib import Path

RAW_STORAGE_DIR = Path("data/raw")
