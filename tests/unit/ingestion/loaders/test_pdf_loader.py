from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter

from ingestion.loaders.pdf_loader import LoadedDocument, load_pdf


def _write_blank_pdf(path: Path, page_count: int) -> None:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)


def test_load_pdf_returns_one_entry_per_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _write_blank_pdf(pdf_path, page_count=3)

    loaded = load_pdf(pdf_path)

    assert isinstance(loaded, LoadedDocument)
    assert loaded.filename == "sample.pdf"
    assert len(loaded.pages) == 3


def test_load_pdf_blank_page_yields_empty_string(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank.pdf"
    _write_blank_pdf(pdf_path, page_count=1)

    loaded = load_pdf(pdf_path)

    assert loaded.pages == [""]
