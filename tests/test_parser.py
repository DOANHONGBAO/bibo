from pathlib import Path

import pymupdf
import pytest

from bibo.ingestion.parser import DocumentError, parse_document, parse_document_cached
from bibo.ingestion.pipeline import file_sha256


def test_pdf_page_numbers(tmp_path):
    path = tmp_path / "sample.pdf"
    pdf = pymupdf.open()
    for sentence in ("First page research text", "Second page result text"):
        page = pdf.new_page()
        page.insert_text((72, 72), sentence)
    pdf.save(path)
    pdf.close()
    pages = parse_document(path)
    assert [page.page_number for page in pages] == [1, 2]
    assert "First page" in pages[0].content
    assert "Second page" in pages[1].content


@pytest.mark.parametrize("suffix,content", [('.txt', 'Xin chào nghiên cứu'),
                                           ('.md', '# Heading\nUseful findings')])
def test_text_formats(tmp_path, suffix, content):
    path = tmp_path / f"note{suffix}"
    path.write_text(content, encoding="utf-8")
    assert parse_document(path)[0].content == content
    assert parse_document(path)[0].page_number is None


def test_empty_unreadable_and_unsupported(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text(" \n ", encoding="utf-8")
    with pytest.raises(DocumentError, match="empty"):
        parse_document(empty)
    invalid = tmp_path / "invalid.md"
    invalid.write_bytes(b"\xff\xfe")
    with pytest.raises(DocumentError, match="UTF-8"):
        parse_document(invalid)
    unsupported = tmp_path / "paper.docx"
    unsupported.write_bytes(b"dummy")
    with pytest.raises(DocumentError, match="Unsupported"):
        parse_document(unsupported)


def test_scanned_pdf_reports_ocr_limitation(tmp_path):
    path = tmp_path / "scan.pdf"
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.save(path)
    pdf.close()
    with pytest.raises(DocumentError, match="OCR"):
        parse_document(path)


def test_pdf_cache_reuses_text_without_parsing_again(tmp_path, monkeypatch):
    path = tmp_path / "cached.pdf"
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Cached research text")
    pdf.save(path)
    pdf.close()
    file_hash = file_sha256(path)
    cache_dir = tmp_path / "cache"
    pages, hit = parse_document_cached(path, file_hash=file_hash, cache_dir=cache_dir)
    assert not hit and pages[0].page_number == 1
    monkeypatch.setattr("bibo.ingestion.parser.parse_document",
                        lambda _: (_ for _ in ()).throw(AssertionError("parsed twice")))
    cached, hit = parse_document_cached(path, file_hash=file_hash, cache_dir=cache_dir)
    assert hit and cached == pages
