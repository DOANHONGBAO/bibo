"""Read local documents without modifying the source files."""

from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
PARSER_VERSION = "pymupdf-text-v1"
PARSER_CONFIG = "get_text:text;sort:false;no_ocr"


class DocumentError(ValueError):
    """A document cannot be indexed."""


@dataclass(frozen=True)
class PageText:
    content: str
    page_number: int | None = None


def parse_document(path: Path) -> list[PageText]:
    path = path.expanduser()
    if not path.exists():
        raise DocumentError(f"Document does not exist: {path}")
    if not path.is_file():
        raise DocumentError(f"Not a file: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentError(f"Unsupported document type: {suffix or '(none)'}")
    if suffix == ".pdf":
        import pymupdf

        try:
            with pymupdf.open(path) as pdf:
                pages = [PageText(page.get_text("text"), page.number + 1) for page in pdf]
        except (pymupdf.FileDataError, RuntimeError) as exc:
            raise DocumentError(f"Cannot read PDF {path}: {exc}") from exc
        if not any(page.content.strip() for page in pages):
            raise DocumentError(f"No extractable text in {path}; scanned PDFs need OCR, which is not supported")
        return pages
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise DocumentError(f"Cannot decode {path} as UTF-8") from exc
    except OSError as exc:
        raise DocumentError(f"Cannot read {path}: {exc}") from exc
    if not content.strip():
        raise DocumentError(f"Document is empty: {path}")
    return [PageText(content)]


def parse_document_cached(
    path: Path, *, file_hash: str, cache_dir: Path,
) -> tuple[list[PageText], bool]:
    """Cache PDF text by content, parser version and extraction settings."""
    if path.suffix.lower() != ".pdf":
        return parse_document(path), False
    import pymupdf

    parser_version = f"{PARSER_VERSION}:{pymupdf.VersionBind}"
    key = hashlib.sha256(
        f"{file_hash}:{parser_version}:{PARSER_CONFIG}".encode("utf-8")
    ).hexdigest()
    cache_file = cache_dir / f"{key}.json.gz"
    if cache_file.is_file():
        try:
            with gzip.open(cache_file, "rt", encoding="utf-8") as source:
                payload = json.load(source)
            if (payload["file_hash"] == file_hash
                    and payload["parser_version"] == parser_version
                    and payload["parser_config"] == PARSER_CONFIG):
                pages = [PageText(item["content"], item["page_number"])
                         for item in payload["pages"]]
                if pages and any(page.content.strip() for page in pages):
                    return pages, True
        except (OSError, EOFError, ValueError, KeyError, TypeError):
            # A broken generated cache is replaced from the unchanged source.
            pass
    pages = parse_document(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "file_hash": file_hash, "parser_version": parser_version,
        "parser_config": PARSER_CONFIG, "filename": path.name,
        "source_path": str(path.resolve()),
        "pages": [{"content": page.content, "page_number": page.page_number}
                  for page in pages],
    }
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_dir, suffix=".tmp", delete=False,
        ) as temporary:
            temporary_name = Path(temporary.name)
        with gzip.open(temporary_name, "wt", encoding="utf-8") as target:
            json.dump(payload, target, ensure_ascii=False)
        os.replace(temporary_name, cache_file)
    finally:
        if temporary_name is not None:
            temporary_name.unlink(missing_ok=True)
    return pages, False
