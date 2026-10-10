"""Read local documents without modifying the source files."""

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


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
        import fitz

        try:
            with fitz.open(path) as pdf:
                pages = [PageText(page.get_text("text"), page.number + 1) for page in pdf]
        except (fitz.FileDataError, RuntimeError) as exc:
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
