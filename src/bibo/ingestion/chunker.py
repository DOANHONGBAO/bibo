"""Word-window chunking with preferred paragraph and sentence breaks."""

from dataclasses import dataclass
import hashlib
import re

from bibo.ingestion.parser import PageText


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    filename: str
    source_path: str
    page_number: int | None
    file_hash: str


def chunk_pages(
    pages: list[PageText], *, document_id: str, filename: str, source_path: str,
    file_hash: str, size_words: int = 500, overlap_words: int = 75,
) -> list[Chunk]:
    if size_words < 1 or not 0 <= overlap_words < size_words:
        raise ValueError("Invalid chunk size or overlap")
    chunks: list[Chunk] = []
    for page in pages:
        text = page.content
        words = list(re.finditer(r"\S+", text))
        start = 0
        while start < len(words):
            maximum_end = min(start + size_words, len(words))
            end = maximum_end
            if maximum_end < len(words):
                # Prefer a boundary in the last fifth of the target window.
                earliest = start + max(1, size_words * 4 // 5)
                for candidate in range(maximum_end, earliest - 1, -1):
                    gap = text[words[candidate - 1].end():words[candidate].start()]
                    previous = words[candidate - 1].group()
                    if "\n\n" in gap or previous.endswith((".", "?", "!", ":", ";")):
                        end = candidate
                        break
            content = text[words[start].start():words[end - 1].end()]
            index = len(chunks)
            chunk_id = hashlib.sha256(f"{document_id}:{index}:{content}".encode("utf-8")).hexdigest()
            chunks.append(Chunk(chunk_id, document_id, content, index, filename,
                                source_path, page.page_number, file_hash))
            if end == len(words):
                break
            start = max(start + 1, end - overlap_words)
    return chunks
