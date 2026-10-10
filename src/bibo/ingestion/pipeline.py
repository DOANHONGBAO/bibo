"""Import documents into a local vector index."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

from bibo.config import Settings
from bibo.embeddings.encoder import Encoder, load_encoder
from bibo.ingestion.chunker import chunk_pages
from bibo.ingestion.parser import DocumentError, SUPPORTED_SUFFIXES, parse_document
from bibo.storage.vector_store import VectorStore


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    filename: str
    chunk_count: int
    duplicate: bool = False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_document(
    path: Path, *, settings: Settings, store: VectorStore,
    encoder: Encoder | None = None, progress: bool = True,
) -> IngestResult:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise DocumentError(f"Document does not exist or is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise DocumentError(f"Unsupported document type: {path.suffix or '(none)'}")
    file_hash = file_sha256(path)
    document_id = file_hash
    if store.document_complete(document_id):
        return IngestResult(document_id, path.name, store.document_chunk_count(document_id), True)
    if store.document_chunk_count(document_id):
        # A previous process may have stopped between Chroma batches.
        store.remove_document(document_id)
    pages = parse_document(path)
    chunks = chunk_pages(
        pages, document_id=document_id, filename=path.name,
        source_path=str(path), file_hash=file_hash,
        size_words=settings.chunk_size_words,
        overlap_words=settings.chunk_overlap_words,
    )
    if not chunks:
        raise DocumentError(f"No indexable text in {path}")
    encoder = encoder or load_encoder(settings.model_id, settings.embedding_dimension,
                                     settings.model_cache, settings.batch_size)
    vectors = encoder.encode([chunk.content for chunk in chunks])
    store.add_document(chunks, vectors, batch_size=settings.batch_size, progress=progress)
    return IngestResult(document_id, path.name, len(chunks))


def supported_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise DocumentError(f"Directory does not exist: {directory}")
    return sorted(path for path in directory.rglob("*")
                  if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)
