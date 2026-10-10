"""Import documents into a local vector index."""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import time

from bibo.config import Settings
from bibo.embeddings.encoder import Encoder, load_encoder
from bibo.ingestion.chunker import chunk_pages
from bibo.ingestion.parser import DocumentError, SUPPORTED_SUFFIXES, parse_document_cached
from bibo.storage.vector_store import VectorStore


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    filename: str
    chunk_count: int
    duplicate: bool = False
    repaired: bool = False
    changed: bool = False
    cache_hit: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    token_count: int = 0
    mean_tokens: float = 0.0
    p95_tokens: int = 0


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
    total_start = time.perf_counter()
    timings: dict[str, float] = {}
    path = path.expanduser().resolve()
    if not path.is_file():
        raise DocumentError(f"Document does not exist or is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise DocumentError(f"Unsupported document type: {path.suffix or '(none)'}")
    started = time.perf_counter()
    file_hash = file_sha256(path)
    timings["file_hash"] = time.perf_counter() - started
    document_id = file_hash
    state = store.document_state(document_id)
    if state.complete:
        if file_sha256(path) != file_hash:
            raise DocumentError(f"Document changed while checking the index: {path}; retry")
        timings["total"] = time.perf_counter() - total_start
        return IngestResult(document_id, path.name, state.count, True, timings=timings)
    changed = bool(store.document_ids_for_path(str(path)) - {document_id})
    started = time.perf_counter()
    pages, cache_hit = parse_document_cached(
        path, file_hash=file_hash, cache_dir=settings.data_dir / "parsed_cache",
    )
    timings["parsing"] = time.perf_counter() - started
    timings["text_normalization"] = 0.0  # Source text is kept verbatim.
    started = time.perf_counter()
    chunks = chunk_pages(
        pages, document_id=document_id, filename=path.name,
        source_path=str(path), file_hash=file_hash,
        size_words=settings.chunk_size_words,
        overlap_words=settings.chunk_overlap_words,
    )
    timings["chunking"] = time.perf_counter() - started
    if not chunks:
        raise DocumentError(f"No indexable text in {path}")
    expected_ids = {chunk.chunk_id for chunk in chunks}
    if state.count and (not state.consistent or state.expected_count != len(chunks)
                        or not state.ids.issubset(expected_ids)):
        raise DocumentError(
            f"Partial index for {path.name} is inconsistent; no vectors were deleted. "
            "Back up the database before manual repair."
        )
    missing = [chunk for chunk in chunks if chunk.chunk_id not in state.ids]
    if not missing:
        raise DocumentError(f"Index metadata for {path.name} is incomplete; manual repair required")
    started = time.perf_counter()
    encoder = encoder or load_encoder(settings.model_id, settings.embedding_dimension,
                                     settings.model_cache, settings.batch_size,
                                     settings.model_dir)
    timings["model_initialization"] = time.perf_counter() - started
    started = time.perf_counter()
    vectors = encoder.encode([chunk.content for chunk in missing])
    timings["embedding_total"] = time.perf_counter() - started
    if file_sha256(path) != file_hash:
        raise DocumentError(f"Document changed during ingestion: {path}; retry")
    metrics = getattr(encoder, "last_metrics", None)
    if metrics is not None:
        timings["tokenization"] = metrics.tokenization_seconds
        timings["embedding_inference"] = metrics.inference_seconds
    started = time.perf_counter()
    try:
        store.add_document(missing, vectors, batch_size=settings.batch_size,
                           progress=progress, total_chunks=len(chunks))
    except ValueError as exc:
        if store.document_complete(document_id):
            timings["total"] = time.perf_counter() - total_start
            return IngestResult(document_id, path.name, len(chunks), True,
                                timings=timings)
        raise DocumentError(
            f"Concurrent or inconsistent ingestion for {path.name}; retry when other "
            "ingestion processes finish. Existing vectors were kept."
        ) from exc
    timings["chroma_insertion"] = time.perf_counter() - started
    if not store.document_complete(document_id):
        raise DocumentError(f"Index for {path.name} remains partial after insertion")
    timings["total"] = time.perf_counter() - total_start
    return IngestResult(
        document_id, path.name, len(chunks), repaired=bool(state.count), changed=changed,
        cache_hit=cache_hit, timings=timings,
        token_count=metrics.tokens if metrics is not None else 0,
        mean_tokens=metrics.mean_tokens if metrics is not None else 0.0,
        p95_tokens=metrics.p95_tokens if metrics is not None else 0,
    )


def supported_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise DocumentError(f"Directory does not exist: {directory}")
    return sorted(path for path in directory.rglob("*")
                  if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)
