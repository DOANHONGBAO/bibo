"""Search the persistent index using the configured embedding model."""

import time

from bibo.config import Settings
from bibo.embeddings.encoder import Encoder, load_encoder
from bibo.storage.vector_store import SearchHit, VectorStore


def search_passages(
    query: str, *, top_k: int, settings: Settings, store: VectorStore,
    encoder: Encoder | None = None, timings: dict[str, float] | None = None,
) -> list[SearchHit]:
    total_start = time.perf_counter()
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not store.count():
        if timings is not None:
            timings["total"] = time.perf_counter() - total_start
        return []
    started = time.perf_counter()
    encoder = encoder or load_encoder(settings.model_id, settings.embedding_dimension,
                                     settings.model_cache, settings.batch_size,
                                     settings.model_dir)
    if timings is not None:
        timings["model_initialization"] = time.perf_counter() - started
    started = time.perf_counter()
    vector = encoder.encode([query])[0]
    if timings is not None:
        timings["query_embedding"] = time.perf_counter() - started
        metrics = getattr(encoder, "last_metrics", None)
        if metrics is not None:
            timings["tokenization"] = metrics.tokenization_seconds
            timings["inference"] = metrics.inference_seconds
    started = time.perf_counter()
    hits = store.search(vector, top_k)
    if timings is not None:
        timings["vector_retrieval"] = time.perf_counter() - started
        timings["total"] = time.perf_counter() - total_start
    return hits
