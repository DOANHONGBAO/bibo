"""Search the persistent index using the configured embedding model."""

from bibo.config import Settings
from bibo.embeddings.encoder import Encoder, load_encoder
from bibo.storage.vector_store import SearchHit, VectorStore


def search_passages(
    query: str, *, top_k: int, settings: Settings, store: VectorStore,
    encoder: Encoder | None = None,
) -> list[SearchHit]:
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not store.count():
        return []
    encoder = encoder or load_encoder(settings.model_id, settings.embedding_dimension,
                                     settings.model_cache, settings.batch_size)
    return store.search(encoder.encode([query])[0], top_k)
