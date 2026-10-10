"""BGE-M3 encoder, loaded once per process when needed."""

from pathlib import Path
from functools import lru_cache
from typing import Sequence

import numpy as np


class Encoder:
    def __init__(self, model_id: str, dimension: int, cache_dir: Path, batch_size: int) -> None:
        self.model_id = model_id
        self.dimension = dimension
        self.batch_size = batch_size
        cache_dir.mkdir(parents=True, exist_ok=True)
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        model_cache = cache_dir / f"models--{model_id.replace('/', '--')}" / "snapshots"
        cached = model_cache.exists() and any(model_cache.glob("*/modules.json"))
        try:
            self._model = SentenceTransformer(
                model_id, device=device, cache_folder=str(cache_dir),
                local_files_only=cached,
            )
        except (OSError, ValueError):
            if not cached:
                raise
            # A partial cache can exist after an interrupted download.
            self._model = SentenceTransformer(
                model_id, device=device, cache_folder=str(cache_dir),
                local_files_only=False,
            )
        if hasattr(self._model, "get_embedding_dimension"):
            actual_dimension = self._model.get_embedding_dimension()
        else:
            actual_dimension = self._model.get_sentence_embedding_dimension()
        if actual_dimension != dimension:
            raise ValueError(f"Model dimension {actual_dimension} does not match expected {dimension}")

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = np.asarray(self._model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=len(texts) > self.batch_size,
        ))
        if vectors.shape != (len(texts), self.dimension):
            raise ValueError(f"Unexpected embedding shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding contains non-finite values")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("Embedding is not unit-normalized")
        return vectors.tolist()


@lru_cache(maxsize=2)
def load_encoder(model_id: str, dimension: int, cache_dir: Path, batch_size: int) -> Encoder:
    """Reuse model weights across documents and queries in one process."""
    return Encoder(model_id, dimension, cache_dir, batch_size)
