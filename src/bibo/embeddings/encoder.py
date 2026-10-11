"""Offline BGE-M3 dense encoder, reused within one Python process."""

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
import statistics
import time
from typing import Sequence

import numpy as np

from bibo.embeddings.model_cache import find_local_model


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncodeMetrics:
    chunks: int
    tokens: int
    mean_tokens: float
    p95_tokens: int
    tokenization_seconds: float
    inference_seconds: float

    @property
    def chunks_per_second(self) -> float:
        return self.chunks / self.inference_seconds if self.inference_seconds else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.inference_seconds if self.inference_seconds else 0.0


class Encoder:
    def __init__(
        self, model_id: str, dimension: int, cache_dir: Path, batch_size: int,
        model_dir: Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.dimension = dimension
        self.batch_size = batch_size
        self.last_metrics: EncodeMetrics | None = None
        model_path = find_local_model(cache_dir, model_id, model_dir)
        if model_path is None:
            raise FileNotFoundError(
                "BGE-M3 is not cached completely. Run 'python -m bibo.cli prepare-model' "
                "once with network access."
            )
        self.model_path = model_path
        started = time.perf_counter()
        import torch
        from sentence_transformers import SentenceTransformer

        self._torch = torch
        self.torch_version = torch.__version__
        self.cuda_available = torch.cuda.is_available()
        self.cuda_device_name = torch.cuda.get_device_name(0) if self.cuda_available else None
        self.device = "cuda" if self.cuda_available else "cpu"
        self._model = SentenceTransformer(
            str(model_path), device=self.device, cache_folder=str(cache_dir),
            local_files_only=True, trust_remote_code=False,
        )
        actual_dimension = (self._model.get_embedding_dimension()
                            if hasattr(self._model, "get_embedding_dimension")
                            else self._model.get_sentence_embedding_dimension())
        if actual_dimension != dimension:
            raise ValueError(f"Model dimension {actual_dimension} does not match expected {dimension}")
        self.model_load_seconds = time.perf_counter() - started
        logger.info(
            "PyTorch %s; CUDA=%s%s; device=%s; model=%s; load=%.3fs; batch=%d",
            self.torch_version, self.cuda_available,
            f" ({self.cuda_device_name})" if self.cuda_device_name else "",
            self.device, self.model_path, self.model_load_seconds, self.batch_size,
        )

    def token_lengths(self, texts: Sequence[str]) -> tuple[list[int], float]:
        """Count full input tokens so the model never silently truncates text."""
        started = time.perf_counter()
        tokenizer = self._model._first_module().tokenizer
        tokenized = tokenizer(list(texts), add_special_tokens=True, truncation=False)
        lengths = [len(ids) for ids in tokenized["input_ids"]]
        return lengths, time.perf_counter() - started

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        lengths, tokenize_seconds = self.token_lengths(texts)
        maximum = self._model.max_seq_length
        if maximum is not None and max(lengths) > maximum:
            raise ValueError(
                f"Input has {max(lengths)} tokens, above model limit {maximum}; "
                "reduce BIBO_CHUNK_SIZE_WORDS before indexing into a new collection"
            )
        started = time.perf_counter()
        with self._torch.inference_mode():
            vectors = np.asarray(self._model.encode(
                list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=len(texts) > self.batch_size,
            ))
        inference_seconds = time.perf_counter() - started
        if vectors.shape != (len(texts), self.dimension):
            raise ValueError(f"Unexpected embedding shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding contains non-finite values")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("Embedding is not unit-normalized")
        ordered = sorted(lengths)
        self.last_metrics = EncodeMetrics(
            chunks=len(texts), tokens=sum(lengths), mean_tokens=statistics.mean(lengths),
            p95_tokens=ordered[int(0.95 * (len(ordered) - 1))],
            tokenization_seconds=tokenize_seconds, inference_seconds=inference_seconds,
        )
        logger.info(
            "Embedded %d chunks (%d tokens, mean %.1f, p95 %d) in %.3fs "
            "(%.2f chunks/s, %.1f tokens/s); batch=%d; device=%s",
            self.last_metrics.chunks, self.last_metrics.tokens,
            self.last_metrics.mean_tokens, self.last_metrics.p95_tokens,
            self.last_metrics.inference_seconds, self.last_metrics.chunks_per_second,
            self.last_metrics.tokens_per_second, self.batch_size, self.device,
        )
        return vectors.tolist()


@lru_cache(maxsize=2)
def load_encoder(
    model_id: str, dimension: int, cache_dir: Path, batch_size: int,
    model_dir: Path | None = None,
) -> Encoder:
    """Reuse model weights only inside the current process."""
    return Encoder(model_id, dimension, cache_dir, batch_size, model_dir)
