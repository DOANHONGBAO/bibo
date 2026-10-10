"""Small deterministic embeddings for tests; no model download."""

import hashlib
import math

import pytest

from bibo.config import Settings


class FakeEncoder:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.device = "cpu"

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = [float(digest[index % len(digest)] - 127) for index in range(self.dimension)]
            norm = math.sqrt(sum(value * value for value in values))
            vectors.append([value / norm for value in values])
        return vectors


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", model_cache=tmp_path / "cache",
                    chunk_size_words=8, chunk_overlap_words=2, batch_size=2,
                    embedding_dimension=8)


@pytest.fixture
def fake_encoder(settings):
    return FakeEncoder(settings.embedding_dimension)
