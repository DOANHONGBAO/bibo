"""Configuration for the local retrieval pipeline."""

from dataclasses import dataclass
import os
from pathlib import Path


MODEL_ID = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model_cache: Path
    chunk_size_words: int = 500
    chunk_overlap_words: int = 75
    batch_size: int = 4
    model_id: str = MODEL_ID
    embedding_dimension: int = EMBEDDING_DIMENSION
    model_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.chunk_size_words < 1:
            raise ValueError("Chunk size must be positive")
        if not 0 <= self.chunk_overlap_words < self.chunk_size_words:
            raise ValueError("Chunk overlap must be nonnegative and smaller than chunk size")
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")
        if self.embedding_dimension < 1:
            raise ValueError("Embedding dimension must be positive")

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(__file__).resolve().parents[2]
        data_dir = Path(os.getenv("BIBO_DATA_DIR", str(project_root / "data"))).expanduser()
        cache = Path(os.getenv("BIBO_MODEL_CACHE", str(data_dir / "model_cache"))).expanduser()
        return cls(
            data_dir=data_dir,
            model_cache=cache,
            chunk_size_words=int(os.getenv("BIBO_CHUNK_SIZE_WORDS", "500")),
            chunk_overlap_words=int(os.getenv("BIBO_CHUNK_OVERLAP_WORDS", "75")),
            batch_size=int(os.getenv("BIBO_BATCH_SIZE", "4")),
            model_dir=Path(os.environ["BIBO_MODEL_DIR"]).expanduser()
            if os.getenv("BIBO_MODEL_DIR") else None,
        )
