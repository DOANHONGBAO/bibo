import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bibo.embeddings.encoder import Encoder


def test_encoder_shape_normalization_and_dimension(monkeypatch, tmp_path):
    class Model:
        def __init__(self, model_id, device, cache_folder, local_files_only):
            assert model_id == "BAAI/bge-m3"
            assert device == "cpu"

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            return np.tile(np.array([[1.0, 0.0, 0.0]]), (len(texts), 1))

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        SimpleNamespace(SentenceTransformer=Model))
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    encoder = Encoder("BAAI/bge-m3", 3, tmp_path, 2)
    assert encoder.encode(["Việt", "English"]) == [[1.0, 0.0, 0.0]] * 2
    with pytest.raises(ValueError, match="dimension"):
        Encoder("BAAI/bge-m3", 1024, tmp_path, 2)
