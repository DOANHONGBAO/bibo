import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bibo.embeddings.encoder import Encoder


def test_encoder_shape_normalization_and_dimension(monkeypatch, tmp_path):
    class Model:
        max_seq_length = 1024

        def __init__(self, model_id, device, cache_folder, local_files_only,
                     trust_remote_code):
            assert model_id == str(tmp_path)
            assert device == "cpu"
            assert local_files_only is True
            assert trust_remote_code is False

        def get_embedding_dimension(self):
            return 3

        def _first_module(self):
            return SimpleNamespace(tokenizer=lambda texts, **kwargs: {
                "input_ids": [[1] * len(text.split()) for text in texts]
            })

        def encode(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            return np.tile(np.array([[1.0, 0.0, 0.0]]), (len(texts), 1))

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        SimpleNamespace(SentenceTransformer=Model))
    monkeypatch.setattr("bibo.embeddings.encoder.find_local_model",
                        lambda cache_dir, model_id, model_dir: tmp_path)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    encoder = Encoder("BAAI/bge-m3", 3, tmp_path, 2)
    assert encoder.encode(["Việt", "English"]) == [[1.0, 0.0, 0.0]] * 2
    assert encoder.last_metrics.tokens == 2
    encoder._model.max_seq_length = 1
    with pytest.raises(ValueError, match="above model limit"):
        encoder.encode(["two words"])
    with pytest.raises(ValueError, match="dimension"):
        Encoder("BAAI/bge-m3", 1024, tmp_path, 2)
