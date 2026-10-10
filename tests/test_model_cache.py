from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from bibo.embeddings.model_cache import find_local_model, prepare_model


def _make_complete_snapshot(path: Path) -> None:
    path.mkdir(parents=True)
    for name in ("modules.json", "config.json", "model.safetensors", "tokenizer.json"):
        (path / name).write_bytes(b"test")


def test_complete_cache_reused_offline_without_download(tmp_path, monkeypatch):
    snapshot = tmp_path / "models--BAAI--bge-m3" / "snapshots" / "revision"
    _make_complete_snapshot(snapshot)
    assert find_local_model(tmp_path, "BAAI/bge-m3") == snapshot.resolve()
    # Offline mode returns before importing or calling the Hub client.
    path, downloaded = prepare_model(tmp_path, "BAAI/bge-m3", offline=True)
    assert path == snapshot.resolve()
    assert not downloaded


def test_partial_cache_never_used_silently(tmp_path):
    snapshot = tmp_path / "models--BAAI--bge-m3" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "modules.json").write_text("{}", encoding="utf-8")
    assert find_local_model(tmp_path, "BAAI/bge-m3") is None
    with pytest.raises(FileNotFoundError, match="complete local model"):
        prepare_model(tmp_path, "BAAI/bge-m3", offline=True)


def test_explicit_model_directory(tmp_path):
    model_dir = tmp_path / "custom-model"
    _make_complete_snapshot(model_dir)
    assert find_local_model(tmp_path / "cache", "BAAI/bge-m3", model_dir) == model_dir.resolve()


def test_prepare_downloads_once_then_reuses_cache(tmp_path, monkeypatch):
    snapshot = tmp_path / "models--BAAI--bge-m3" / "snapshots" / "revision"
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        _make_complete_snapshot(snapshot)
        return str(snapshot)

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        SimpleNamespace(snapshot_download=fake_download))
    path, downloaded = prepare_model(tmp_path, "BAAI/bge-m3")
    assert downloaded and path == snapshot.resolve()
    path, downloaded = prepare_model(tmp_path, "BAAI/bge-m3")
    assert not downloaded and len(calls) == 1
    assert "*.bin" in calls[0]["ignore_patterns"]
