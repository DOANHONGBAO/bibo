"""Explicit local BGE-M3 cache and one-time model preparation."""

from pathlib import Path


def _complete_model(directory: Path) -> bool:
    """Check the core files before attempting an offline model load."""
    return (
        (directory / "modules.json").is_file()
        and (directory / "config.json").is_file()
        and any((directory / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
        and any((directory / name).is_file() for name in ("tokenizer.json", "sentencepiece.bpe.model"))
    )


def find_local_model(cache_dir: Path, model_id: str, model_dir: Path | None = None) -> Path | None:
    """Return a complete local path, without any network request."""
    if model_dir is not None:
        return model_dir.resolve() if _complete_model(model_dir) else None
    snapshots = cache_dir / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted(snapshots.iterdir(), key=lambda path: path.stat().st_mtime,
                        reverse=True)
    return next((path.resolve() for path in candidates if path.is_dir() and _complete_model(path)), None)


def prepare_model(
    cache_dir: Path, model_id: str, *, model_dir: Path | None = None,
    offline: bool = False,
) -> tuple[Path, bool]:
    """Reuse a complete cache or explicitly download a public model once."""
    existing = find_local_model(cache_dir, model_id, model_dir)
    if existing is not None:
        return existing, False
    if offline:
        raise FileNotFoundError("No complete local model; run 'bibo prepare-model' with network access")
    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "repo_id": model_id, "cache_dir": str(cache_dir),
        # BGE-M3 publishes both formats; one weight format is enough.
        "ignore_patterns": ["*.bin", "*.onnx", "*.h5", "*.msgpack", "*.ot"],
    }
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(model_dir)
    downloaded = Path(snapshot_download(**kwargs))
    result = find_local_model(cache_dir, model_id, model_dir)
    if result is None:
        raise RuntimeError(f"Model download did not produce a complete model at {downloaded}")
    return result, True
