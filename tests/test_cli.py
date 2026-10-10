from bibo import cli
from bibo.ingestion.pipeline import ingest_document
from bibo.storage.vector_store import VectorStore


def test_cli_list_stats_and_confirmed_remove(settings, fake_encoder, tmp_path,
                                             monkeypatch, capsys):
    source = tmp_path / "note.txt"
    source.write_text("research result one two three four five six", encoding="utf-8")
    store = VectorStore(settings)
    result = ingest_document(source, settings=settings, store=store,
                             encoder=fake_encoder, progress=False)
    monkeypatch.setattr(cli.Settings, "from_env", lambda: settings)
    assert cli.main(["list"]) == 0
    assert result.document_id in capsys.readouterr().out
    assert cli.main(["stats"]) == 0
    assert "Documents: 1" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert cli.main(["remove", result.document_id]) == 0
    assert VectorStore(settings).count() == result.chunk_count
    assert cli.main(["remove", result.document_id, "--yes"]) == 0
    assert source.exists()
    assert VectorStore(settings).count() == 0


def test_cli_rejects_empty_query_and_missing_path(settings, monkeypatch, capsys):
    monkeypatch.setattr(cli.Settings, "from_env", lambda: settings)
    assert cli.main(["search", " "]) == 1
    assert cli.main(["ingest", "missing.txt"]) == 1
