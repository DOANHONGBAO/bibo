from pathlib import Path

import fitz
import pytest

from bibo.ingestion.pipeline import ingest_document, supported_files
from bibo.retrieval.search import search_passages
from bibo.storage.vector_store import VectorStore


def test_persistent_ingestion_duplicate_search_and_metadata(settings, fake_encoder, tmp_path):
    source = tmp_path / "paper.txt"
    source.write_text("alpha beta gamma delta epsilon zeta eta theta iota kappa", encoding="utf-8")
    store = VectorStore(settings)
    first = ingest_document(source, settings=settings, store=store,
                            encoder=fake_encoder, progress=False)
    assert first.chunk_count > 1
    assert not first.duplicate
    second = ingest_document(source, settings=settings, store=store,
                             encoder=fake_encoder, progress=False)
    assert second.duplicate
    assert store.count() == first.chunk_count
    # Reopen the Chroma collection to demonstrate persistence after restart.
    reopened = VectorStore(settings)
    assert reopened.count() == first.chunk_count
    assert reopened.list_documents()[0].filename == "paper.txt"
    assert reopened.list_documents()[0].source_path == str(source.resolve())
    metadata = reopened._collection.get(where={"document_id": first.document_id},
                                        include=["metadatas"])["metadatas"]
    assert all(item["file_hash"] == first.document_id for item in metadata)
    assert all(item["chunk_count"] == first.chunk_count for item in metadata)
    hits = search_passages("alpha beta", top_k=3, settings=settings,
                           store=reopened, encoder=fake_encoder)
    assert hits and all(hit.document_id == first.document_id for hit in hits)
    assert all(hit.page_number is None for hit in hits)
    assert all(hit.content in source.read_text(encoding="utf-8") for hit in hits)
    assert hits == sorted(hits, key=lambda hit: hit.similarity, reverse=True)
    assert reopened.remove_document(first.document_id) == first.chunk_count
    assert source.exists()
    assert reopened.count() == 0


def test_empty_index_and_validation(settings, fake_encoder):
    store = VectorStore(settings)
    assert search_passages("Vietnamese query", top_k=5, settings=settings,
                           store=store, encoder=fake_encoder) == []
    with pytest.raises(ValueError, match="empty"):
        search_passages(" ", top_k=5, settings=settings, store=store, encoder=fake_encoder)
    with pytest.raises(ValueError, match="positive"):
        search_passages("query", top_k=0, settings=settings, store=store, encoder=fake_encoder)


def test_index_configuration_guard(settings):
    VectorStore(settings)
    changed = type(settings)(data_dir=settings.data_dir, model_cache=settings.model_cache,
                             chunk_size_words=settings.chunk_size_words,
                             chunk_overlap_words=settings.chunk_overlap_words,
                             batch_size=settings.batch_size, model_id="different-model",
                             embedding_dimension=settings.embedding_dimension)
    with pytest.raises(ValueError, match="incompatible"):
        VectorStore(changed)


def test_directory_skips_unsupported(tmp_path):
    (tmp_path / "a.txt").write_text("content", encoding="utf-8")
    (tmp_path / "b.csv").write_text("ignored", encoding="utf-8")
    assert [path.name for path in supported_files(tmp_path)] == ["a.txt"]


def test_synthetic_bilingual_pipeline_persists(tmp_path, settings):
    class BilingualStub:
        def encode(self, texts):
            return [[1.0] + [0.0] * 7 if ("false positive" in text.lower()
                    or "dương tính giả" in text.lower()) else [0.0, 1.0] + [0.0] * 6
                    for text in texts]

    sample = tmp_path / "sample.md"
    sample.write_bytes((Path(__file__).parent / "fixtures" / "sample.md").read_bytes())
    store = VectorStore(settings)
    result = ingest_document(sample, settings=settings, store=store,
                             encoder=BilingualStub(), progress=False)
    assert result.chunk_count > 0
    reopened = VectorStore(settings)
    for query in ("How to reduce false positives?", "Làm sao giảm dương tính giả?"):
        hits = search_passages(query, top_k=2, settings=settings,
                               store=reopened, encoder=BilingualStub())
        assert hits and hits[0].filename == "sample.md"
        assert hits[0].content in sample.read_text(encoding="utf-8")


def test_pdf_ingestion_keeps_page_metadata(tmp_path, settings, fake_encoder):
    path = tmp_path / "paper.pdf"
    pdf = fitz.open()
    for text in ("Methods for reducing false positives", "Results from a second page"):
        page = pdf.new_page()
        page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()
    store = VectorStore(settings)
    result = ingest_document(path, settings=settings, store=store,
                             encoder=fake_encoder, progress=False)
    assert result.chunk_count == 2
    rows = store._collection.get(where={"document_id": result.document_id},
                                 include=["metadatas"])
    assert {metadata["page_number"] for metadata in rows["metadatas"]} == {1, 2}
    assert all(metadata["filename"] == "paper.pdf" for metadata in rows["metadatas"])


def test_failed_batch_rolls_back_for_retry(tmp_path, settings, fake_encoder,
                                           monkeypatch):
    path = tmp_path / "long.txt"
    path.write_text(" ".join(f"word{i}" for i in range(30)), encoding="utf-8")
    store = VectorStore(settings)
    original_add = store._collection.add
    calls = 0

    def failing_add(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated write failure")
        return original_add(*args, **kwargs)

    monkeypatch.setattr(store._collection, "add", failing_add)
    with pytest.raises(RuntimeError, match="simulated"):
        ingest_document(path, settings=settings, store=store,
                        encoder=fake_encoder, progress=False)
    assert store.count() == 0
    monkeypatch.setattr(store._collection, "add", original_add)
    result = ingest_document(path, settings=settings, store=store,
                             encoder=fake_encoder, progress=False)
    assert store.count() == result.chunk_count
