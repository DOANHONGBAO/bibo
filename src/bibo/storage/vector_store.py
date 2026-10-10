"""ChromaDB adapter; other modules use only this interface."""

from dataclasses import dataclass
from typing import Sequence

from bibo.config import Settings
from bibo.ingestion.chunker import Chunk


COLLECTION_NAME = "research_passages"


@dataclass(frozen=True)
class DocumentSummary:
    document_id: str
    filename: str
    source_path: str
    chunk_count: int
    status: str = "complete"


@dataclass(frozen=True)
class DocumentState:
    ids: frozenset[str]
    expected_count: int | None
    complete: bool
    consistent: bool

    @property
    def count(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    content: str
    filename: str
    source_path: str
    page_number: int | None
    chunk_index: int
    similarity: float


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        import chromadb

        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.data_dir / "chroma"),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        expected = {
            "hnsw:space": "cosine",
            "model_id": settings.model_id,
            "dimension": settings.embedding_dimension,
            "chunk_size_words": settings.chunk_size_words,
            "chunk_overlap_words": settings.chunk_overlap_words,
        }
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME, metadata=expected,
        )
        actual = self._collection.metadata or {}
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Existing index configuration is incompatible with current model or chunk settings; "
                "use a different BIBO_DATA_DIR or restore the original settings"
            )

    def count(self) -> int:
        return self._collection.count()

    def document_chunk_count(self, document_id: str) -> int:
        return len(self._collection.get(where={"document_id": document_id}, include=[])["ids"])

    def document_ids_for_path(self, source_path: str) -> set[str]:
        rows = self._collection.get(where={"source_path": source_path},
                                    include=["metadatas"])
        return {item["document_id"] for item in rows["metadatas"]}

    def document_state(self, document_id: str) -> DocumentState:
        rows = self._collection.get(where={"document_id": document_id}, include=["metadatas"])
        ids = frozenset(rows["ids"])
        if not ids:
            return DocumentState(ids, None, False, True)
        metadata = rows["metadatas"]
        expected_values = {item.get("chunk_count") for item in metadata}
        expected = next(iter(expected_values)) if len(expected_values) == 1 else None
        indices = {item.get("chunk_index") for item in metadata}
        consistent = (isinstance(expected, int) and expected > 0
                      and all(isinstance(index, int) and 0 <= index < expected
                              for index in indices)
                      and all(item.get("file_hash") == document_id
                              and item.get("document_id") == document_id
                              for item in metadata)
                      and len(indices) == len(ids))
        complete = consistent and len(ids) == expected and indices == set(range(expected))
        return DocumentState(ids, expected, complete, consistent)

    def document_complete(self, document_id: str) -> bool:
        return self.document_state(document_id).complete

    def add_document(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *,
        batch_size: int, progress: bool = True, total_chunks: int | None = None,
    ) -> None:
        if len(chunks) != len(vectors) or not chunks:
            raise ValueError("Chunks and embeddings must have the same nonzero length")
        if any(len(vector) != self._collection.metadata["dimension"] for vector in vectors):
            raise ValueError("Embedding dimension does not match the collection")
        document_id = chunks[0].document_id
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError("A batch may contain only one document")
        total_chunks = total_chunks or len(chunks)
        if total_chunks < len(chunks):
            raise ValueError("Total chunk count cannot be smaller than inserted batch")
        existing_ids = set(self._collection.get(
            ids=[chunk.chunk_id for chunk in chunks], include=[],
        )["ids"])
        if existing_ids:
            raise ValueError("Some requested chunks are already indexed; refresh document state")
        from tqdm import tqdm

        starts = range(0, len(chunks), batch_size)
        inserted_ids: list[str] = []
        try:
            for start in tqdm(starts, total=(len(chunks) + batch_size - 1) // batch_size,
                              desc="Indexing", disable=not progress):
                batch = chunks[start:start + batch_size]
                metadata = []
                for chunk in batch:
                    item = {
                        "document_id": chunk.document_id,
                        "filename": chunk.filename,
                        "source_path": chunk.source_path,
                        "chunk_index": chunk.chunk_index,
                        "file_hash": chunk.file_hash,
                        "chunk_count": total_chunks,
                    }
                    if chunk.page_number is not None:
                        item["page_number"] = chunk.page_number
                    metadata.append(item)
                self._collection.add(
                    ids=[chunk.chunk_id for chunk in batch],
                    documents=[chunk.content for chunk in batch],
                    embeddings=[list(vector) for vector in vectors[start:start + batch_size]],
                    metadatas=metadata,
                )
                inserted_ids.extend(chunk.chunk_id for chunk in batch)
        except Exception:
            # Retain pre-existing chunks; only roll back confirmed batches from this call.
            if inserted_ids:
                self._collection.delete(ids=inserted_ids)
            raise

    def remove_document(self, document_id: str) -> int:
        count = self.document_chunk_count(document_id)
        if count:
            self._collection.delete(where={"document_id": document_id})
        return count

    def list_documents(self) -> list[DocumentSummary]:
        rows = self._collection.get(include=["metadatas"])
        grouped: dict[str, DocumentSummary] = {}
        counts: dict[str, int] = {}
        expected_counts: dict[str, set[int]] = {}
        indices: dict[str, set[int]] = {}
        hashes_valid: dict[str, bool] = {}
        for metadata in rows["metadatas"]:
            document_id = metadata["document_id"]
            counts[document_id] = counts.get(document_id, 0) + 1
            expected_counts.setdefault(document_id, set()).add(metadata["chunk_count"])
            indices.setdefault(document_id, set()).add(metadata["chunk_index"])
            hashes_valid[document_id] = (hashes_valid.get(document_id, True)
                                         and metadata.get("file_hash") == document_id)
            grouped[document_id] = DocumentSummary(document_id, metadata["filename"],
                                                    metadata["source_path"], 0)
        return sorted(
            [DocumentSummary(doc.document_id, doc.filename, doc.source_path,
                             counts[doc.document_id],
                             "complete" if expected_counts[doc.document_id]
                             == {counts[doc.document_id]}
                             and hashes_valid[doc.document_id]
                             and indices[doc.document_id]
                             == set(range(counts[doc.document_id])) else "partial")
             for doc in grouped.values()],
            key=lambda doc: (doc.filename.lower(), doc.document_id),
        )

    def search(self, vector: Sequence[float], top_k: int) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if len(vector) != self._collection.metadata["dimension"]:
            raise ValueError("Query embedding dimension does not match the index")
        count = self.count()
        if not count:
            return []
        rows = self._collection.query(
            query_embeddings=[list(vector)], n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, content, metadata, distance in zip(
            rows["ids"][0], rows["documents"][0], rows["metadatas"][0],
            rows["distances"][0], strict=True,
        ):
            hits.append(SearchHit(
                chunk_id=chunk_id, document_id=metadata["document_id"],
                content=content, filename=metadata["filename"],
                source_path=metadata["source_path"],
                page_number=metadata.get("page_number"),
                chunk_index=metadata["chunk_index"],
                similarity=1.0 - float(distance),
            ))
        return hits
