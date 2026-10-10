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

    def document_complete(self, document_id: str) -> bool:
        rows = self._collection.get(where={"document_id": document_id}, include=["metadatas"])
        if not rows["ids"]:
            return False
        expected = rows["metadatas"][0]["chunk_count"]
        return len(rows["ids"]) == expected

    def add_document(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *,
        batch_size: int, progress: bool = True,
    ) -> None:
        if len(chunks) != len(vectors) or not chunks:
            raise ValueError("Chunks and embeddings must have the same nonzero length")
        if any(len(vector) != self._collection.metadata["dimension"] for vector in vectors):
            raise ValueError("Embedding dimension does not match the collection")
        document_id = chunks[0].document_id
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError("A batch may contain only one document")
        if self.document_chunk_count(document_id):
            raise ValueError("Document already has indexed chunks")
        from tqdm import tqdm

        starts = range(0, len(chunks), batch_size)
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
                        "chunk_count": len(chunks),
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
        except Exception:
            # Batched writes are not atomic in Chroma; remove any partial document.
            self._collection.delete(where={"document_id": document_id})
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
        for metadata in rows["metadatas"]:
            document_id = metadata["document_id"]
            counts[document_id] = counts.get(document_id, 0) + 1
            grouped[document_id] = DocumentSummary(document_id, metadata["filename"],
                                                    metadata["source_path"], 0)
        return sorted(
            [DocumentSummary(doc.document_id, doc.filename, doc.source_path,
                             counts[doc.document_id]) for doc in grouped.values()],
            key=lambda doc: (doc.filename.lower(), doc.document_id),
        )

    def search(self, vector: Sequence[float], top_k: int) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if len(vector) != self._collection.metadata["dimension"]:
            raise ValueError("Query embedding dimension does not match the index")
        if not self.count():
            return []
        rows = self._collection.query(
            query_embeddings=[list(vector)], n_results=min(top_k, self.count()),
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
