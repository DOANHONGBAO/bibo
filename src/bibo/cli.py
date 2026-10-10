"""Command-line interface for local document retrieval."""

import argparse
import logging
from pathlib import Path
import sys

from bibo.config import Settings
from bibo.ingestion.parser import DocumentError
from bibo.ingestion.pipeline import ingest_document, supported_files
from bibo.retrieval.search import search_passages
from bibo.storage.vector_store import VectorStore


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bibo", description="Local research document retrieval")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Import one PDF, TXT or Markdown file")
    ingest.add_argument("path", type=Path)
    ingest_dir = sub.add_parser("ingest-dir", help="Import supported files recursively")
    ingest_dir.add_argument("path", type=Path)
    search = sub.add_parser("search", help="Find passages using Vietnamese or English")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    sub.add_parser("list", help="List indexed documents")
    sub.add_parser("stats", help="Show index statistics")
    remove = sub.add_parser("remove", help="Remove indexed chunks, leaving source file untouched")
    remove.add_argument("document_id")
    remove.add_argument("--yes", action="store_true", help="Confirm without an interactive prompt")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Redirected Windows consoles may default to a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and stream.encoding.lower() != "utf-8":
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        store = VectorStore(settings)
        if args.command == "ingest":
            result = ingest_document(args.path, settings=settings, store=store)
            status = "Already indexed" if result.duplicate else "Indexed"
            print(f"{status}: {result.filename} | ID: {result.document_id} | chunks: {result.chunk_count}")
        elif args.command == "ingest-dir":
            files = supported_files(args.path)
            if not files:
                print("No supported files found (.pdf, .txt, .md).")
                return 0
            indexed = skipped = failed = 0
            for path in files:
                try:
                    result = ingest_document(path, settings=settings, store=store)
                    skipped += int(result.duplicate)
                    indexed += int(not result.duplicate)
                    print(f"{'Skipped duplicate' if result.duplicate else 'Indexed'}: {path}")
                except (DocumentError, OSError, ValueError) as exc:
                    failed += 1
                    logger.error("Could not index %s: %s", path, exc)
            print(f"Finished: {indexed} indexed, {skipped} duplicates, {failed} failed")
            return 1 if failed else 0
        elif args.command == "search":
            hits = search_passages(args.query, top_k=args.top_k, settings=settings, store=store)
            if not hits:
                print("No passages found. Import documents first or try another query.")
            for rank, hit in enumerate(hits, 1):
                page = f" | Page: {hit.page_number}" if hit.page_number is not None else ""
                print(f"Rank {rank} | Document: {hit.filename}{page} | "
                      f"Cosine similarity: {hit.similarity:.4f}")
                print(f"Source: {hit.source_path} | Document ID: {hit.document_id}")
                print(f"Content:\n{hit.content}\n")
        elif args.command == "list":
            documents = store.list_documents()
            if not documents:
                print("No indexed documents.")
            for doc in documents:
                print(f"{doc.filename} | ID: {doc.document_id} | chunks: {doc.chunk_count}")
                print(f"  {doc.source_path}")
        elif args.command == "stats":
            print(f"Documents: {len(store.list_documents())}")
            print(f"Chunks: {store.count()}")
            print(f"Model: {settings.model_id} ({settings.embedding_dimension} dimensions)")
            print(f"Index: {settings.data_dir / 'chroma'}")
        elif args.command == "remove":
            count = store.document_chunk_count(args.document_id)
            if not count:
                print(f"Document not found: {args.document_id}")
                return 1
            if not args.yes:
                try:
                    response = input(f"Remove {count} indexed chunks for {args.document_id}? [y/N] ")
                except EOFError:
                    response = ""
                if response.strip().lower() not in {"y", "yes"}:
                    print("Cancelled.")
                    return 0
            removed = store.remove_document(args.document_id)
            print(f"Removed {removed} indexed chunks. Original file was not deleted.")
        return 0
    except (DocumentError, OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception:
        logger.exception("Unexpected failure")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
