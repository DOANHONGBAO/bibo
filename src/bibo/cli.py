"""Command-line interface for local document retrieval."""

import argparse
import logging
from pathlib import Path
import sys
import time

from bibo.config import Settings
from bibo.embeddings.encoder import load_encoder
from bibo.embeddings.model_cache import prepare_model
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
    ingest.add_argument("--profile", action="store_true", help="Show stage timings")
    ingest_dir = sub.add_parser("ingest-dir", help="Import supported files recursively")
    ingest_dir.add_argument("path", type=Path)
    ingest_dir.add_argument("--profile", action="store_true", help="Show stage timings")
    search = sub.add_parser("search", help="Find passages using Vietnamese or English")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--profile", action="store_true", help="Show stage timings")
    interactive = sub.add_parser("interactive", help="Search repeatedly with one loaded model")
    interactive.add_argument("--top-k", type=int, default=5)
    interactive.add_argument("--profile", action="store_true", help="Show query timings")
    prepare = sub.add_parser("prepare-model", help="Download or verify BGE-M3 once")
    prepare.add_argument("--offline", action="store_true", help="Only use existing local files")
    benchmark = sub.add_parser("benchmark", help="Profile fixed document and query workload")
    benchmark.add_argument("--document", type=Path, help="PDF to benchmark; default is a synthetic fixture")
    sub.add_parser("list", help="List indexed documents")
    sub.add_parser("stats", help="Show index statistics")
    remove = sub.add_parser("remove", help="Remove indexed chunks, leaving source file untouched")
    remove.add_argument("document_id")
    remove.add_argument("--yes", action="store_true", help="Confirm without an interactive prompt")
    return parser


def _print_hits(hits) -> None:
    if not hits:
        print("No passages found. Import documents first or try another query.")
    for rank, hit in enumerate(hits, 1):
        page = f" | Page: {hit.page_number}" if hit.page_number is not None else ""
        print(f"Rank {rank} | Document: {hit.filename}{page} | "
              f"Cosine similarity: {hit.similarity:.4f}")
        print(f"Source: {hit.source_path} | Document ID: {hit.document_id}")
        print(f"Content:\n{hit.content}\n")


def _print_timings(timings: dict[str, float]) -> None:
    print("Timing (seconds): " + ", ".join(
        f"{name}={duration:.3f}" for name, duration in timings.items()
    ))


def main(argv: list[str] | None = None) -> int:
    cli_start = time.perf_counter()
    # Redirected Windows consoles may default to a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower() != "utf-8":
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    if getattr(args, "profile", False) or args.command in {"prepare-model", "benchmark"}:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        settings = Settings.from_env()
        if args.command == "prepare-model":
            model_path, downloaded = prepare_model(
                settings.model_cache, settings.model_id,
                model_dir=settings.model_dir, offline=args.offline,
            )
            encoder = load_encoder(settings.model_id, settings.embedding_dimension,
                                   settings.model_cache, settings.batch_size,
                                   settings.model_dir)
            print(f"Model: {model_path} ({'downloaded' if downloaded else 'cached'})")
            print(f"PyTorch: {encoder.torch_version} | CUDA: {encoder.cuda_available}"
                  f"{f' ({encoder.cuda_device_name})' if encoder.cuda_device_name else ''}")
            print(f"Device: {encoder.device} | Load: {encoder.model_load_seconds:.3f}s | "
                  f"Batch size: {encoder.batch_size}")
            return 0
        if args.command == "benchmark":
            from bibo.benchmark import run_benchmark

            report_path, summary = run_benchmark(settings, args.document)
            print(summary)
            print(f"JSON report: {report_path}")
            return 0
        started = time.perf_counter()
        store = VectorStore(settings)
        store_init_seconds = time.perf_counter() - started
        if args.command == "ingest":
            result = ingest_document(args.path, settings=settings, store=store)
            status = ("Skipped unchanged" if result.duplicate else
                      "Repaired partial index" if result.repaired else
                      "Indexed changed version (old kept)" if result.changed else "Indexed")
            print(f"{status}: {result.filename} | ID: {result.document_id} | chunks: {result.chunk_count}")
            if args.profile:
                _print_timings({"store_initialization": store_init_seconds,
                                **result.timings, "cli_total": time.perf_counter() - cli_start})
                if result.token_count:
                    print(f"Tokens: {result.token_count} | Mean: {result.mean_tokens:.1f} | "
                          f"p95: {result.p95_tokens}")
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
                    status = ("Skipped unchanged" if result.duplicate else
                              "Repaired partial index" if result.repaired else
                              "Indexed changed version (old kept)" if result.changed
                              else "Indexed")
                    print(f"{status}: {path}")
                    if args.profile:
                        _print_timings(result.timings)
                except (DocumentError, OSError, ValueError) as exc:
                    failed += 1
                    logger.error("Could not index %s: %s", path, exc)
            print(f"Finished: {indexed} indexed, {skipped} duplicates, {failed} failed")
            if args.profile:
                _print_timings({"store_initialization": store_init_seconds,
                                "cli_total": time.perf_counter() - cli_start})
            return 1 if failed else 0
        elif args.command == "search":
            timings: dict[str, float] = {}
            hits = search_passages(args.query, top_k=args.top_k, settings=settings,
                                   store=store, timings=timings)
            _print_hits(hits)
            if args.profile:
                _print_timings({"store_initialization": store_init_seconds,
                                **timings, "cli_total": time.perf_counter() - cli_start})
        elif args.command == "interactive":
            if args.top_k < 1:
                raise ValueError("top_k must be positive")
            if not store.count():
                print("No indexed documents. Import documents first.")
                return 0
            encoder = load_encoder(settings.model_id, settings.embedding_dimension,
                                   settings.model_cache, settings.batch_size,
                                   settings.model_dir)
            print(f"Interactive search ready ({encoder.device}). Type exit or quit to stop.")
            if args.profile:
                _print_timings({"store_initialization": store_init_seconds,
                                "model_initialization": getattr(encoder, "model_load_seconds", 0.0),
                                "interactive_ready": time.perf_counter() - cli_start})
            while True:
                try:
                    query = input("bibo> ").strip()
                except EOFError:
                    print()
                    break
                if query.lower() in {"exit", "quit"}:
                    break
                if not query:
                    continue
                timings = {}
                hits = search_passages(query, top_k=args.top_k, settings=settings,
                                       store=store, encoder=encoder, timings=timings)
                _print_hits(hits)
                if args.profile:
                    _print_timings(timings)
        elif args.command == "list":
            documents = store.list_documents()
            if not documents:
                print("No indexed documents.")
            for doc in documents:
                print(f"{doc.filename} | ID: {doc.document_id} | "
                      f"chunks: {doc.chunk_count} | status: {doc.status}")
                print(f"  {doc.source_path}")
        elif args.command == "stats":
            documents = store.list_documents()
            print(f"Documents: {len(documents)} (partial: "
                  f"{sum(doc.status == 'partial' for doc in documents)})")
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
