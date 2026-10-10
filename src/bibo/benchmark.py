"""Local performance benchmark with no source text in its report."""

from dataclasses import replace
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from uuid import uuid4

from bibo.config import Settings
from bibo.embeddings.encoder import Encoder
from bibo.ingestion.chunker import chunk_pages
from bibo.ingestion.parser import parse_document, parse_document_cached
from bibo.ingestion.pipeline import file_sha256
from bibo.storage.vector_store import VectorStore


QUERY = "What methods are evaluated in this paper?"


def _measure(operation):
    started = time.perf_counter()
    value = operation()
    return value, time.perf_counter() - started


def _rss_mib() -> float | None:
    if sys.platform != "win32":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD),
                    ("peak_working_set", ctypes.c_size_t),
                    ("working_set", ctypes.c_size_t),
                    ("quota_peak_paged_pool", ctypes.c_size_t),
                    ("quota_paged_pool", ctypes.c_size_t),
                    ("quota_peak_nonpaged_pool", ctypes.c_size_t),
                    ("quota_nonpaged_pool", ctypes.c_size_t),
                    ("pagefile_usage", ctypes.c_size_t),
                    ("peak_pagefile_usage", ctypes.c_size_t)]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),
                                     ctypes.byref(counters), counters.cb):
        return None
    return counters.working_set / 2**20


def _synthetic_pdf(path: Path) -> Path:
    if path.is_file():
        return path
    import pymupdf

    path.parent.mkdir(parents=True, exist_ok=True)
    sentence = ("This synthetic research page compares retrieval methods, decision thresholds, "
                "false positive rates, and evaluation evidence across experiments. ")
    pdf = pymupdf.open()
    for page_number in range(16):
        page = pdf.new_page(width=595, height=842)
        text = f"Synthetic page {page_number + 1}.\n\n" + sentence * 22
        remaining = page.insert_textbox(pymupdf.Rect(40, 40, 555, 800), text,
                                        fontsize=8, lineheight=1.1)
        if remaining < 0:
            raise RuntimeError("Synthetic benchmark page did not fit")
    pdf.save(path)
    pdf.close()
    return path


def _subprocess_seconds(settings: Settings, command: list[str]) -> float:
    env = os.environ.copy()
    env["BIBO_DATA_DIR"] = str(settings.data_dir)
    env["BIBO_MODEL_CACHE"] = str(settings.model_cache)
    env["BIBO_CHUNK_SIZE_WORDS"] = str(settings.chunk_size_words)
    env["BIBO_CHUNK_OVERLAP_WORDS"] = str(settings.chunk_overlap_words)
    env["BIBO_BATCH_SIZE"] = str(settings.batch_size)
    if settings.model_dir is not None:
        env["BIBO_MODEL_DIR"] = str(settings.model_dir)
    env["HF_HUB_OFFLINE"] = "1"
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "bibo.cli", *command],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.decode(errors="replace")[-800:]
        raise RuntimeError(f"Benchmark subprocess {command[0]} exited "
                           f"{result.returncode}: {details}")
    return time.perf_counter() - started


def run_benchmark(settings: Settings, document: Path | None) -> tuple[Path, str]:
    """Compare CPU batches on fixed text, then measure local search stages."""
    run_started = time.perf_counter()
    project_root = Path(__file__).resolve().parents[2]
    run_dir = settings.data_dir / "benchmark_runs" / uuid4().hex
    source = (document.expanduser().resolve() if document is not None else
              _synthetic_pdf(settings.data_dir / "benchmark_fixture.pdf"))
    if not source.is_file() or source.suffix.lower() != ".pdf":
        raise ValueError("Benchmark document must be an existing PDF")
    file_hash, hash_s = _measure(lambda: file_sha256(source))
    pages, parse_s = _measure(lambda: parse_document(source))
    _, cache_write_s = _measure(lambda: parse_document_cached(
        source, file_hash=file_hash, cache_dir=run_dir / "parsed_cache"))
    (_, cache_hit), cache_read_s = _measure(lambda: parse_document_cached(
        source, file_hash=file_hash, cache_dir=run_dir / "parsed_cache"))
    chunks, chunk_s = _measure(lambda: chunk_pages(
        pages, document_id=file_hash, filename=source.name,
        source_path=str(source), file_hash=file_hash,
        size_words=settings.chunk_size_words,
        overlap_words=settings.chunk_overlap_words,
    ))
    if not chunks:
        raise ValueError("Benchmark PDF has no indexable text")
    texts = [chunk.content for chunk in chunks]
    encoder, model_load_s = _measure(lambda: Encoder(
        settings.model_id, settings.embedding_dimension, settings.model_cache,
        settings.batch_size, settings.model_dir,
    ))
    lengths, tokenization_s = encoder.token_lengths(texts)
    original_limit = encoder._model.max_seq_length
    if original_limit is not None and max(lengths) > original_limit:
        raise ValueError("Benchmark text exceeds the model sequence limit")
    encoder.encode(["warm-up query"])
    batch_results = []
    vectors_by_batch = {}
    for batch_size in (4, 8, 16):
        encoder.batch_size = batch_size
        before_memory = _rss_mib()
        vectors, elapsed = _measure(lambda: encoder.encode(texts))
        vectors_by_batch[batch_size] = vectors
        batch_results.append({
            "batch_size": batch_size, "seconds": elapsed,
            "chunks_per_second": len(chunks) / elapsed,
            "tokens_per_second": sum(lengths) / elapsed,
            "rss_before_mib": before_memory, "rss_after_mib": _rss_mib(),
        })
    fastest = min(batch_results, key=lambda item: item["seconds"])
    safe_length = next((limit for limit in (1024, 2048, 4096)
                        if max(lengths) <= limit and (original_limit is None or limit < original_limit)),
                       None)
    sequence_result = None
    if safe_length is not None:
        encoder.batch_size = fastest["batch_size"]
        encoder._model.max_seq_length = safe_length
        _, elapsed = _measure(lambda: encoder.encode(texts))
        sequence_result = {"max_sequence_length": safe_length, "seconds": elapsed,
                           "batch_size": fastest["batch_size"], "no_truncation": True}
        encoder._model.max_seq_length = original_limit

    temp_settings = replace(settings, data_dir=run_dir)
    store, store_init_s = _measure(lambda: VectorStore(temp_settings))
    vectors = vectors_by_batch.get(settings.batch_size)
    if vectors is None:
        encoder.batch_size = settings.batch_size
        vectors, _ = _measure(lambda: encoder.encode(texts))
    _, insertion_s = _measure(lambda: store.add_document(
        chunks, vectors, batch_size=settings.batch_size, progress=False,
    ))
    encoder.batch_size = settings.batch_size
    query_vector, query_embed_s = _measure(lambda: encoder.encode([QUERY]))
    hits, retrieval_s = _measure(lambda: store.search(query_vector[0], 5))
    _, warm_query_embed_s = _measure(lambda: encoder.encode([QUERY]))
    _, warm_retrieval_s = _measure(lambda: store.search(query_vector[0], 5))
    result_ids = [hit.chunk_id for hit in hits]
    report = {
        "schema_version": 1,
        "workload": {"source": "user_pdf" if document else "synthetic_pdf",
                     "sha256": file_hash, "pages": len(pages), "chunks": len(chunks),
                     "query": QUERY},
        "hardware": {"torch": encoder.torch_version,
                     "cuda_available": encoder.cuda_available,
                     "cuda_device_name": encoder.cuda_device_name,
                     "selected_device": encoder.device,
                     "original_max_sequence_length": original_limit},
        "token_lengths": {"total": sum(lengths), "mean": statistics.mean(lengths),
                          "p95": sorted(lengths)[int(0.95 * (len(lengths) - 1))],
                          "max": max(lengths)},
        "timings_seconds": {
            "file_hash": hash_s, "pdf_parse": parse_s,
            "text_normalization": 0.0, "chunking": chunk_s,
            "parsed_cache_write": cache_write_s, "parsed_cache_read": cache_read_s,
            "model_initialization": model_load_s, "tokenization": tokenization_s,
            "chroma_initialization": store_init_s, "chroma_insertion": insertion_s,
            "query_embedding": query_embed_s, "vector_retrieval": retrieval_s,
            "warm_query_embedding": warm_query_embed_s,
            "warm_vector_retrieval": warm_retrieval_s,
        },
        "throughput": {"pdf_pages_per_second": len(pages) / parse_s},
        "batch_comparison": batch_results,
        "fastest_batch_on_this_workload": fastest["batch_size"],
        "shorter_sequence_trial": sequence_result,
        "cache_hit_confirmed": cache_hit,
        "top_chunk_ids": result_ids,
    }
    baseline_file = project_root / "data" / "baseline_metrics.json"
    if baseline_file.is_file():
        baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
        if (baseline.get("workload", {}).get("document") == source.name
                and baseline.get("workload", {}).get("chunks") == len(chunks)):
            report["baseline_comparison"] = {
                "baseline_embedding_seconds": baseline["timings_seconds"]["embedding_inference"],
                "baseline_pdf_parse_seconds": baseline["timings_seconds"]["pdf_parse"],
                "baseline_chroma_insertion_seconds": baseline["timings_seconds"]["chroma_insertion"],
                "baseline_model_load_seconds": baseline["timings_seconds"]["model_initialization"],
                "top_ids_match": result_ids == baseline["results"]["temporary_top_ids"],
            }
    del encoder
    gc.collect()
    report["timings_seconds"]["cli_stats_process"] = _subprocess_seconds(temp_settings, ["stats"])
    report["timings_seconds"]["cold_cli_search_process"] = _subprocess_seconds(
        temp_settings, ["search", QUERY, "--top-k", "5"],
    )
    report["timings_seconds"]["total_benchmark"] = time.perf_counter() - run_started
    results_dir = project_root / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = results_dir / f"benchmark-{timestamp}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = (
        f"Device: {report['hardware']['selected_device']} | PyTorch {report['hardware']['torch']}\n"
        f"PDF: {len(pages)} pages in {parse_s:.3f}s; {len(chunks)} chunks, "
        f"{sum(lengths)} tokens (mean {statistics.mean(lengths):.1f}, "
        f"p95 {report['token_lengths']['p95']})\n"
        f"Model load: {model_load_s:.3f}s | fastest batch: {fastest['batch_size']} "
        f"({fastest['seconds']:.3f}s) | Chroma insert: {insertion_s:.3f}s\n"
        f"Warm query: {warm_query_embed_s + warm_retrieval_s:.3f}s | "
        f"cold CLI search: {report['timings_seconds']['cold_cli_search_process']:.3f}s"
    )
    return output, summary
