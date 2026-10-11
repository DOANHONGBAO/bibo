# BIBO retrieval performance measurements

Measurements use one local 8-page text PDF (16 chunks, 9,565 BGE-M3 tokens) and the fixed query “What methods are evaluated in this paper?”. Only timings and aggregate counts are recorded here; source text and generated indexes stay under ignored `data/` paths. Timing uses `time.perf_counter()`. A separate process measures the user-facing cold CLI search.

## Baseline before optimization

Machine: Windows, Python 3.11.17, PyTorch 2.14.1+cpu, 8 logical CPUs, 4 PyTorch CPU threads; CUDA unavailable. Embedding batch size was 16 and model maximum sequence length was 8,192 tokens. Mean chunk length was 597.8 tokens, p95 808, maximum 946.

| Stage | Seconds |
| --- | ---: |
| File hashing | 0.0014 |
| PDF text extraction | 0.1771 |
| Chunking | 0.0017 |
| Model initialization | 16.8016 |
| Tokenization measurement | 0.0084 |
| Embedding 16 chunks | 47.2496 |
| ChromaDB insertion | 0.3975 |
| Query embedding with model loaded | 0.1319 |
| Vector retrieval from test index | 0.0059 |
| New CLI search process, total wall time | 25.993 |

PDF extraction ran at about 45 pages/second. Embedding throughput was 0.339 chunks/second or 202 tokens/second. A `stats` CLI process took 2.105 seconds including Python startup and local Chroma initialization. No text normalization step existed; source text was retained verbatim.

The bottleneck is BGE-M3 inference on CPU, followed by cold model initialization on each separate search command. PDF parsing and Chroma insertion are comparatively small. The old partial-ingestion path deleted already indexed chunks, and the old batch insertion rejected a document with any existing chunks; the new path repairs only missing chunks. The former `fitz` import also emitted a deprecation warning.

## CPU batch probe on the same 16 chunks

The model was loaded once, then each batch size encoded the same ordered chunks. Working-set memory is approximate process RSS after each trial; the trials ran sequentially, so memory values can retain allocator caches.

| Batch | Seconds | Chunks/second | RSS after, MiB |
| ---: | ---: | ---: | ---: |
| 4 | 38.874 | 0.412 | 2,001 |
| 8 | 42.443 | 0.377 | 2,195 |
| 16 | 41.705 | 0.384 | 2,523 |
| 8, max length 1,024 | 46.568 | 0.344 | 2,196 |

Batch 4 was fastest in this probe and used less memory. All input lengths fit within 1,024 tokens for this PDF, but lowering the sequence limit did not improve speed, so the model default remains unchanged. FP16 and CUDA were not evaluated because this PyTorch installation has no CUDA support. These measurements do not establish cross-language retrieval quality.

## After optimization, same PDF and query

The detailed machine-readable report is generated locally in `benchmarks/results/` (ignored by Git). On this run, PyTorch was still CPU-only, and the 16 chunks had the same 9,565 tokens. The top five retrieved chunk IDs matched the baseline test index exactly.

| Stage | Before, seconds | After, seconds | Interpretation |
| --- | ---: | ---: | --- |
| PDF text extraction | 0.177 | 0.162 | Small relative cost; cache hit was 0.008 s. |
| Model initialization | 16.802 | 22.171 | Cold load varies; separate CLI processes still reload. |
| Embed 16 chunks | 47.250 at batch 16 | 34.388 at batch 4 | 27% less elapsed time in these runs. |
| ChromaDB insertion | 0.398 | 1.241 | Still small compared with embedding; no speedup claim. |
| New CLI search, wall time | 25.993 | 26.164 | Essentially unchanged, as expected for a new process. |
| Warm query encoding + retrieval | about 0.138 | 0.125 | The `interactive` command reuses the loaded model. |

Within the after-optimization run, batch 4 took 34.388 s (0.465 chunks/s, about 2,005 MiB RSS after); batch 8 took 42.700 s; batch 16 took 48.798 s (about 2,527 MiB RSS after). The same batch 4 with a safe 1,024-token maximum took 34.024 s, within normal run-to-run variation, so the model's 8,192-token limit remains. The observed speed difference across runs cannot be attributed solely to code changes because CPU load, disk cache and warm-up also vary; the within-run batch comparison supports choosing batch 4 on this hardware.

The cold command remains slow because a new Python process loads about 2 GB of model weights. In an interactive session, the model and Chroma collection stay open and subsequent query encoding plus retrieval measured about 0.125 s on this workload. This is a latency measurement, not a validation of semantic relevance. The PDF cache stores extracted text locally under ignored `data/parsed_cache/`; no original document or vector was removed during optimization.

## Reproduce

```powershell
python -m bibo.cli prepare-model --offline
python -m bibo.cli benchmark --document "documents\YOUR_PDF.pdf"
python -m bibo.cli search "What methods are evaluated in this paper?" --profile
```

The benchmark writes detailed JSON to ignored `benchmarks/results/` and uses a separate temporary Chroma collection under ignored `data/benchmark_runs/`. It does not modify the research index. Results can vary with OS cache, CPU contention and thermal state; compare stage timings on the same workload and check result IDs before claiming a speedup.
