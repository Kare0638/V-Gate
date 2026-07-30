# V-Gate Benchmark Report

Generated: 2026-07-30T06:36:26.561474+00:00
Engine: dry-run
Concurrency: 2, requests per scenario: 10

> **Dry-run baseline.** No GPU/model is used; the backend is a mock that echoes a fixed response after a synthetic delay of `15ms + 2ms * max_tokens` per batch call (see `VGATE_DRYRUN_SIMULATED_LATENCY_MS` in `vgate/backends/base.py`). This isolates the batcher/cache/HTTP-layer behavior from real inference cost and is **not** a GPU throughput measurement. A vLLM/SGLang single-worker baseline on real hardware is still needed (`--engine-type vllm|sglang`) before quoting real tokens/sec numbers.

## Baseline (max_batch_size=8, all-unique prompts)

- Concurrency: 2
- Total requests: 10
- Unique prompts: 64
- Max tokens: 64
- Failures: 0
- Wall time: 0.9369s

| Metric | Value |
|---|---|
| Latency mean (s) | 0.1873 |
| Latency p50 (s) | 0.1938 |
| Latency p95 (s) | 0.1953 |
| Latency p99 (s) | 0.1953 |
| Latency max (s) | 0.1953 |
| Tokens/sec | 85.39 |
| Requests/sec | 10.67 |
| Avg batch size | 2.0 |
| Batches formed | 5 |
| Deduplicated requests | 0 |
| Avg queue time (s) | 0.0406 |
| Avg TTFT (s) | 0.0 |
| Avg TPOT (s) | 0.009 |
| Cache hit rate (this run) | 0.0 |
| Cache hits / misses (this run) | 0 / 10 |

## Batch size sweep: max_batch_size=1

- Concurrency: 2
- Total requests: 10
- Unique prompts: 64
- Max tokens: 64
- Failures: 0
- Wall time: 1.4431s

| Metric | Value |
|---|---|
| Latency mean (s) | 0.2742 |
| Latency p50 (s) | 0.288 |
| Latency p95 (s) | 0.2911 |
| Latency p99 (s) | 0.2911 |
| Latency max (s) | 0.2911 |
| Tokens/sec | 55.44 |
| Requests/sec | 6.93 |
| Avg batch size | 1.0 |
| Batches formed | 10 |
| Deduplicated requests | 0 |
| Avg queue time (s) | 0.1278 |
| Avg TTFT (s) | 0.0 |
| Avg TPOT (s) | 0.0179 |
| Cache hit rate (this run) | 0.0 |
| Cache hits / misses (this run) | 0 / 10 |

## Batch size sweep: max_batch_size=4

- Concurrency: 2
- Total requests: 10
- Unique prompts: 64
- Max tokens: 64
- Failures: 0
- Wall time: 0.9496s

| Metric | Value |
|---|---|
| Latency mean (s) | 0.1899 |
| Latency p50 (s) | 0.1941 |
| Latency p95 (s) | 0.1946 |
| Latency p99 (s) | 0.1946 |
| Latency max (s) | 0.1946 |
| Tokens/sec | 84.24 |
| Requests/sec | 10.53 |
| Avg batch size | 2.0 |
| Batches formed | 5 |
| Deduplicated requests | 0 |
| Avg queue time (s) | 0.043 |
| Avg TTFT (s) | 0.0 |
| Avg TPOT (s) | 0.0089 |
| Cache hit rate (this run) | 0.0 |
| Cache hits / misses (this run) | 0 / 10 |

## Batch size sweep: max_batch_size=16

- Concurrency: 2
- Total requests: 10
- Unique prompts: 64
- Max tokens: 64
- Failures: 0
- Wall time: 0.9298s

| Metric | Value |
|---|---|
| Latency mean (s) | 0.1858 |
| Latency p50 (s) | 0.1941 |
| Latency p95 (s) | 0.1958 |
| Latency p99 (s) | 0.1958 |
| Latency max (s) | 0.1958 |
| Tokens/sec | 86.04 |
| Requests/sec | 10.76 |
| Avg batch size | 2.0 |
| Batches formed | 5 |
| Deduplicated requests | 0 |
| Avg queue time (s) | 0.0388 |
| Avg TTFT (s) | 0.0 |
| Avg TPOT (s) | 0.0089 |
| Cache hit rate (this run) | 0.0 |
| Cache hits / misses (this run) | 0 / 10 |

## Batch size sweep: max_batch_size=32

- Concurrency: 2
- Total requests: 10
- Unique prompts: 64
- Max tokens: 64
- Failures: 0
- Wall time: 0.937s

| Metric | Value |
|---|---|
| Latency mean (s) | 0.1873 |
| Latency p50 (s) | 0.1939 |
| Latency p95 (s) | 0.1942 |
| Latency p99 (s) | 0.1942 |
| Latency max (s) | 0.1942 |
| Tokens/sec | 85.38 |
| Requests/sec | 10.67 |
| Avg batch size | 2.0 |
| Batches formed | 5 |
| Deduplicated requests | 0 |
| Avg queue time (s) | 0.0406 |
| Avg TTFT (s) | 0.0 |
| Avg TPOT (s) | 0.009 |
| Cache hit rate (this run) | 0.0 |
| Cache hits / misses (this run) | 0 / 10 |

## Cache impact: 3 repeated prompts (high reuse) vs. baseline's all-unique prompts

- Concurrency: 2
- Total requests: 10
- Unique prompts: 3
- Max tokens: 64
- Failures: 0
- Wall time: 0.381s

| Metric | Value |
|---|---|
| Latency mean (s) | 0.0761 |
| Latency p50 (s) | 0.002 |
| Latency p95 (s) | 0.1944 |
| Latency p99 (s) | 0.1944 |
| Latency max (s) | 0.1944 |
| Tokens/sec | 209.96 |
| Requests/sec | 26.24 |
| Avg batch size | 2.0 |
| Batches formed | 2 |
| Deduplicated requests | 1 |
| Avg queue time (s) | 0.0401 |
| Avg TTFT (s) | 0.0 |
| Avg TPOT (s) | 0.0119 |
| Cache hit rate (this run) | 0.6 |
| Cache hits / misses (this run) | 6 / 4 |
