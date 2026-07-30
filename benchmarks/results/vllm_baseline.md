# V-Gate vLLM Benchmark (RTX 3060 Laptop, real GPU)

Generated: 2026-07-30T06:56:04.271430+00:00

> **Real GPU baseline.** NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM, ~5GB free), model `Qwen/Qwen2.5-1.5B-Instruct-AWQ`, vLLM 0.26.0, CUDA 12.4, WSL2. This replaces the dry-run-only numbers in `baseline.md` for the batch-size and cache-impact stories.
>
> **Two real bugs were found and fixed while producing this data** (see git history for `vgate/backends/vllm_backend.py` and `benchmarks/run_report.py`):
> 1. vLLM under WSL2 defaults pinned memory off out of caution even on kernels that support it; without `VLLM_WSL2_ENABLE_PIN_MEMORY=1` the engine crashes at startup ("UVA is not available").
> 2. `VLLMBackend` was reading `metrics.first_token_time`/`arrival_time`/`finished_time`, fields that no longer exist on installed vLLM's `RequestStateStats` (now `first_token_ts`/`last_token_ts`/`first_token_latency`), AND `LLM()` defaults `disable_log_stats=True` unlike `EngineArgs`. Together these meant TTFT/TPOT were silently always 0 for the real vLLM backend before this run — nobody would have noticed without testing against a live GPU.
>
> **Known caveat in this data:** `max_batch_size` is a trigger threshold for when `RequestBatcher` starts draining its queue, not a hard cap on how many requests get pulled into one batch (`_process_batch` drains the *entire* current queue). Under concurrent burst arrival, the `max_batch_size=1` scenario below still shows an average realized batch size of 4.0, not 1 — this is a real gap in `vgate/batcher.py`, not a benchmark artifact. The `cache impact` scenario also shows `avg_ttft_s: 0.0`; its single real (non-cached) batch completed fast enough that vLLM's async stats snapshot did not capture a first-token timestamp for it — treat the batch_1/8/32 TTFT numbers as the reliable ones.
>
> **Sample size:** each scenario is 24-40 requests / 5-6 batches on a laptop GPU with only one run each — enough to show the batch_1 vs. batch_8 direction clearly, but batch_8 vs. batch_32 (which realize the same avg batch size of 8.0 under this concurrency) differ mostly by run-to-run noise, not a real effect. Don't over-read small deltas between those two.

## Batch size sweep: max_batch_size=1

- Concurrency: 8
- Total requests: 24
- Unique prompts: 64
- Max tokens: 64
- Failures: 0
- Wall time: 6.3054s

| Metric | Value |
|---|---|
| Latency mean (s) | 2.055 |
| Latency p50 (s) | 2.1717 |
| Latency p95 (s) | 2.6624 |
| Latency p99 (s) | 2.6626 |
| Latency max (s) | 2.6626 |
| Tokens/sec | 171.12 |
| Requests/sec | 3.81 |
| Avg batch size | 4.0 |
| Batches formed | 6 |
| Deduplicated requests | 0 |
| Avg queue time (s) | 0.9553 |
| Avg TTFT (s) | 0.0506 |
| Avg TPOT (s) | 0.0164 |
| Cache hit rate (this run) | 0.0 |
| Cache hits / misses (this run) | 0 / 24 |

## Batch size sweep: max_batch_size=8

- Concurrency: 8
- Total requests: 40
- Unique prompts: 64
- Max tokens: 64
- Failures: 0
- Wall time: 6.1817s

| Metric | Value |
|---|---|
| Latency mean (s) | 1.2358 |
| Latency p50 (s) | 1.1936 |
| Latency p95 (s) | 1.5264 |
| Latency p99 (s) | 1.5268 |
| Latency max (s) | 1.5268 |
| Tokens/sec | 294.09 |
| Requests/sec | 6.47 |
| Avg batch size | 8.0 |
| Batches formed | 5 |
| Deduplicated requests | 0 |
| Avg queue time (s) | 0.0004 |
| Avg TTFT (s) | 0.0996 |
| Avg TPOT (s) | 0.0174 |
| Cache hit rate (this run) | 0.0 |
| Cache hits / misses (this run) | 0 / 40 |

## Batch size sweep: max_batch_size=32

- Concurrency: 8
- Total requests: 40
- Unique prompts: 64
- Max tokens: 64
- Failures: 0
- Wall time: 9.6132s

| Metric | Value |
|---|---|
| Latency mean (s) | 1.922 |
| Latency p50 (s) | 1.9941 |
| Latency p95 (s) | 2.1149 |
| Latency p99 (s) | 2.1151 |
| Latency max (s) | 2.1151 |
| Tokens/sec | 189.12 |
| Requests/sec | 4.16 |
| Avg batch size | 8.0 |
| Batches formed | 5 |
| Deduplicated requests | 0 |
| Avg queue time (s) | 0.039 |
| Avg TTFT (s) | 0.1265 |
| Avg TPOT (s) | 0.0268 |
| Cache hit rate (this run) | 0.0 |
| Cache hits / misses (this run) | 0 / 40 |

## Cache impact: 3 repeated prompts (high reuse) vs. baseline's all-unique prompts

- Concurrency: 8
- Total requests: 40
- Unique prompts: 3
- Max tokens: 64
- Failures: 0
- Wall time: 1.3958s

| Metric | Value |
|---|---|
| Latency mean (s) | 0.2787 |
| Latency p50 (s) | 0.0072 |
| Latency p95 (s) | 1.366 |
| Latency p99 (s) | 1.3663 |
| Latency max (s) | 1.3663 |
| Tokens/sec | 1834.03 |
| Requests/sec | 28.66 |
| Avg batch size | 8.0 |
| Batches formed | 1 |
| Deduplicated requests | 5 |
| Avg queue time (s) | 0.0013 |
| Avg TTFT (s) | 0.0 |
| Avg TPOT (s) | 0.0173 |
| Cache hit rate (this run) | 0.8 |
| Cache hits / misses (this run) | 32 / 8 |
