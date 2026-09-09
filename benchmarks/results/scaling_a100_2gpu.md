# 1-vs-N Worker Scaling

Measured on 2026-09-09 02:12 UTC on rented 2x A100-SXM4-80GB.

Rendered from `scaling_a100_2gpu.json` — the raw data of that run — after two
reporting bugs were fixed: a sentence about concurrent slots that only holds
for the synthetic backend, and a missing hardware block. The measurements are
unchanged and were not re-run; only the prose around them was regenerated.
Do not edit by hand.

## What this measures, and what it does not

One gateway in front of N worker processes, same load at each N.

Each worker runs a **real vLLM engine on its own GPU**, so this is
throughput the hardware actually produced — not the synthetic
dry-run backend the earlier report used. The gateway holds no model;
it routes.

There is no declared per-worker capacity to measure against here, so
the baseline is the **measured** single-worker rate and the ideal is
linear scaling from it. The gap is what the gateway and the pool cost
together.

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-32B-Instruct` |
| Quantization | none (native precision) |
| Max model length | 2048 |
| GPU memory utilization | 0.95 |
| GPUs per worker | 1 (`CUDA_VISIBLE_DEVICES` pinned per process) |
| Client concurrency | 128 |
| Requests per run | 384 |
| Repeats per point | 3 |
| Gateway admission limit | 256 |
| **Measured single-worker rate** | **5.42 req/s** |

The hardware, captured rather than described. A throughput figure
without it cannot be reproduced or compared to a run anywhere else:

```
index, name, memory.total [MiB], driver_version
0, NVIDIA A100-SXM4-80GB, 81920 MiB, 580.126.16
1, NVIDIA A100-SXM4-80GB, 81920 MiB, 580.126.16
```

```
GPU0	GPU1	CPU Affinity	NUMA Affinity
GPU0	 X 	NV12	0-63	0
GPU1	NV12	 X 	64-127	1
```

Prompts are all distinct, so neither the result cache nor in-flight
deduplication can stand in for work the pool actually did.

## Throughput

| Workers | Ideal req/s | Median req/s | Spread | Efficiency | Speedup vs 1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | **5.4** | 5.4–5.4 | 100% | 1.00x |
| 2 | 11 | **10.8** | 10.8–10.8 | 100% | 1.99x |

Efficiency is measured over ideal. The shortfall is what the gateway
costs: one HTTP hop per request, admission bookkeeping, and the
round-robin choice.

## Latency

| Workers | p50 | p95 | p99 |
|---:|---:|---:|---:|
| 1 | 23574 ms | 28259 ms | 29183 ms |
| 2 | 11775 ms | 14460 ms | 17568 ms |

At client concurrency 128, how much of that a pool
can hold is set by KV cache, not by a configured number: a worker
admits sequences until its cache is full and queues the rest. So the
latency below is mostly queueing, and halving the queue per GPU is
what halves it.

## Request distribution

Round-robin, so an even split is the expected result. An uneven one would
mean the routing is not doing what it claims.

- **1 worker(s)**: `8111`: 384
- **2 worker(s)**: `8111`: 192, `8112`: 192

## Sanity checks

- **scaling runs** (6 run(s)): 0 failed, 0 cache hit(s), 0 deduplicated.

All three counts must be zero in every scenario. A failed request
inflates nothing but still divides into wall time; a cache hit or a
coalesced duplicate is throughput the pool did not actually produce.
