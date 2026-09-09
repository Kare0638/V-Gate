# What the A100 run cost, and what it found

The measurement is in
[`scaling_a100_2gpu.md`](../../benchmarks/results/scaling_a100_2gpu.md):
**1.99x across two A100s**, 5.4 to 10.8 req/s, p95 halved, requests split
192/192, zero failures over three repeats.

This is the record of getting there, because the four bugs between the first
run and that number are the part worth reading. Every one of them produced
output that looked like a result.

## The setup

2x A100-SXM4-80GB, `NV12` between them, `Qwen2.5-32B-Instruct` at FP16 — one
worker per GPU, pinned with `CUDA_VISIBLE_DEVICES`. Data parallelism across
workers, which is what this project contributes; sharding a model across
devices is the engine's job.

## Four failures, in the order they appeared

**`ninja` was not on PATH.** vLLM needs it to compile CUDA graphs. It is
installed inside the venv, not the system path, so the engine died during
warmup with `FileNotFoundError`. Only a real engine reaches that code.

**Every request failed, at several hundred "req/s".** The rate was real and
meaningless: throughput counts requests over wall time regardless of outcome,
so it was measuring how fast requests were being *rejected*. The cause was
`max_tokens=0` — deliberate for the dry-run backend, where it makes the
synthetic cost exactly the configured latency, and rejected by vLLM, which
requires at least 1. The worker raised on every call and the gateway demoted
it. The harness now refuses to start a real-engine run with a value below 1.

**A two-worker run showed no speedup, and only the distribution said why.**
5.6 req/s against 5.4, which reads as "a second GPU buys nothing" — but the
requests had split 80/16 instead of 48/48. The topology waited on `/ready`,
which reports whether the gateway has *a* usable worker. That is the right
question for Kubernetes and the wrong one for a measurement: it returned as
soon as the first worker was admitted, and the sweep ran against a
half-available pool. It now waits for every worker.

**Throughput was pinned at 5.4 req/s no matter what.** Fixing the split gave
a perfect 48/48 and *still* no speedup. Raising offered concurrency 8x moved
p95 from 6s to 53s and left throughput unchanged — the signature of a hard
ceiling upstream of the workers.

`RemoteBackend.generate` is a synchronous httpx call dispatched through
`run_in_executor`, so outbound concurrency is capped by the default thread
pool: `min(32, cpu_count + 4)`. At ~6 seconds per generation that is
`32 / 6 = 5.3 req/s`, and **no number of GPUs moves it**.

This limit was already documented. It had been measured on a laptop, where
100ms synthetic generations made it a ~180 req/s ceiling that looked like
comfortable headroom. Real generations are sixty times slower, and the same
32 threads became the binding constraint on the entire system. Sizing the
pool to the admission limit restored linear scaling.

It is a bound, not a fix. Threads blocked on HTTP are pure overhead, and an
async client would take them out of the path; that stays in ROADMAP.

## What this says about the earlier evidence

The synthetic 1-vs-N report was not wrong — it scales the same way, and its
caveat about measuring gateway fan-out rather than GPU throughput was
accurate. What it could not show is which constraint binds first at real
timescales. A ceiling that reads as headroom at 100ms is the whole system at
6s, and only the real engine made that visible.
