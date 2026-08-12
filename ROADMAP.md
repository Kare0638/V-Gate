# V-Gate Distributed Inference Roadmap

> **Priority:** the `Phase 0-8` numbering below is internal to this document and does **not** express project priority. [README.md](README.md)'s Roadmap section holds the authoritative ordering.
>
> Goal: evolve from a single-node LLM gateway into distributed inference serving with reliable behavior, observability, routing, and measurable performance evidence.
>
> Strategy: fix correctness and measurement first, then add production reliability, then distributed serving, and only then lower-level performance modules. Jumping directly to C++/CUDA would make the system harder to evaluate unless the main request path, benchmark story, and failure model are already solid.

---

## 1. Project Positioning

The target is a distributed LLM inference system: a gateway tier that owns admission, batching, caching, and routing, in front of a pool of inference workers. The gateway tier exists today; the worker tier does not.

What is implemented is an OpenAI-style LLM gateway with:

- FastAPI API gateway
- vLLM / SGLang backend adapters
- dynamic micro-batching and batch-level deduplication
- LRU result cache
- Prometheus metrics
- structured logging
- OpenTelemetry tracing
- API key authentication and rate limiting
- Docker / docker-compose / Kubernetes manifests
- Python client SDK

Current gaps:

- OpenAI API compatibility is still shallow. Streaming (`stream: true`, SSE) now works end-to-end for the dry-run backend and real vLLM (verified on GPU); SGLang still raises a clear `NotImplementedError` until its async engine path lands (Phase 2 task 8).
- `RequestBatcher` still forms a static, sealed Python-side batch per window (queue drain, `max_batch_size`/`max_wait_time_ms`) before handing prompts to the backend — new requests still can't join an already-formed batch at that layer. Below that layer, `VLLMBackend` now submits each prompt as an independent `AsyncLLMEngine.generate()` call, so vLLM's own scheduler does get to interleave/continuously-batch them at the GPU level even though the Python-side batch above it is still static. Full task 9 (redefine `RequestBatcher` as dedup/admission/fan-out, not batch construction) is not done.
- The runtime is still a single gateway with a single local backend, not real multi-worker serving.
- Dry-run and single-GPU vLLM reports are checked in, but they predate parts of the current async serving path. A refreshed streaming baseline, repeat-run variance analysis, and 1-vs-N-worker evidence are still missing.
- Cache is RAM-only. There is no persistent local disk cache layer, and cache value depends on process lifetime. This is an intentional, benchmark-gated decision (Phase 1.5), not an oversight — current traffic shows no L1 eviction pressure.
- Backpressure, request timeout, circuit breaking, and worker failure handling are incomplete.
- The embedding endpoint is currently a mock MVP implementation.
- The v2 C++/CUDA work is documented as a proposal, not implemented runtime code.

---

## 2. Engineering Maturity Targets

### Level 1: Single-Node Gateway

Goal: make the project a credible single-node LLM serving gateway rather than a toy demo. It should run locally and in containers, have tests, clear documentation, and enough observability to debug normal traffic.

The design should be able to explain:

- Why a gateway layer is useful.
- How batching can improve GPU utilization.
- How cache keys are designed.
- How rate limiting protects the service.
- How Prometheus metrics help locate latency issues.
- How Docker and Kubernetes manifests deploy the system.

Completion criteria:

- README gets a new user running in roughly 5 minutes.
- Dry-run mode is stable.
- Unit and integration tests pass.
- `/v1/chat/completions` supports basic non-streaming calls.
- `/metrics` and `/stats` explain current service state.
- A baseline benchmark report exists.

### Level 2: Reliable LLM Serving

Goal: show that the system handles real LLM serving concerns: streaming, tail latency, overload, concurrency, and failure isolation.

The design should be able to explain:

- The difference between TTFT and TPOT.
- How SSE streaming reduces perceived latency.
- How queue timeout and load shedding protect p95/p99 latency.
- Why true continuous batching should be delegated to vLLM/SGLang schedulers instead of hand-rolled in the Python gateway.
- How batching should behave when requests use different sampling parameters.
- Why circuit breakers are needed.
- How metrics indicate whether bottlenecks are in the gateway, queue, backend, or GPU.

Completion criteria:

- `/v1/chat/completions` supports streaming.
- vLLM/SGLang async serving paths are available for token-level scheduling and continuous batching.
- Batching parameter semantics are correct.
- Backpressure and request timeouts exist.
- Benchmarks report p50/p95/p99.
- A Grafana dashboard or dashboard JSON/screenshots exist.
- Failure injection tests exist.

### Level 3: Distributed Inference Serving

Goal: evolve the system from a single-node service wrapper into distributed inference serving with multiple workers, routing, health checks, and failure isolation.

The design should be able to explain:

- How the worker registry is maintained.
- How health checks affect routing.
- The tradeoffs between round-robin, least-inflight, and EWMA latency routing.
- How workers are removed and recovered by circuit breakers.
- Whether caching should live at the gateway level, worker level, or in a distributed cache.
- Whether autoscaling should use CPU, GPU utilization, queue length, tokens/sec, or latency.
- How rollout and rollback should work.

Completion criteria:

- Multiple workers are supported.
- The gateway performs worker discovery, health checks, and routing.
- EWMA or least-inflight routing is implemented.
- Circuit breakers exist.
- Worker failure tests exist.
- End-to-end benchmark compares 1 worker vs N workers.
- Kubernetes can deploy gateway + workers.

---

## 3. Recommended Implementation Order

This ordering is internal to this document. [README.md](README.md)'s Roadmap section holds the authoritative project priority.

### Phase 0: Credibility Fixes

**Status: done.** All six tasks are implemented and covered by tests (`pytest tests/ vgate-client/tests/` — 146 + 48 passing):

1. `cache.enabled=False` now makes `ResultCache.get/put` a true no-op instead of being silently ignored.
2. `RequestBatcher._process_batch` now groups deduplicated requests by `(temperature, top_p, max_tokens)` and dispatches one backend call per group, so requests with different sampling params can no longer silently share (and corrupt) each other's params. Fixing this also fixed the `max_batch_size`-is-only-a-trigger gap noted in Phase 1: `_process_batch` now drains at most `max_batch_size` requests per call instead of the whole queue, and `stop()` loops until the queue is fully drained so a burst larger than one batch can't strand unresolved requests on shutdown.
3. `RequestBatcher.submit()` takes an optional `timeout` and cleans up the queue entry on both `asyncio.TimeoutError` and caller-side cancellation.
4. `ChatCompletionRequest.messages` is now `list[ChatMessage]` (was a bare, unvalidated `list`) — a malformed message now returns 422 instead of an unhandled 500 (`'str' object has no attribute 'get'`).
5. README documents `/v1/embeddings` as a mock MVP implementation.
6. `.github/workflows/tests.yml` runs the dry-run server suite and the client SDK suite on every push/PR to `main`.

Priority: highest.

Reason: these issues affect project trust. Fix the semantics of the main path before expanding the architecture.

Tasks:

1. Make `cache.enabled` actually take effect.
2. Fix batching behavior when sampling parameters differ.
3. Add request timeout and cancellation tests.
4. Improve OpenAI-style request/response models.
5. Document that embeddings are currently mock.
6. Add CI.

Recommended implementation:

- `ResultCache.get/put` should become miss/no-op when cache is disabled.
- Batch by `(temperature, top_p, max_tokens)`, or only batch requests with compatible parameters.
- This fix applies to the legacy static micro-batching path until Phase 2 moves GPU batching responsibility into the async engine scheduler.
- Add a timeout option to `RequestBatcher.submit()`.
- Run `VGATE_DRY_RUN=true pytest` in GitHub Actions.

Acceptance criteria:

- `pytest tests vgate-client/tests` passes.
- README does not overstate unimplemented behavior.
- `/stats` reflects the cache disabled state.
- Every config option can be explained against actual runtime behavior.

Expected outcome:

- The project moves from feature collection to credible engineering baseline.
- The single-node gateway becomes safe to present publicly.

---

### Phase 1: Benchmark And Observability Report

**Status: done.** `benchmarks/bench_load.py` (concurrent HTTP load generator) and `benchmarks/run_report.py` (multi-scenario orchestrator) are implemented; `benchmarks/results/baseline.md` (dry-run) and `benchmarks/results/vllm_baseline.md` (real GPU: RTX 3060 Laptop, Qwen2.5-1.5B-AWQ, vLLM 0.26) are both checked in. Batcher tracks `avg_queue_time_s`/`avg_ttft_s`/`avg_tpot_s` and `/v1/benchmark` reports TTFT/TPOT percentiles plus batch/cache stats.

Producing the real GPU baseline surfaced and fixed three bugs that only show up against live hardware/dependencies: (1) `time.time()` for queue-time deltas could go negative on wall-clock adjustment, now `time.monotonic()`; (2) vLLM under WSL2 needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1` or it crashes at startup; (3) `VLLMBackend` read renamed/removed vLLM metrics fields plus relied on a stats-collection flag that `LLM()` defaults off, so TTFT/TPOT were silently always 0 for the real backend. It also surfaced a fourth issue, fixed in Phase 0: `max_batch_size` was only a trigger threshold, not a hard cap. See `benchmarks/results/vllm_baseline.md` for the original data.

Priority: highest.

Reason: AI infrastructure work needs data. Without benchmark evidence, optimization claims are weak.

Tasks:

1. Extend the benchmark CLI.
2. Add a concurrent load benchmark.
3. Record p50 / p95 / p99.
4. Record TTFT / TPOT / tokens/sec.
5. Record cache hit rate, batch size, and queue time.
6. Record RAM cache pressure: working-set size, eviction rate, and repeated requests beyond L1 capacity.
7. Output a Markdown benchmark report.

Recommended implementation:

- Add `benchmarks/bench_load.py`.
- Support:
  - `--concurrency`
  - `--requests`
  - `--prompt-file`
  - `--stream` once Phase 2 streaming is available
  - `--output json|markdown`
- Store reports under `benchmarks/results/`.

The report should include:

- dry-run baseline
- vLLM single-worker baseline
- batch size impact on latency and throughput
- cache hit impact on throughput
- cache working-set and eviction-pressure analysis
- p95/p99 tail-latency analysis

Acceptance criteria:

- One command generates a benchmark report.
- README links to the report.
- The report explains throughput, latency, and resource-use tradeoffs with data.

Expected outcome:

- The project moves from "batching exists" to "batching was measured and analyzed."
- This is the main bridge from a basic gateway to reliable serving work.

---

### Phase 1.5: SQLite L2 Cache Decision Gate

**Status: decided — deferred.** `cache.enabled` now works correctly (Phase 0), and both benchmark reports (`benchmarks/results/baseline.md`, `benchmarks/results/vllm_baseline.md`) are in. Across every scenario in both reports, cache size never exceeded single digits against a default `maxsize` of 1000 — no run came close to `ResultCache`'s L1 eviction threshold, so there is no measured working-set pressure to justify an L2. Per this phase's own acceptance criteria, L2 implementation stays deferred until a real workload (or a benchmark scenario deliberately designed to exceed `maxsize`) shows L1 eviction actually happening. The design below is kept as-is for when that evidence exists.

Priority: medium, benchmark-gated design gate.

Reason: a local disk cache can be useful when repeated requests exceed RAM cache capacity, but it should not be implemented before `cache.enabled` works correctly and benchmark data shows real eviction pressure. This phase decides whether L2 is worth building; implementation remains deferred unless data supports it.

Design:

- L1: current RAM `ResultCache`, small and fast.
- L2: local sqlite3 cache, larger and persistent across process restarts.
- SQLite should run in WAL mode.
- Synchronous sqlite3 calls should be wrapped in `run_in_executor` to avoid blocking the event loop.
- Use write-through semantics: `put()` writes to both L1 and L2.
- Use promote-on-read semantics: L2 hits are copied back into L1.

Read path:

```text
get(key)
  -> L1 RAM hit: return
  -> L1 miss: check L2 sqlite
  -> L2 hit: promote to L1, return
  -> L2 miss: true miss, call backend
```

Write path:

```text
put(key, value)
  -> write L1
  -> write L2
```

Required metrics:

- `vgate_cache_l1_hits_total`
- `vgate_cache_l2_hits_total`
- `vgate_cache_true_misses_total`
- `vgate_cache_l1_evictions_total`
- `vgate_cache_l2_evictions_total`
- `vgate_cache_l2_lookup_seconds`
- `vgate_cache_l2_write_seconds`

Acceptance criteria:

- Benchmark report includes cache working-set size, L1 eviction pressure, and repeated requests beyond L1 capacity.
- A short design note explains whether sqlite L2 is justified for the measured workload.
- If justified, the design specifies independent enable/disable flags, WAL mode, executor-wrapped sqlite I/O, write-through writes, and promote-on-read behavior.
- If not justified, the roadmap explicitly defers L2 cache implementation.

Expected outcome:

- Cache architecture has a measured decision point. L1/L2 implementation only proceeds when traffic data justifies the added complexity.

---

### Phase 2: Streaming And Engine-Native Continuous Batching

**Status: in progress — tasks 1-6 done; tasks 8, 9 not started.** `stream: true` is supported end-to-end for both the dry-run backend and real vLLM: `POST /v1/chat/completions` returns real SSE (`curl -N` shows incremental chunks, verified against a live GPU) with OpenAI-style delta chunks.

Task 4 (client SDK streaming) is done: both `VGate.chat.stream(...)` and `AsyncVGate.chat.stream(...)` consume the server's SSE response and yield OpenAI-shaped `ChatCompletionChunk` objects (`chunk.choices[0].delta.content`), verified against a live dry-run server for both the sync and async paths. Mid-stream failures (a `data: {"error": ...}` event, or a dropped connection) raise immediately rather than being retried, since a retry after tokens have already been yielded would duplicate text. `chat.stream(...)` also supports `with`/`async with` so a caller that stops iterating early (`break`) still closes the underlying SSE connection deterministically, and a connection that ends without a `data: [DONE]` event raises `ServerError` instead of silently looking like a clean completion.

`VLLMBackend` now runs on `AsyncLLMEngine` instead of the offline `LLM()` class (task 6, done) — a single engine instance backs both `generate()` (the batch-shaped call `RequestBatcher` still uses) and `stream_generate()`, avoiding a second model load / GPU-memory-budget conflict. `generate()` bridges from its calling worker thread to the engine's owning event loop via `asyncio.run_coroutine_threadsafe`. A side effect worth noting: even the non-streaming path now submits each prompt as an independent `engine.generate()` call rather than one sealed `LLM.generate(list, ...)` call, so it already gets real continuous batching from vLLM's own scheduler — ahead of the full `RequestBatcher` redefinition in task 9. `SGLangBackend.stream_generate()` still raises `NotImplementedError` (task 8 not started).

Task 5 (streaming metrics) is done: `vgate_stream_ttft_seconds`, `vgate_stream_tpot_seconds`, `vgate_stream_duration_seconds`, `vgate_stream_tokens_total`, and `vgate_stream_requests_total{status="completed"|"error"|"cancelled"}` are separate series from the batcher's `vgate_ttft_seconds`/`vgate_tpot_seconds` — the two paths measure fundamentally different things (engine-reported metrics vs. gateway-side wall clock against SSE chunk arrival), so reusing one series with a `mode` label would make neither queryable on its own and would change what every existing caller/dashboard sees. TPOT is token-weighted (`decode_time / decode_tokens`, accumulated per delta via each delta's cumulative `num_tokens`), not chunk-averaged, since a single SSE delta can carry more than one token. Fixing this surfaced a real pre-existing bug, in two forms: the role chunk's `yield` originally sat outside the `try`, so a disconnect before any content arrived went uncaught and unrecorded; and `finally: yield "data: [DONE]\n\n"` was illegal once a disconnect delivered `GeneratorExit` while the generator was suspended at *any* yield — including the one inside the `except Exception` error-event handler, which a sibling `except GeneratorExit` clause can't catch — raising `RuntimeError: async generator ignored GeneratorExit`. Fixed by moving the role-chunk yield inside `try`, defaulting `status` to `"cancelled"`, and moving the final `yield "data: [DONE]\n\n"` entirely out of `finally` (metrics-only there now) to after the try/except, which is only reached when no exception is still propagating. `final_num_tokens` is also now updated *before* each content yield rather than after, so a disconnect immediately following a delivered chunk still counts that chunk's tokens. `benchmarks/bench_load.py --stream` sends `stream: true`, consumes SSE, reports client-observed TTFT percentiles, and gets its `tokens/sec` from diffing the server's own `vgate_stream_tokens_total` counter rather than counting client-side SSE content-delta events (which undercounts whenever a delta packs more than one token).

Known, intentional limitation: the streaming path in `main.py` (`_stream_chat_completion`) calls `engine.backend.stream_generate()` directly and bypasses `RequestBatcher` entirely — no cache lookup, no batch-level dedup, no admission control for streamed requests yet. That integration is task 9, not started.

Not yet done: the SGLang async backend path (task 8) and redefining `RequestBatcher` (task 9).

Priority: high.

Reason: streaming is a baseline LLM serving capability. TTFT only becomes meaningful to users when tokens can be delivered incrementally. True continuous batching should be handled by vLLM/SGLang engine schedulers because they can observe token scheduling, KV-cache state, memory pressure, and preemption decisions.

Tasks:

1. Support `stream: true`.
2. Implement SSE responses.
3. Add OpenAI-style delta chunks.
4. Add streaming support to the client SDK.
5. Add streaming metrics: TTFT, stream duration, tokens streamed.
6. Add an async streaming backend protocol.
7. Add a vLLM AsyncLLMEngine backend path.
8. Add an SGLang `async_generate(..., stream=True)` backend path.
9. Redefine `RequestBatcher` as in-flight deduplication, fan-out, admission control, and cache integration rather than GPU batch construction.

Recommended implementation:

- Add `stream: bool = False` to the request model.
- Add a backend streaming protocol with per-request async iterators.
- Implement token-by-token mock streaming in the dry-run backend first.
- For vLLM, use AsyncLLMEngine or the equivalent async engine available in the installed vLLM version.
- For SGLang, use `async_generate(..., stream=True)` where available.
- Submit independent requests to the engine scheduler instead of sealing Python-side batches.
- Cache only complete finished outputs, not partial streaming chunks.
- On client disconnect or cancellation, abort the backend request when the engine supports it.

Acceptance criteria:

- `curl -N` shows incremental chunks.
- New requests can be submitted while earlier requests are still decoding.
- The Python gateway no longer seals GPU batches for the async engine path; independent requests are submitted to the vLLM/SGLang scheduler.
- Benchmarks compare legacy static micro-batching vs async engine path for TTFT, p95 latency, and tokens/sec.
- Python SDK supports:
  - `for chunk in client.chat.stream(...)`
  - `async for chunk in client.chat.stream(...)`
- Streaming and non-streaming paths both have tests.
- Cancellation tests prove backend abort behavior or documented fallback behavior.

Expected outcome:

- V-Gate looks and behaves more like a real LLM serving gateway.
- The project can explain TTFT, perceived latency, SSE connection management, and the boundary between gateway-level deduplication and engine-level continuous batching.

---

### Phase 3: Backpressure And Reliability

Priority: high.

Reason: reliable infrastructure must define what happens under overload. Batching alone is not enough; the system must avoid unbounded memory growth and cascading failure.

Tasks:

1. Add maximum queue length.
2. Add queue timeout.
3. Handle request cancellation.
4. Add load shedding.
5. Define 429/503 behavior.
6. Improve graceful shutdown.
7. Add overload metrics.

Recommended configuration:

```yaml
reliability:
  max_queue_size: 1024
  queue_timeout_ms: 2000
  overload_status_code: 503
  shed_policy: "reject_new"
```

Key metrics:

- `vgate_queue_rejected_total`
- `vgate_queue_timeout_total`
- `vgate_queue_size`
- `vgate_overload_active`

Acceptance criteria:

- Under high concurrency, the service does not grow memory without bound.
- When the queue limit is exceeded, the service returns stable 503 or 429 responses.
- Cancellation does not leave unresolved futures behind.
- Shutdown processes or rejects pending requests deterministically.

Expected outcome:

- The system moves from "can run" to "knows how to fail."
- Tail latency and overload behavior become explainable with metrics.

---

### Phase 4: Multi-Worker Serving

Priority: highest for distributed serving.

Reason: distributed inference serving depends on worker management, routing, health checks, and failure isolation.

Tasks:

1. Split gateway and worker responsibilities.
2. Add worker HTTP/gRPC API.
3. Maintain a worker registry in the gateway.
4. Add worker health checks.
5. Add routing strategies:
   - round-robin
   - least-inflight
   - EWMA latency
6. Add worker circuit breakers.
7. Add worker recovery.
8. Add multi-worker benchmarks.
9. Add minimal gateway-to-worker authentication suitable for a local/private cluster.

Suggested directory structure:

```text
vgate/
  gateway/
    router.py
    registry.py
    circuit_breaker.py
  worker/
    app.py
    client.py
  routing/
    round_robin.py
    least_inflight.py
    ewma.py
```

Suggested APIs:

- `GET /worker/health`
- `GET /worker/stats`
- `POST /worker/generate`
- `POST /admin/workers/register`
- `GET /admin/workers`

Acceptance criteria:

- Local docker-compose can start 1 gateway + 2 workers.
- If one worker is killed under active load, the gateway detects it, routes around it, and no request is silently dropped (chaos-style failure test).
- When the worker recovers, it can receive traffic again.
- Benchmarks show throughput improvement with multiple workers.
- `/stats` shows inflight requests, latency, and state for each worker.
- Gateway-worker calls use a minimal authentication mechanism. mTLS and audit logging are deferred to governance/security hardening.

Expected outcome:

- V-Gate enters distributed inference serving territory.
- The architecture can explain routing, failure isolation, service discovery, and scaling strategy.

---

### Phase 5: Kubernetes Productionization

Priority: medium.

Reason: Kubernetes is the delivery layer. It becomes much more meaningful after the service itself supports multi-worker serving.

Tasks:

1. Split Kubernetes deployments into gateway and worker deployments.
2. Use GPU overlays for workers.
3. Use CPU deployment for the gateway.
4. Extend HPA signals beyond CPU to queue length / latency.
5. Add a Helm chart.
6. Add rollout / rollback documentation.
7. Add Prometheus alert rules.

Suggested artifacts:

- `helm/vgate/`
- `k8s/overlays/multi-worker/`
- `monitoring/alerts.yaml`
- `docs/runbook.md`

Acceptance criteria:

- `helm install vgate ./helm/vgate` can deploy the stack.
- Alert rules cover error rate, p95 latency, queue overload, and worker down.
- The runbook explains how to investigate high latency, worker crashes, and cache miss spikes.

Expected outcome:

- Stronger productionization evidence.
- The system becomes easier to operate, not just easier to run.

---

### Phase 5.5: Control Plane And Rollouts

Priority: medium-low, after the multi-worker and Kubernetes paths are stable.

Reason: model registry, canary rollout, live config reload, mTLS, and audit logging are valuable for production maturity, but they should not block the core distributed serving milestone.

Tasks:

1. Add a lightweight model/config version registry.
2. Support canary rollout to a subset of workers.
3. Support rollback without downtime.
4. Add audit logging for worker registration and config changes.
5. Add mTLS or equivalent strong gateway-to-worker authentication.

Acceptance criteria:

- A new model/config version can be rolled out to a subset of workers and rolled back.
- Config changes are audit-logged.
- Gateway-worker traffic is strongly authenticated.
- Runbook documents rollout, rollback, and failed canary recovery.

Expected outcome:

- V-Gate gains a clear control-plane story without delaying the core serving and routing work.

---

### Phase 6: Correctness And Evaluation Gate

Priority: medium.

Reason: AI infrastructure is not only about speed. It must also keep outputs trustworthy across backend changes, streaming modes, and cache behavior.

Tasks:

1. Add a golden prompt set.
2. Compare output structure and stability across backends.
3. Add regression evaluation.
4. Add schema correctness checks.
5. Check consistency between streaming and non-streaming outputs.
6. Check cache-hit result consistency.

Suggested directory:

```text
evals/
  golden_prompts.jsonl
  run_eval.py
  reports/
```

Acceptance criteria:

- One command generates an evaluation report.
- CI runs a lightweight evaluation.
- Backend adapter changes can catch output-format regressions.

Expected outcome:

- V-Gate distinguishes itself from generic backend serving projects.
- Correctness, regression detection, and release gates become part of the system.

---

### Phase 7: C++ Cache Or Low-Level Performance Module

Priority: medium-high, but after the main serving path.

Reason: C++/CUDA work is valuable only when it is attached to measured bottlenecks. Start with a C++ sharded cache because it has a clear boundary, is testable, and has lower risk than CUDA sampling.

Tasks:

1. Implement a C++ sharded LRU cache.
2. Expose Python bindings with pybind11.
3. Add a Python cache facade that chooses C++ when available and falls back to Python otherwise.
4. Add correctness tests.
5. Add microbenchmarks.
6. Compare Python cache p95/p99 under high concurrency against the C++ cache.

Suggested directory:

```text
csrc/
  cache/
    include/
    src/
    bindings/
    tests/
benchmarks/
  bench_cache_python.py
  bench_cache_cpp.py
```

Acceptance criteria:

- Python fallback works when the C++ extension is unavailable.
- C++ cache is automatically enabled when the extension is available.
- Benchmarks show reduced lock contention.
- Documentation explains shard count, locking strategy, eviction behavior, and memory overhead.

Expected outcome:

- The project gains a concrete low-level systems component.
- Performance-sensitive backend infrastructure requirements become easier to address with code and measurements.

---

### Phase 8: GPU / Inference Optimization Deep Work

Priority: low-to-medium, as a stretch phase.

Reason: CUDA and speculative decoding are difficult and only worth doing after the earlier roadmap has made the serving system credible. Otherwise they require high effort with unclear engineering payoff.

Possible directions:

1. Speculative decoding scheduler
2. Prefix cache / prompt cache
3. KV-cache-aware routing
4. CUDA sampling kernel
5. Deeper native vLLM/SGLang integration

Recommended first:

- Prefix-cache metrics
- KV-cache-aware routing design + prototype

Not recommended as the first item:

- A full custom CUDA top-p/top-k sampling implementation

Acceptance criteria:

- Benchmarks are clear.
- Correctness comparison exists.
- Profiling results exist.
- Bottlenecks and benefits are documented.

Expected outcome:

- The project moves toward deeper inference systems / performance engineering.
- This is not required for medium maturity, but it raises the ceiling significantly.

---

## 4. Suggested Timeline

This week-by-week plan is stale as a calendar: Weeks 1-3 are done, but the later weeks were never executed on that schedule. Read it as relative effort and dependency order, not as the project schedule.

### Week 1: Credibility Fixes

- Fix `cache.enabled`.
- Fix batching parameter semantics.
- Add timeout/cancellation tests.
- Add CI.
- Update README with mock embedding and roadmap status.

Deliverables:

- Stable tests
- Clear README
- Public single-node demo baseline

### Week 2: Benchmark Report

- Add load benchmark.
- Output p50/p95/p99.
- Measure cache hit rate, working-set size, and eviction pressure.
- Generate Markdown report.
- Link benchmark results from README.

Deliverables:

- `benchmarks/results/baseline.md`
- Quantified performance story
- Data and design decision for whether sqlite L2 cache is justified

### Week 3: Streaming

- Add `stream` to request model.
- Implement dry-run streaming.
- Implement SSE endpoint.
- Add async backend protocol.
- Add vLLM AsyncLLMEngine path.
- Add SGLang async streaming path.
- Add SDK streaming.
- Add streaming tests.

Deliverables:

- Core LLM serving capability
- Engine-native continuous batching path

### Week 4: Backpressure

- Add max queue size.
- Add queue timeout.
- Add overload response.
- Handle cancellation.
- Add overload metrics.

Deliverables:

- Reliability story
- p95/p99 and overload tradeoff analysis

### Weeks 5-6: Multi-Worker

- Split gateway / worker.
- Add worker registry.
- Add health checks.
- Add least-inflight routing.
- Add EWMA routing.
- Add circuit breakers.
- Add minimal gateway-to-worker authentication.
- Add docker-compose multi-worker setup.

Deliverables:

- Distributed serving demo
- Multi-worker benchmark

### Week 7: Kubernetes And Runbook

- Split gateway/worker Kubernetes deployments.
- Add Helm chart.
- Add alert rules.
- Add runbook.

Deliverables:

- Productionization evidence

### Week 8: C++ Cache

- Add C++ sharded LRU.
- Add pybind11 bindings.
- Add Python fallback.
- Add cache benchmark.

Deliverables:

- Low-level performance module

### Deferred: Control Plane And Security Hardening

- Add model/config registry, canary rollout, rollback, audit logging, and mTLS after the distributed serving path is stable.

Deliverables:

- Control-plane maturity without blocking the serving milestones

### Deferred: SQLite L2 Cache

- Implement only after the Phase 1.5 decision gate shows repeated requests beyond RAM cache capacity.
- Use sqlite3 WAL mode, write-through semantics, and executor-wrapped disk I/O.

Deliverables:

- Optional persistent local cache layer

---

## 5. Milestone Narrative

The descriptions below are target narratives for each completed milestone. They should be read as roadmap outcomes, not as current implementation status unless the corresponding phase is already complete.

### Single-Node Gateway

V-Gate is an OpenAI-style LLM serving gateway that supports vLLM/SGLang backend selection. It improves throughput with dynamic micro-batching and batch-level deduplication, reduces repeated work with an LRU cache, exposes Prometheus metrics, structured logs, and OpenTelemetry traces, and ships with API-key auth, rate limiting, Docker/Kubernetes manifests, and a Python SDK.

### Reliable LLM Serving

V-Gate adds systematic benchmark reporting, p50/p95/p99 latency tracking, TTFT, TPOT, and tokens/sec metrics. It supports streaming to reduce perceived latency, delegates true continuous batching to vLLM/SGLang async schedulers, and uses queue timeouts, max queue size, load shedding, and overload metrics to keep the system bounded under high concurrency.

### Distributed Inference Serving

V-Gate evolves into a gateway-worker architecture. The gateway maintains a worker registry, routes using health checks and EWMA/least-inflight latency signals, tracks inflight load per worker, removes unhealthy workers with circuit breakers, and verifies scaling behavior with 1-worker vs N-worker benchmarks.

### Control Plane And Rollouts

V-Gate adds model/config versioning, canary rollout, rollback, audit logging, and strong gateway-to-worker authentication after the core distributed serving path is stable.

---

## 6. Public Project Highlights

These highlight statements are milestone targets. Keep README and release notes tied to implemented behavior.

### Single-Node Gateway

- Target: an OpenAI-style LLM serving gateway with FastAPI, vLLM/SGLang backend adapters, dynamic micro-batching, LRU caching, API-key authentication, Prometheus metrics, OpenTelemetry tracing, Docker/Kubernetes deployment, and a sync/async Python SDK.

### Reliable LLM Serving

- Target: streaming chat completions, engine-native continuous batching, in-flight request deduplication, overload protection, and benchmark tooling measuring TTFT, TPOT, p95/p99 latency, tokens/sec, queue time, and cache hit-rate under concurrent load.

### Distributed Inference Serving

- Target: a distributed inference serving architecture with gateway-worker separation, worker registry, health checks, EWMA/least-inflight routing, circuit breakers, graceful degradation, multi-worker benchmarks, Kubernetes deployment, and production runbooks.

### Control Plane And Rollouts

- Target: model/config registry, canary rollout, rollback, audit logging, and strong gateway-to-worker authentication without blocking the core serving milestones.

### Low-Level Performance Module

- Target: a pybind11-backed C++ sharded LRU cache with Python fallback, correctness tests, and microbenchmarks demonstrating reduced lock contention and improved p95/p99 cache latency under high concurrency.

---

## 7. Non-Priorities For The Current Stage

These are useful eventually, but should not be prioritized before the serving path is reliable and measurable:

- Writing a CUDA sampling kernel first
- Building speculative decoding first
- Adding sqlite L2 cache before cache metrics show RAM eviction pressure and repeated requests beyond L1 capacity
- Creating polished dashboards before real benchmark data exists
- Adding Helm before the service supports multi-worker mode
- Adding a complex admin UI
- Expanding many OpenAI parameters before streaming/backpressure is complete

Reason: these tasks are expensive and do not solve the current system's most important gaps. Medium-maturity AI infrastructure is more about boundaries, failure handling, scalability, and measurement than isolated advanced components.

---

## 8. Final Direction

If this roadmap is followed:

- Phase 0-1 produce a credible single-node AI gateway.
- Phase 1.5 produces a measured sqlite L2 cache decision, not necessarily an implementation.
- Phase 2-3 produce reliable LLM serving behavior with streaming and engine-native continuous batching.
- Phase 4-6 produce distributed inference serving with production signals.
- Phase 5.5 optionally adds control-plane rollout and security hardening after the core serving path is stable.
- Phase 7 adds a concrete systems/performance module.
- Phase 8 moves toward deeper inference systems performance work.

The optimal route is not to start with the hardest component. It is:

1. Make current behavior semantically correct.
2. Prove improvements with benchmark data.
3. Add LLM serving essentials such as streaming and engine-native continuous batching.
4. Add overload protection and reliability controls.
5. Add multi-worker distributed serving.
6. Add Kubernetes productionization and runbooks.
7. Add optional control-plane/security hardening after distributed serving is stable.
8. Add optional sqlite L2 cache only when cache metrics justify it.
9. Add C++/CUDA lower-level performance work after bottlenecks are measured.

This sequencing lets every phase improve the project independently while keeping the architecture story coherent from single-node gateway to distributed inference serving.
