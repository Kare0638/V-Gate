# V-Gate v2 Architecture Proposal: Python Control Plane + C++ Data Plane

> **Status**: Design proposal. The C++/CUDA data plane, speculative decoding scheduler, and `csrc/` source tree described here are not implemented in the current V-Gate runtime unless explicitly added in a future phase.
>
> **Purpose**: explore a hybrid architecture that keeps Python as the control plane while moving selected hot-path components to C++/CUDA for lower latency, lower lock contention, and clearer performance boundaries.

---

## 1. Current Architecture

The current V-Gate runtime is a Python-first LLM gateway:

```text
HTTP request
  -> FastAPI
  -> SecurityMiddleware
  -> ObservabilityMiddleware
  -> RequestBatcher
  -> ResultCache
  -> vLLM or SGLang backend
  -> Response
```

Important current characteristics:

- `ResultCache` uses `OrderedDict` and `asyncio.Lock`.
- Request batching runs in Python and calls the backend through a synchronous inference path in a thread pool.
- Sampling is delegated to the backend implementation, for example vLLM `SamplingParams`.
- Memory allocation and cached values are ordinary Python objects.

Potential bottlenecks for future scale:

| Component | Current Implementation | Potential Bottleneck |
|---|---|---|
| Cache | `OrderedDict` + `asyncio.Lock` | Single-lock serialization, GIL contention, Python object overhead |
| Sampling | Backend-native black box | Limited customizability and limited kernel-level visibility |
| Decode loop | Standard autoregressive generation | One target-model forward pass per generated token |
| Memory management | Python heap objects | GC overhead and allocation churn under high request volume |

This proposal does not claim these bottlenecks dominate today. Each phase must be justified with benchmarks before implementation is considered complete.

---

## 2. Target Architecture

The proposed v2 architecture keeps the existing Python gateway as the control plane and adds optional native modules for data-plane hot paths.

```text
+--------------------------------------------------------------------+
|                    Python Control Plane                             |
|  FastAPI -> security -> metrics -> tracing -> batching -> routing    |
|                                                                    |
|      pybind11 FFI                         PyTorch C++ Extension     |
|          |                                           |              |
|          v                                           v              |
|  +----------------------+                +----------------------+   |
|  | C++ Data Plane       |                | CUDA Compute Plane    |   |
|  |                      |                |                      |   |
|  | Sharded LRU Cache    |                | Top-K / Top-P kernels |   |
|  | Memory Pool          |                | Sampling ops          |   |
|  | Native statistics    |                | Nsight profiling      |   |
|  +----------------------+                +----------------------+   |
|                                                                    |
|  Optional: speculative decoding scheduler                           |
|  Draft model -> candidate tokens -> target model verification        |
+--------------------------------------------------------------------+
```

Design goals:

- Keep the public API and Python integration stable.
- Introduce native modules behind facade interfaces.
- Preserve Python fallback paths when native extensions are unavailable.
- Require benchmark evidence before treating a native module as a performance win.
- Keep failure modes explicit and observable.

---

## 3. Proposed Directory Layout

```text
V-Gate/
  csrc/
    CMakeLists.txt
    cache/
      include/
        lru_cache.h
        memory_pool.h
        shard.h
      src/
        lru_cache.cpp
        memory_pool.cpp
        shard.cpp
      bindings/
        py_cache.cpp
      tests/
        test_lru_cache.cpp
        test_memory_pool.cpp
        bench_cache.cpp
    cuda/
      include/
        sampling.cuh
      kernels/
        topk_sampling.cu
        topp_sampling.cu
        softmax.cu
      torch_ext/
        sampling_ops.cpp
      tests/
        test_topk.py
        test_topp.py
    third_party/
      pybind11/
  vgate/
    cache.py
    cuda_ops.py
    speculative.py
  benchmarks/
    bench_cache_python.py
    bench_cache_cpp.py
    bench_sampling.py
    bench_speculative.py
    results/
  setup.py
  pyproject.toml
  docker/
    Dockerfile.v2
```

The layout is intentionally additive. The current Python runtime should continue to work without `csrc/`.

---

## 4. Phase 1: Native Sharded LRU Cache

### Motivation

The current cache is simple and correct, but all operations pass through a single Python lock and Python object management. A native cache can reduce contention and make cache behavior easier to benchmark at high concurrency.

### Design

Recommended design: sharded LRU, not a fully lock-free LRU.

| Option | Complexity | Correctness Risk | Design Value |
|---|---:|---:|---|
| Single-lock LRU | Low | Low | Too close to the current design |
| Sharded LRU | Medium | Medium-low | Good tradeoff between performance and correctness |
| Fully lock-free LRU | High | High | Risky; ABA and memory reclamation are difficult |

The proposed cache uses:

- 64 shards by default.
- `std::shared_mutex` per shard.
- `std::unordered_map` for lookup.
- `std::list` for LRU order.
- Atomic counters for hits, misses, evictions, and size.
- Optional memory pool for cache entries.

Example interface:

```cpp
class LRUCache {
public:
    explicit LRUCache(size_t total_capacity);

    std::optional<std::string> get(const std::string& key);
    void put(const std::string& key, const std::string& value);

    struct Stats {
        uint64_t hits;
        uint64_t misses;
        uint64_t evictions;
        size_t size;
        size_t capacity;
        double hit_rate;
    };

    Stats stats() const;
};
```

### Python Integration

Expose the native cache with pybind11:

- Release the GIL around native `get` and `put`.
- Serialize cached values as JSON strings or msgpack bytes.
- Keep the existing async Python cache API.
- Fall back to the current Python cache when the extension cannot be imported.

Python facade behavior:

```python
try:
    from vgate_cpp_core import NativeCache
except ImportError:
    NativeCache = None

class ResultCache:
    def __init__(self, config):
        self._native = NativeCache(config.maxsize) if NativeCache else None
```

### Validation

Required tests:

- insert / get / update / evict
- LRU order after access
- concurrent reads and writes
- capacity boundaries
- stats accuracy
- Python fallback behavior

Required benchmarks:

- single-thread read/write mix
- multi-thread read-heavy workload
- p50/p95/p99 operation latency
- throughput under increasing thread count
- RSS memory comparison

Target report format:

| Backend | Threads | P50 | P95 | P99 | Throughput | RSS |
|---|---:|---:|---:|---:|---:|---:|
| Python | 1 | measured | measured | measured | measured | measured |
| C++ | 1 | measured | measured | measured | measured | measured |
| Python | 16 | measured | measured | measured | measured | measured |
| C++ | 16 | measured | measured | measured | measured | measured |

---

## 5. Phase 2: CUDA Sampling Operators

### Motivation

Sampling usually happens after logits are produced. A custom sampling operator can be useful for learning and for measuring kernel overhead, memory bandwidth, and numerical behavior. This phase should only proceed after there is a clear benchmark case.

### Scope

Proposed operators:

- top-k sampling
- top-p nucleus sampling
- optional fused softmax + sampling

Potential integration:

- PyTorch C++ extension using `TORCH_LIBRARY`.
- Python wrapper in `vgate/cuda_ops.py`.
- Fallback to PyTorch native operations when CUDA extension is unavailable.

### Design Concerns

Key issues:

- numerical stability in softmax
- memory bandwidth for large vocabulary logits
- kernel launch overhead
- warp divergence during filtering and sampling
- reproducibility and RNG behavior
- compatibility with PyTorch and CUDA versions

### Validation

Correctness checks:

- compare top-k candidates against PyTorch implementation
- compare top-p mask behavior against PyTorch implementation
- verify output distribution statistically over repeated samples
- test extreme logits, low temperature, and small vocabulary cases

Performance checks:

- Nsight Compute report
- SM utilization
- memory throughput
- achieved occupancy
- L2 hit rate
- kernel launch overhead

This phase should publish measured results before claiming any speedup.

---

## 6. Phase 3: Speculative Decoding Scheduler

### Motivation

Autoregressive decoding usually requires one target-model forward pass per generated token. Speculative decoding uses a smaller draft model to propose multiple tokens and a larger target model to verify them. When the draft model is close enough to the target model, TPOT can improve.

### Proposed Flow

```text
prompt
  -> draft model proposes gamma tokens
  -> target model verifies the proposed tokens
  -> accepted tokens are emitted
  -> rejected token falls back to target distribution
  -> repeat
```

### Model Assumptions

The draft and target models should:

- share the same tokenizer
- have compatible chat templates
- fit within available memory
- be benchmarked across task types

Example candidate pair:

| Role | Model | Purpose |
|---|---|---|
| Draft | Qwen2.5-0.5B-Instruct | Fast candidate generation |
| Target | Qwen2.5-1.5B-Instruct-AWQ | Verification and final output |

### Metrics

Required metrics:

- acceptance rate
- speculative step latency
- target verification latency
- TPOT before and after speculative decoding
- rollback / rejection count
- memory use when both models are loaded

### Risk

Speculative decoding can regress performance if:

- draft acceptance rate is low
- draft model overhead is too high
- memory pressure increases paging or lowers batch size
- implementation changes output distribution incorrectly

The feature should be disabled by default and guarded by correctness tests.

---

## 7. Implementation Roadmap

Recommended dependency order:

```text
Phase 1: C++ Cache
  -> CMake / pybind11 setup
  -> memory pool prototype
  -> sharded LRU implementation
  -> Python facade
  -> correctness tests
  -> benchmark report

Phase 2: CUDA Sampling
  -> CUDA build setup
  -> top-k kernel
  -> top-p kernel
  -> PyTorch extension
  -> correctness tests
  -> Nsight report

Phase 3: Speculative Decoding
  -> draft model loading
  -> verify / accept / reject logic
  -> engine integration
  -> benchmark report
  -> correctness gate
```

Phase 1 is the safest first native module because it has:

- clear input/output behavior
- simpler correctness tests
- no dependency on GPU availability
- a straightforward Python fallback

CUDA sampling and speculative decoding should come later because they have a larger correctness surface and higher environment complexity.

---

## 8. Build And Environment Requirements

| Tool | Requirement | Purpose |
|---|---|---|
| GCC / Clang | GCC 9+ or Clang 10+ | C++17 compilation |
| CMake | 3.18+ | Native build system |
| Python | 3.10+ | Control plane |
| pybind11 | 2.11+ | Python/C++ bindings |
| CUDA Toolkit | 11.8+ for Phase 2 | CUDA compilation |
| PyTorch | 2.0+ with CUDA for Phase 2 | CUDA extension integration |
| Nsight Compute | Latest compatible version | GPU profiling |
| Google Test | Optional | C++ unit tests |

The current runtime must remain usable without these native dependencies.

---

## 9. Testing Strategy

Testing layers:

```text
E2E benchmarks
  -> speculative decoding and full serving path

Integration tests
  -> Python facade with native modules

Unit tests
  -> C++ cache
  -> memory pool
  -> CUDA operator correctness
```

Key tests:

| Test | Framework | Purpose |
|---|---|---|
| `test_lru_cache.cpp` | GTest | insert, lookup, eviction, concurrency |
| `test_memory_pool.cpp` | GTest | allocation, release, fragmentation, OOM |
| `test_cache_integration.py` | pytest | native cache preserves Python cache behavior |
| `test_topk.py` | pytest | CUDA top-k behavior matches reference |
| `test_topp.py` | pytest | CUDA top-p behavior matches reference |
| `test_speculative.py` | pytest | accept/reject logic and fallback correctness |
| `bench_cache.py` | benchmark | p99 latency and multi-thread throughput |
| `bench_sampling.py` | benchmark | CUDA vs PyTorch sampling latency |

---

## 10. Target Outcomes

The numbers below are target outcomes for validating the proposal. They must be replaced with measured benchmark results once each phase is implemented.

| Metric | V1 Python Baseline | V2 Target | Target Improvement |
|---|---:|---:|---:|
| Cache p99 latency | measured baseline | measured native result | lower p99 |
| Cache multi-thread throughput | measured baseline | measured native result | higher throughput |
| Cache memory overhead | measured baseline | measured native result | lower RSS |
| Sampling latency | measured baseline | measured native result | lower latency |
| TPOT with speculative decoding | measured baseline | measured result | lower TPOT |

Project summary target:

> V-Gate v2 explores a hybrid LLM serving gateway that keeps Python as the control plane while moving selected hot-path data-plane components to native modules, validated by correctness tests and benchmark reports.

---

## 11. Design Review Questions

Useful review questions:

- Why choose sharded locking instead of a fully lock-free LRU?
- How does the cache avoid or handle false sharing?
- What work happens while the Python GIL is released?
- How is memory ownership handled across Python and C++?
- How do CUDA sampling results match the reference implementation?
- What are the expected bottlenecks: launch overhead, memory bandwidth, or compute?
- How is RNG reproducibility handled in custom sampling?
- What happens when speculative decoding has a low acceptance rate?
- How is correctness validated across standard and speculative decoding?
- What is the fallback path when native modules are unavailable?

---

## 12. Risks And Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---:|---:|---|
| Native build fails on some platforms | Medium | Low | Python fallback, optional extension, CI matrix |
| CUDA version mismatch with PyTorch | Medium | Medium | Use PyTorch compatibility matrix, document supported versions |
| C++ cache has memory safety bugs | Low-medium | High | RAII, sanitizers, fuzzing-style tests, small public API |
| Native cache does not outperform Python in real workload | Medium | Medium | Treat benchmark as gate; keep Python fallback |
| Speculative decoding acceptance rate is low | Medium | Medium | Feature flag, dynamic gamma, benchmark per workload |
| Dual-model loading exceeds GPU memory | Medium | High | quantized draft model, CPU offload, feature disabled by default |
| CUDA sampling changes output distribution | Low-medium | High | statistical tests and reference comparison |

---

## 13. V1 To V2 Comparison

| Area | V1 | V2 Proposal |
|---|---|---|
| Architecture | Python gateway | Python control plane + optional native data plane |
| Cache | `OrderedDict` + `asyncio.Lock` | sharded native LRU + Python fallback |
| Sampling | backend-native sampling | optional custom CUDA sampling |
| Decoding | standard autoregressive decoding | optional speculative decoding |
| Build | Python dependencies | optional CMake + pybind11 + CUDA build |
| Tests | pytest | pytest + native tests + benchmarks |
| Performance evidence | basic benchmark tooling | benchmark-gated native modules |

The v2 proposal is intentionally incremental. Each native component should be independently optional, benchmarked, and reversible.
