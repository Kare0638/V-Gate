# Copyright 2025 the V-Gate authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prometheus metrics definitions for V-Gate.

Provides counters, histograms, and gauges for monitoring system performance.
"""
from prometheus_client import Counter, Histogram, Gauge, Info, REGISTRY

# =============================================================================
# Helper to avoid duplicate registration in tests
# =============================================================================

def _safe_metric(metric_class, name, documentation, labelnames=None, buckets=None):
    """Create metric, handling duplicate registration gracefully."""
    # Check if already registered by looking at names in registry
    # Counter 'foo' creates 'foo_total', 'foo_created', etc.
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]

    # For counters, also check for _total suffix
    if f"{name}_total" in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[f"{name}_total"]

    # Create new metric
    kwargs = {}
    if labelnames:
        kwargs["labelnames"] = labelnames
    if buckets is not None and metric_class == Histogram:
        kwargs["buckets"] = buckets

    return metric_class(name, documentation, **kwargs)


# =============================================================================
# Application Info
# =============================================================================

APP_INFO = _safe_metric(Info, "vgate", "V-Gate application information")

# =============================================================================
# Request Metrics
# =============================================================================

REQUEST_COUNT = _safe_metric(
    Counter,
    "vgate_requests_total",
    "Total number of requests received",
    labelnames=["endpoint", "method", "status"]
)

REQUEST_LATENCY = _safe_metric(
    Histogram,
    "vgate_request_latency_seconds",
    "Request latency in seconds",
    labelnames=["endpoint", "method"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

REQUEST_IN_PROGRESS = _safe_metric(
    Gauge,
    "vgate_requests_in_progress",
    "Number of requests currently being processed",
    labelnames=["endpoint"]
)

# =============================================================================
# Batch Processing Metrics
# =============================================================================

BATCH_SIZE = _safe_metric(
    Histogram,
    "vgate_batch_size",
    "Number of requests per batch",
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32]
)

BATCH_PROCESSING_TIME = _safe_metric(
    Histogram,
    "vgate_batch_processing_seconds",
    "Batch processing time in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

BATCH_QUEUE_TIME = _safe_metric(
    Histogram,
    "vgate_batch_queue_time_seconds",
    "Time requests spend waiting in queue",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

PENDING_REQUESTS = _safe_metric(
    Gauge,
    "vgate_pending_requests",
    "Number of requests waiting for an admission permit"
)

INFLIGHT_INFERENCES = _safe_metric(
    Gauge,
    "vgate_inflight_inferences",
    "Number of inferences currently executing on a backend"
)

TOTAL_BATCHES = _safe_metric(
    Counter,
    "vgate_batches_total",
    "Total number of batches processed"
)

# =============================================================================
# Inference Metrics
# =============================================================================

TTFT = _safe_metric(
    Histogram,
    "vgate_ttft_seconds",
    "Time to first token in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

TPOT = _safe_metric(
    Histogram,
    "vgate_tpot_seconds",
    "Time per output token in seconds",
    buckets=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1]
)

TOKENS_GENERATED = _safe_metric(
    Counter,
    "vgate_tokens_generated_total",
    "Total number of tokens generated"
)

INFERENCE_ERRORS = _safe_metric(
    Counter,
    "vgate_inference_errors_total",
    "Total number of inference errors",
    labelnames=["error_type"]
)

UNIQUE_PROMPTS_PER_BATCH = _safe_metric(
    Histogram,
    "vgate_unique_prompts_per_batch",
    "Number of unique prompts per batch (after deduplication)",
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 12, 16]
)

# =============================================================================
# Streaming Metrics
#
# Kept separate from TTFT/TPOT above rather than reusing them with a "mode"
# label: the two paths measure fundamentally different things (non-streaming
# TTFT/TPOT come from vLLM engine-reported metrics; streaming TTFT/TPOT are
# gateway-side wall-clock measurements against SSE chunk arrival), so mixing
# them into one series would make neither queryable on its own and would
# change the existing series' meaning for every current caller/dashboard.
# =============================================================================

STREAM_TTFT = _safe_metric(
    Histogram,
    "vgate_stream_ttft_seconds",
    "Time to first token for streaming chat completions (gateway-side wall clock), in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

STREAM_TPOT = _safe_metric(
    Histogram,
    "vgate_stream_tpot_seconds",
    "Token-weighted time per output token for streaming chat completions, in seconds",
    buckets=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1]
)

STREAM_DURATION = _safe_metric(
    Histogram,
    "vgate_stream_duration_seconds",
    "Duration of successfully completed streaming chat completions (start to [DONE]), in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

STREAM_TOKENS = _safe_metric(
    Counter,
    "vgate_stream_tokens_total",
    "Total number of tokens generated by streaming chat completions"
)

STREAM_REQUESTS = _safe_metric(
    Counter,
    "vgate_stream_requests_total",
    "Total number of streaming chat completion requests by outcome",
    labelnames=["status"]
)

# =============================================================================
# Cache Metrics
# =============================================================================

CACHE_HITS = _safe_metric(
    Counter,
    "vgate_cache_hits_total",
    "Total number of cache hits"
)

CACHE_MISSES = _safe_metric(
    Counter,
    "vgate_cache_misses_total",
    "Total number of cache misses"
)

CACHE_SIZE = _safe_metric(
    Gauge,
    "vgate_cache_size",
    "Current number of entries in the cache"
)

CACHE_EVICTIONS = _safe_metric(
    Counter,
    "vgate_cache_evictions_total",
    "Total number of cache evictions"
)

# =============================================================================
# Deduplication Metrics
# =============================================================================

DEDUPLICATED_REQUESTS = _safe_metric(
    Counter,
    "vgate_deduplicated_requests_total",
    "Total number of requests deduplicated within batches"
)

DEDUP_RATIO = _safe_metric(
    Gauge,
    "vgate_dedup_ratio",
    "Current deduplication ratio (deduplicated / total in last batch)"
)


# =============================================================================
# Worker Routing Metrics
# =============================================================================
#
# `worker` is the configured endpoint, which is bounded by config and does not
# grow with traffic. No request-scoped value (prompt, request id, error text)
# is used as a label.

WORKER_HEALTHY = _safe_metric(
    Gauge,
    "vgate_worker_healthy",
    "Whether a worker is currently in rotation (1) or removed (0)",
    labelnames=["worker"]
)

WORKER_STATE_CHANGES = _safe_metric(
    Counter,
    "vgate_worker_state_changes_total",
    "Worker health transitions",
    labelnames=["worker", "transition"]
)

WORKER_REQUESTS = _safe_metric(
    Counter,
    "vgate_worker_requests_total",
    "Requests dispatched to a worker by outcome",
    labelnames=["worker", "outcome"]
)

WORKER_RETRIES = _safe_metric(
    Counter,
    "vgate_worker_retries_total",
    "Requests retried on another worker after a connection failure",
    labelnames=["worker"]
)

WORKER_LATENCY = _safe_metric(
    Histogram,
    "vgate_worker_latency_seconds",
    "Time spent in a worker generate call",
    labelnames=["worker"]
)


def init_app_info(version: str = "0.3.0", model: str = "unknown"):
    """Initialize application info metric."""
    APP_INFO.info({
        "version": version,
        "model": model
    })
