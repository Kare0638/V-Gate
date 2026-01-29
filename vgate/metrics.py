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
    "Number of requests waiting in the batch queue"
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


def init_app_info(version: str = "0.3.0", model: str = "unknown"):
    """Initialize application info metric."""
    APP_INFO.info({
        "version": version,
        "model": model
    })
