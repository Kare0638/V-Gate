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

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import json
import time
import uuid

# Import V-Gate modules
from vgate.config import get_config
from vgate.engine import VGateEngine
from vgate.batcher import RequestBatcher
from vgate.logging_config import setup_logging, get_logger
from vgate.metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, REQUEST_IN_PROGRESS,
    TOKENS_GENERATED, INFERENCE_ERRORS,
    STREAM_TTFT, STREAM_TPOT, STREAM_DURATION, STREAM_TOKENS, STREAM_REQUESTS,
    init_app_info
)
from vgate.security import SecurityMiddleware
from vgate.tracing import init_tracing, shutdown_tracing, get_current_trace_id
from vgate.health_checker import WorkerHealthChecker
from vgate.worker_registry import NoHealthyWorkersError
from vgate import worker_api

# Prometheus client for metrics endpoint
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Load configuration
config = get_config()

# Setup logging from config
logger = setup_logging(level=config.logging.level, json_format=config.logging.json_format)
app_logger = get_logger("vgate.app")

# Version from config
APP_VERSION = config.version

# Role decides which half of the system this process runs: a gateway serves the
# public API and owns batching/caching, a worker only runs inference.
IS_WORKER = config.role == "worker"

# Lazy initialization - will be set in lifespan
engine = None
batcher = None
health_checker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup/shutdown."""
    global engine, batcher, health_checker

    # Initialize tracing before other components
    init_tracing(config)

    # Instrument FastAPI with OpenTelemetry if tracing is enabled
    if config.tracing.enabled:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)

    # Initialize the VGateEngine with config (inside lifespan for multiprocessing safety)
    engine = VGateEngine()

    # Initialize app info for Prometheus
    init_app_info(version=APP_VERSION, model=config.model.model_id)

    if IS_WORKER:
        # A worker only runs inference. Batching, caching, and admission
        # control stay on the gateway; running them here too would batch and
        # cache every request twice.
        worker_api.set_engine(engine)
        app_logger.info(
            "V-Gate worker started",
            extra={"extra_data": {
                "version": APP_VERSION,
                "model": config.model.model_id,
                "engine_type": config.model.engine_type,
                "security_enabled": config.security.enabled,
            }}
        )
        yield
        engine.backend.shutdown()
        shutdown_tracing()
        app_logger.info("V-Gate worker stopped")
        return

    # Initialize the RequestBatcher (uses config defaults)
    batcher = RequestBatcher(engine=engine)

    # Probe workers in the background so recovery is noticed on an idle
    # gateway too, not only when a request happens to retry a dead worker.
    if engine.is_remote:
        health_checker = WorkerHealthChecker(
            registry=engine.backend.registry,
            interval_seconds=config.worker.health_check_interval_seconds,
            timeout_seconds=config.worker.health_check_timeout_seconds,
            api_key=config.worker.api_key,
        )
        await health_checker.start()

    # Startup: start the batcher
    await batcher.start()
    app_logger.info(
        "V-Gate started",
        extra={"extra_data": {
            "version": APP_VERSION,
            "model": config.model.model_id,
            "engine_type": config.model.engine_type,
            "inference": "remote" if engine.is_remote else "in-process",
            "worker_endpoints": config.worker.endpoints,
            "batch_config": {
                "max_batch_size": config.batch.max_batch_size,
                "max_wait_time_ms": config.batch.max_wait_time_ms,
            },
            "cache_config": {
                "enabled": config.cache.enabled,
                "maxsize": config.cache.maxsize,
            },
            "security_config": {
                "enabled": config.security.enabled,
                "api_keys_count": len(config.security.api_keys),
                "rate_limiting_enabled": config.security.rate_limiting.enabled,
            }
        }}
    )
    yield
    # Shutdown: stop the batcher
    await batcher.stop()
    if health_checker is not None:
        await health_checker.stop()
    # getattr: shutdown must not raise on an unexpected engine object, or the
    # real reason the process is going down gets masked by an AttributeError.
    if getattr(engine, "is_remote", False):
        engine.backend.shutdown()
    shutdown_tracing()
    app_logger.info("V-Gate stopped")


app = FastAPI(
    title="V-Gate LLM Inference Worker" if IS_WORKER else "V-Gate LLM Inference Gateway",
    description=(
        "Internal inference worker for a V-Gate gateway. Not a public API surface."
        if IS_WORKER else
        "An LLM inference gateway providing an OpenAI-shaped Chat Completions subset "
        "with streaming, admission control, dynamic micro-batching, result caching, "
        "and observability."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

# Mounted only in the worker role, so a gateway never exposes an unbatched,
# uncached inference endpoint. /internal/generate is deliberately not in
# security.exempt_paths: with security enabled it requires the same bearer key
# as any other route, which is what authenticates the gateway to the worker.
if IS_WORKER:
    app.include_router(worker_api.router)

# Add security middleware (runs before observability middleware)
# Note: Middlewares are executed in LIFO order, so add security first
app.add_middleware(SecurityMiddleware, config=config.security)


# Request/Response middleware for logging and metrics
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Middleware for request logging and Prometheus metrics."""
    # Use OTel trace_id when available, fallback to 8-char UUID
    trace_id = get_current_trace_id()
    request_id = trace_id if trace_id else str(uuid.uuid4())[:8]

    start_time = time.perf_counter()

    # Track in-progress requests
    endpoint = request.url.path
    REQUEST_IN_PROGRESS.labels(endpoint=endpoint).inc()

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        raise
    finally:
        # Calculate latency
        latency = time.perf_counter() - start_time

        # Build exemplar for metric-trace correlation
        exemplar = {"trace_id": trace_id} if trace_id else None

        # Update Prometheus metrics
        REQUEST_COUNT.labels(
            endpoint=endpoint,
            method=request.method,
            status=str(status_code)
        ).inc(exemplar=exemplar)
        REQUEST_LATENCY.labels(
            endpoint=endpoint,
            method=request.method
        ).observe(latency, exemplar=exemplar)
        REQUEST_IN_PROGRESS.labels(endpoint=endpoint).dec()

        # Log request completion (skip /metrics and /health for less noise)
        if endpoint not in ["/metrics", "/health", "/metrics/prometheus"]:
            app_logger.info(
                "Request completed",
                extra={"extra_data": {
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "method": request.method,
                    "path": endpoint,
                    "status": status_code,
                    "latency_ms": round(latency * 1000, 2)
                }}
            )

    # Add request ID header
    response.headers["X-Request-ID"] = request_id
    return response


# Request models for OpenAI-like API
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    stream: bool = False


class EmbeddingRequest(BaseModel):
    model: str
    input: str


# Helper function to convert messages to a prompt string
def messages_to_prompt(messages: list[ChatMessage]) -> str:
    prompt_parts = [f"{m.role.capitalize()}: {m.content}" for m in messages]
    return "\n".join(prompt_parts) + "\nAssistant:"


def gateway_only() -> None:
    """
    Reject client-facing routes in the worker role.

    A worker has no batcher and no cache, so these handlers would fail on a
    None batcher. Returning 404 states the intent: this surface does not exist
    on a worker, rather than existing and being broken.
    """
    if IS_WORKER:
        raise HTTPException(
            status_code=404, detail="Not available in worker role; call the gateway instead"
        )


@app.get("/health", summary="Health Check")
async def health_check():
    """
    Returns the health status of the V-Gate service.

    Available in both roles: the gateway's worker health checks poll this.
    """
    return {"status": "ok", "version": APP_VERSION, "role": config.role}


async def _stream_chat_completion(prompt: str, request: ChatCompletionRequest):
    """
    SSE generator for streaming chat completions.

    NOTE: this bypasses RequestBatcher entirely — it calls
    engine.backend.stream_generate() directly, so streaming requests get no
    cache lookup, no batch-level deduplication, and no admission control yet.
    RequestBatcher is still GPU-batch-oriented and can't hand a partial
    result back mid-generation. Folding streaming into batcher-provided
    dedup/admission is ROADMAP.md Phase 2 task 9, not done here.
    """
    completion_id = "chatcmpl-" + str(uuid.uuid4())[:8]
    created = int(time.time())
    started_at = time.monotonic()

    def _chunk(delta: dict, finish_reason: str = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    # Token-weighted TPOT bookkeeping: a single delta can carry more than one
    # token (see vllm_backend.py's stream_generate), so averaging the time
    # between chunks would measure time-per-chunk, not time-per-token.
    ttft_recorded = False
    prev_num_tokens = 0
    prev_delta_time = started_at
    decode_time = 0.0
    decode_tokens = 0
    final_num_tokens = 0
    # Pessimistic default: covers a disconnect before/during the very first
    # (role) yield, which is now inside this try block so it's caught below
    # instead of propagating uncaught with nothing recorded.
    status = "cancelled"

    try:
        yield _chunk({"role": "assistant"})
        sampling_params = engine.backend.create_sampling_params(
            temperature=request.temperature, top_p=request.top_p, max_tokens=request.max_tokens
        )
        async for piece in engine.backend.stream_generate(prompt, sampling_params):
            delta = piece.get("delta")
            num_tokens = piece.get("num_tokens", prev_num_tokens)
            now = time.monotonic()
            # Set before the yield below: if a disconnect interrupts that
            # yield, the chunk was still sent to the client, so its tokens
            # must already be reflected in final_num_tokens by then.
            final_num_tokens = num_tokens
            if delta:
                if not ttft_recorded:
                    STREAM_TTFT.observe(now - started_at)
                    ttft_recorded = True
                else:
                    token_increment = num_tokens - prev_num_tokens
                    interval = now - prev_delta_time
                    if token_increment > 0:
                        decode_time += interval
                        decode_tokens += token_increment
                prev_num_tokens = num_tokens
                prev_delta_time = now
                yield _chunk({"content": delta})
        yield _chunk({}, finish_reason="stop")
        status = "completed"
        if decode_tokens > 0:
            STREAM_TPOT.observe(decode_time / decode_tokens)
        STREAM_DURATION.observe(time.monotonic() - started_at)
        app_logger.info(
            "Streamed chat completion",
            extra={"extra_data": {"completion_id": completion_id, "tokens": final_num_tokens}}
        )
    except (GeneratorExit, asyncio.CancelledError):
        # Client disconnected or the request was cancelled — not a backend
        # failure, so it isn't counted as an inference error.
        status = "cancelled"
        raise
    except Exception as e:
        status = "error"
        INFERENCE_ERRORS.labels(error_type=type(e).__name__).inc()
        # The 200 OK + SSE headers are already flushed by this point, so an
        # HTTP error status is no longer possible; surface the failure as an
        # SSE error event instead (mirrors how OpenAI's API reports
        # mid-stream failures). If the client disconnects while this very
        # yield is in flight, GeneratorExit propagates past this except
        # clause (it doesn't match Exception) straight to finally below —
        # status stays "error" rather than "cancelled", an acceptable
        # tie-break since a real backend failure already happened first.
        app_logger.error(
            "Streaming chat completion error",
            extra={"extra_data": {
                "completion_id": completion_id,
                "error": str(e),
                "error_type": type(e).__name__
            }}
        )
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': type(e).__name__}})}\n\n"
    finally:
        # Never yield here: yielding while a GeneratorExit raised by any of
        # the yields above is still propagating (i.e. status == "cancelled",
        # or a disconnect during the error-event yield) raises "RuntimeError:
        # async generator ignored GeneratorExit". Metrics only.
        STREAM_REQUESTS.labels(status=status).inc()
        if final_num_tokens > 0:
            STREAM_TOKENS.inc(final_num_tokens)
            TOKENS_GENERATED.inc(final_num_tokens)

    # Only reached when the try block above completed without an exception
    # still propagating (i.e. not on the cancelled path, and not if a
    # disconnect interrupted the error-event yield above).
    if status != "cancelled":
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", summary="Create Chat Completion",
          dependencies=[Depends(gateway_only)])
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Generates a chat completion response from the specified model.
    Non-streaming requests are automatically batched for improved throughput.
    Streaming requests (stream=true) bypass the batcher; see
    _stream_chat_completion for why.
    """
    if request.stream:
        # Reject before any SSE bytes are written. Once the 200 and stream
        # headers are flushed the only way to report this is an in-band error
        # event, which reads to a client as "the request succeeded and then
        # something went wrong mid-stream" — misleading when the backend never
        # supported streaming at all.
        if not getattr(engine.backend, "supports_streaming", True):
            raise HTTPException(
                status_code=501,
                detail=(
                    "Streaming is not supported when the gateway forwards to remote "
                    "workers. Send stream=false, or run a gateway with an in-process backend."
                ),
            )
        prompt = messages_to_prompt(request.messages)
        return StreamingResponse(
            _stream_chat_completion(prompt, request),
            media_type="text/event-stream",
        )

    try:
        # Convert messages to a single prompt string for the engine
        prompt = messages_to_prompt(request.messages)

        # Submit to batcher for batched processing
        response = await batcher.submit(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )

        # Adapt engine's response to OpenAI-like format
        return {
            "id": "chatcmpl-" + str(uuid.uuid4())[:8],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response["text"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": response.get("prompt_tokens", 0),
                "completion_tokens": response.get("total_tokens", 0),
                "total_tokens": response.get("total_tokens", 0) + response.get("prompt_tokens", 0),
            },
        }
    except NoHealthyWorkersError as e:
        # Every worker is out of rotation. This is capacity unavailable, not a
        # bad request or a gateway bug, so it is a 503 with Retry-After rather
        # than a 500 — the distinction is what tells a client to back off and
        # retry instead of treating the request as poison.
        app_logger.error(
            "No healthy workers",
            extra={"extra_data": {"error": str(e)}}
        )
        raise HTTPException(
            status_code=503,
            detail=str(e),
            headers={"Retry-After": "5"},
        )
    except Exception as e:
        app_logger.error(
            "Chat completion error",
            extra={"extra_data": {"error": str(e), "error_type": type(e).__name__}}
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings", summary="Create Embeddings",
          dependencies=[Depends(gateway_only)])
async def create_embeddings(request: EmbeddingRequest):
    """
    Generates embeddings for the given input text from the specified model.
    """
    try:
        response = engine.embeddings(request.input)

        # Adapt engine's response to OpenAI-like format
        return {
            "object": "list",
            "data": response["data"],
            "model": request.model,
            "usage": response["usage"],
        }
    except Exception as e:
        app_logger.error(
            "Embeddings error",
            extra={"extra_data": {"error": str(e), "error_type": type(e).__name__}}
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", summary="Prometheus Metrics")
async def prometheus_metrics(request: Request):
    """
    Returns metrics in Prometheus or OpenMetrics format.
    Supports content negotiation: use Accept: application/openmetrics-text
    for OpenMetrics format (required for exemplar export).
    """
    accept = request.headers.get("accept", "")
    if "application/openmetrics-text" in accept:
        from prometheus_client.openmetrics.exposition import generate_latest as om_generate_latest
        return Response(
            content=om_generate_latest(),
            media_type="application/openmetrics-text; version=1.0.0; charset=utf-8",
        )
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/stats", summary="JSON Statistics",
         dependencies=[Depends(gateway_only)])
async def get_stats():
    """
    Returns metrics about the request batching system and cache in JSON format.
    Useful for monitoring and debugging.
    """
    metrics = batcher.get_metrics()
    stats = {
        "batcher": {
            "total_requests": metrics["total_requests"],
            "total_batches": metrics["total_batches"],
            "average_batch_size": metrics["average_batch_size"],
            "pending_requests": metrics["pending_requests"],
            "total_deduplicated": metrics["total_deduplicated"],
            "avg_queue_time_s": metrics["avg_queue_time_s"],
            "avg_ttft_s": metrics["avg_ttft_s"],
            "avg_tpot_s": metrics["avg_tpot_s"],
        },
        "cache": metrics["cache"],
        "config": {
            "batch": {
                "max_batch_size": config.batch.max_batch_size,
                "max_wait_time_ms": config.batch.max_wait_time_ms,
            },
            "cache": {
                "enabled": config.cache.enabled,
                "maxsize": config.cache.maxsize,
            },
            "logging": {
                "level": config.logging.level,
                "json_format": config.logging.json_format,
            },
            "security": {
                "enabled": config.security.enabled,
                "rate_limiting_enabled": config.security.rate_limiting.enabled,
                "exempt_paths": config.security.exempt_paths,
            },
        },
        "version": APP_VERSION,
    }

    # Only present when inference is remote; an in-process gateway has no
    # workers to report on.
    if engine is not None and engine.is_remote:
        stats["workers"] = engine.backend.registry.snapshot()

    return stats


class BenchmarkRequest(BaseModel):
    prompts: list[str] = []
    max_tokens: int = 128
    rounds: int = 3


@app.post("/v1/benchmark", summary="Run Inline Benchmark",
          dependencies=[Depends(gateway_only)])
async def run_benchmark(request: BenchmarkRequest):
    """
    Run a quick inline benchmark through the full pipeline (batcher + cache + engine).
    Returns latency stats and throughput for the current engine_type.
    """
    bench_config = config.benchmark
    prompts = request.prompts or bench_config.prompts
    max_tokens = request.max_tokens
    rounds = request.rounds

    stats_before = batcher.get_metrics()

    latencies: list[float] = []
    token_counts: list[int] = []
    ttfts: list[float] = []
    tpots: list[float] = []

    for _ in range(rounds):
        round_start = time.perf_counter()
        tasks = [
            batcher.submit(prompt, max_tokens=max_tokens)
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks)
        round_latency = time.perf_counter() - round_start
        latencies.append(round_latency)
        token_counts.append(sum(r.get("total_tokens", 0) for r in results))
        ttfts.extend(r["ttft"] for r in results if r.get("ttft", 0) > 0)
        tpots.extend(r["tpot"] for r in results if r.get("tpot", 0) > 0)

    stats_after = batcher.get_metrics()

    total_tokens = sum(token_counts)
    total_time = sum(latencies)
    sorted_lat = sorted(latencies)

    def _percentile(data: list[float], pct: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        idx = min(int(len(s) * pct / 100), len(s) - 1)
        return s[idx]

    new_requests = stats_after["total_requests"] - stats_before["total_requests"]
    new_batches = stats_after["total_batches"] - stats_before["total_batches"]
    new_dedup = stats_after["total_deduplicated"] - stats_before["total_deduplicated"]
    new_cache_hits = stats_after["cache"]["hits"] - stats_before["cache"]["hits"]
    new_cache_misses = stats_after["cache"]["misses"] - stats_before["cache"]["misses"]

    return {
        "engine_type": config.model.engine_type,
        "rounds": rounds,
        "prompts_per_round": len(prompts),
        "latency": {
            "mean_s": round(total_time / rounds, 4),
            "p50_s": round(sorted_lat[len(sorted_lat) // 2], 4),
            "p95_s": round(sorted_lat[int(len(sorted_lat) * 0.95)], 4),
            "total_s": round(total_time, 4),
        },
        "ttft": {
            "mean_s": round(sum(ttfts) / len(ttfts), 4) if ttfts else 0,
            "p50_s": round(_percentile(ttfts, 50), 4),
            "p95_s": round(_percentile(ttfts, 95), 4),
        },
        "tpot": {
            "mean_s": round(sum(tpots) / len(tpots), 4) if tpots else 0,
            "p50_s": round(_percentile(tpots, 50), 4),
            "p95_s": round(_percentile(tpots, 95), 4),
        },
        "batching": {
            "requests": new_requests,
            "batches": new_batches,
            "average_batch_size": round(new_requests / new_batches, 2) if new_batches > 0 else 0,
            "deduplicated": new_dedup,
        },
        "cache": {
            "hits": new_cache_hits,
            "misses": new_cache_misses,
            "hit_rate": (
                round(new_cache_hits / (new_cache_hits + new_cache_misses), 4)
                if (new_cache_hits + new_cache_misses) > 0 else 0
            ),
        },
        "throughput": {
            "total_tokens": total_tokens,
            "tokens_per_second": round(total_tokens / total_time, 2) if total_time > 0 else 0,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.server.host, port=config.server.port)
