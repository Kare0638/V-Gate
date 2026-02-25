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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import time
import uuid

# Import V-Gate modules
from vgate.config import get_config
from vgate.engine import VGateEngine
from vgate.batcher import RequestBatcher
from vgate.logging_config import setup_logging, get_logger
from vgate.metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, REQUEST_IN_PROGRESS,
    init_app_info
)
from vgate.security import SecurityMiddleware
from vgate.tracing import init_tracing, shutdown_tracing, get_current_trace_id

# Prometheus client for metrics endpoint
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Load configuration
config = get_config()

# Setup logging from config
logger = setup_logging(level=config.logging.level, json_format=config.logging.json_format)
app_logger = get_logger("vgate.app")

# Version from config
APP_VERSION = config.version

# Lazy initialization - will be set in lifespan
engine = None
batcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup/shutdown."""
    global engine, batcher

    # Initialize tracing before other components
    init_tracing(config)

    # Instrument FastAPI with OpenTelemetry if tracing is enabled
    if config.tracing.enabled:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)

    # Initialize the VGateEngine with config (inside lifespan for multiprocessing safety)
    engine = VGateEngine()

    # Initialize the RequestBatcher (uses config defaults)
    batcher = RequestBatcher(engine=engine)

    # Initialize app info for Prometheus
    init_app_info(version=APP_VERSION, model=config.model.model_id)

    # Startup: start the batcher
    await batcher.start()
    app_logger.info(
        "V-Gate started",
        extra={"extra_data": {
            "version": APP_VERSION,
            "model": config.model.model_id,
            "engine_type": config.model.engine_type,
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
    shutdown_tracing()
    app_logger.info("V-Gate stopped")


app = FastAPI(
    title="V-Gate AI Model Serving Gateway",
    description="A high-performance AI model serving gateway for various models.",
    version=APP_VERSION,
    lifespan=lifespan,
)

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
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256


class EmbeddingRequest(BaseModel):
    model: str
    input: str


# Helper function to convert messages to a prompt string
def messages_to_prompt(messages: list) -> str:
    prompt_parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        prompt_parts.append(f"{role.capitalize()}: {content}")
    return "\n".join(prompt_parts) + "\nAssistant:"


@app.get("/health", summary="Health Check")
async def health_check():
    """
    Returns the health status of the V-Gate service.
    """
    return {"status": "ok", "version": APP_VERSION}


@app.post("/v1/chat/completions", summary="Create Chat Completion")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Generates a chat completion response from the specified model.
    Requests are automatically batched for improved throughput.
    """
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
    except Exception as e:
        app_logger.error(
            "Chat completion error",
            extra={"extra_data": {"error": str(e), "error_type": type(e).__name__}}
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings", summary="Create Embeddings")
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


@app.get("/stats", summary="JSON Statistics")
async def get_stats():
    """
    Returns metrics about the request batching system and cache in JSON format.
    Useful for monitoring and debugging.
    """
    metrics = batcher.get_metrics()
    return {
        "batcher": {
            "total_requests": metrics["total_requests"],
            "total_batches": metrics["total_batches"],
            "average_batch_size": metrics["average_batch_size"],
            "pending_requests": metrics["pending_requests"],
            "total_deduplicated": metrics["total_deduplicated"],
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


class BenchmarkRequest(BaseModel):
    prompts: list[str] = []
    max_tokens: int = 128
    rounds: int = 3


@app.post("/v1/benchmark", summary="Run Inline Benchmark")
async def run_benchmark(request: BenchmarkRequest):
    """
    Run a quick inline benchmark through the full pipeline (batcher + cache + engine).
    Returns latency stats and throughput for the current engine_type.
    """
    bench_config = config.benchmark
    prompts = request.prompts or bench_config.prompts
    max_tokens = request.max_tokens
    rounds = request.rounds

    latencies: list[float] = []
    token_counts: list[int] = []

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

    total_tokens = sum(token_counts)
    total_time = sum(latencies)
    sorted_lat = sorted(latencies)

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
        "throughput": {
            "total_tokens": total_tokens,
            "tokens_per_second": round(total_tokens / total_time, 2) if total_time > 0 else 0,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.server.host, port=config.server.port)
