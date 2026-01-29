from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import sys
import time
import uuid

# Add the vgate directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'vgate')))

# Import V-Gate modules
from engine import VGateEngine
from batcher import RequestBatcher
from cache import CacheConfig
from logging_config import setup_logging, get_logger, DEFAULT_LOG_LEVEL, DEFAULT_JSON_FORMAT
from metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, REQUEST_IN_PROGRESS,
    init_app_info
)

# Prometheus client for metrics endpoint
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Setup logging
logger = setup_logging(level=DEFAULT_LOG_LEVEL, json_format=DEFAULT_JSON_FORMAT)
app_logger = get_logger("vgate.app")

# Configuration
BATCH_CONFIG = {
    "max_batch_size": int(os.getenv("VGATE_BATCH_SIZE", "8")),
    "max_wait_time_ms": float(os.getenv("VGATE_BATCH_WAIT_MS", "50.0")),
}

CACHE_CONFIG = {
    "maxsize": int(os.getenv("VGATE_CACHE_MAXSIZE", "1000")),
}

LOGGING_CONFIG = {
    "level": DEFAULT_LOG_LEVEL,
    "json_format": DEFAULT_JSON_FORMAT,
}

# Version
APP_VERSION = "0.3.0"

# Initialize the VGateEngine
MODEL_NAME = os.getenv("VGATE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct-AWQ")
engine = VGateEngine(model_name=MODEL_NAME)

# Initialize the RequestBatcher with cache config
batcher = RequestBatcher(
    engine=engine,
    max_batch_size=BATCH_CONFIG["max_batch_size"],
    max_wait_time_ms=BATCH_CONFIG["max_wait_time_ms"],
    cache_config=CacheConfig(maxsize=CACHE_CONFIG["maxsize"]),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup/shutdown."""
    # Initialize app info for Prometheus
    init_app_info(version=APP_VERSION, model=MODEL_NAME)

    # Startup: start the batcher
    await batcher.start()
    app_logger.info(
        "V-Gate started",
        extra={"extra_data": {
            "version": APP_VERSION,
            "model": MODEL_NAME,
            "batch_config": BATCH_CONFIG,
            "cache_config": CACHE_CONFIG
        }}
    )
    yield
    # Shutdown: stop the batcher
    await batcher.stop()
    app_logger.info("V-Gate stopped")


app = FastAPI(
    title="V-Gate AI Model Serving Gateway",
    description="A high-performance AI model serving gateway for various models.",
    version=APP_VERSION,
    lifespan=lifespan,
)


# Request/Response middleware for logging and metrics
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Middleware for request logging and Prometheus metrics."""
    request_id = str(uuid.uuid4())[:8]
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

        # Update Prometheus metrics
        REQUEST_COUNT.labels(
            endpoint=endpoint,
            method=request.method,
            status=str(status_code)
        ).inc()
        REQUEST_LATENCY.labels(
            endpoint=endpoint,
            method=request.method
        ).observe(latency)
        REQUEST_IN_PROGRESS.labels(endpoint=endpoint).dec()

        # Log request completion (skip /metrics and /health for less noise)
        if endpoint not in ["/metrics", "/health", "/metrics/prometheus"]:
            app_logger.info(
                "Request completed",
                extra={"extra_data": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": endpoint,
                    "status": status_code,
                    "latency_ms": round(latency * 1000, 2)
                }}
            )

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
async def prometheus_metrics():
    """
    Returns metrics in Prometheus format.
    Scrape this endpoint with Prometheus for monitoring.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
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
            "batch": BATCH_CONFIG,
            "cache": CACHE_CONFIG,
            "logging": LOGGING_CONFIG,
        },
        "version": APP_VERSION,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
