from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import sys
import time

# Add the vgate directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'vgate')))

# Import the actual VGateEngine and Batcher
from engine import VGateEngine
from batcher import RequestBatcher
from cache import CacheConfig

# Configuration
BATCH_CONFIG = {
    "max_batch_size": 8,       # Max requests per batch
    "max_wait_time_ms": 50.0,  # Max wait time before processing
}

CACHE_CONFIG = {
    "maxsize": int(os.getenv("VGATE_CACHE_MAXSIZE", "1000")),
}

# Initialize the VGateEngine
engine = VGateEngine(model_name="Qwen/Qwen2.5-1.5B-Instruct-AWQ")

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
    # Startup: start the batcher
    await batcher.start()
    yield
    # Shutdown: stop the batcher
    await batcher.stop()


app = FastAPI(
    title="V-Gate AI Model Serving Gateway",
    description="A high-performance AI model serving gateway for various models.",
    version="0.2.0",
    lifespan=lifespan,
)

# Request models for OpenAI-like API
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    # Add other fields as needed for OpenAI compatibility (e.g., temperature, max_tokens)

class EmbeddingRequest(BaseModel):
    model: str
    input: str
    # Add other fields as needed

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
    return {"status": "ok"}

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
        response = await batcher.submit(prompt, max_tokens=256)

        # Adapt engine's response to OpenAI-like format
        return {
            "id": "chatcmpl-" + str(hash(prompt)),
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", summary="Get Batching Metrics")
async def get_metrics():
    """
    Returns metrics about the request batching system and cache.
    Useful for monitoring and debugging.
    """
    metrics = batcher.get_metrics()
    return {
        "batcher": {
            "total_requests": metrics["total_requests"],
            "total_batches": metrics["total_batches"],
            "average_batch_size": metrics["average_batch_size"],
            "pending_requests": metrics["pending_requests"],
        },
        "cache": metrics["cache"],
        "config": {
            "batch": BATCH_CONFIG,
            "cache": CACHE_CONFIG,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
