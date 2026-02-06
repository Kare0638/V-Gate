import pytest


@pytest.fixture
def chat_response_payload():
    """Standard chat completion response matching server format."""
    return {
        "id": "chatcmpl-abc12345",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }


@pytest.fixture
def embedding_response_payload():
    """Standard embedding response matching server format."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1] * 1536,
                "index": 0,
            }
        ],
        "model": "mock-embedding-model",
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 0,
            "total_tokens": 4,
        },
    }


@pytest.fixture
def health_response_payload():
    return {"status": "ok", "version": "0.3.2"}


@pytest.fixture
def stats_response_payload():
    return {
        "batcher": {
            "total_requests": 100,
            "total_batches": 20,
            "average_batch_size": 5.0,
            "pending_requests": 0,
            "total_deduplicated": 10,
        },
        "cache": {
            "size": 50,
            "maxsize": 1000,
            "hits": 30,
            "misses": 70,
            "evictions": 0,
            "hit_rate": 0.3,
        },
        "config": {},
        "version": "0.3.2",
    }
