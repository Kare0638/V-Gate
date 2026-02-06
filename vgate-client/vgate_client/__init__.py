"""V-Gate Python Client SDK.

Usage::

    from vgate_client import VGate, AsyncVGate

    # Synchronous
    client = VGate(base_url="http://localhost:8000", api_key="sk-...")
    resp = client.chat.create(
        model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(resp.choices[0].message.content)
    client.close()

    # Asynchronous
    async with AsyncVGate(base_url="http://localhost:8000", api_key="sk-...") as client:
        resp = await client.chat.create(
            model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            messages=[{"role": "user", "content": "Hello!"}],
        )
"""

from .client import AsyncVGate, VGate
from .exceptions import (
    AuthenticationError,
    ConnectionError,
    RateLimitError,
    ServerError,
    VGateError,
)
from .models import (
    ChatCompletion,
    ChatMessage,
    Choice,
    EmbeddingData,
    EmbeddingResponse,
    HealthResponse,
    RateLimitInfo,
    ResponseMessage,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    # Clients
    "VGate",
    "AsyncVGate",
    # Response models
    "ChatCompletion",
    "Choice",
    "ResponseMessage",
    "EmbeddingResponse",
    "EmbeddingData",
    "HealthResponse",
    "Usage",
    "RateLimitInfo",
    "ChatMessage",
    # Exceptions
    "VGateError",
    "AuthenticationError",
    "RateLimitError",
    "ServerError",
    "ConnectionError",
]
