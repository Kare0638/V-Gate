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
