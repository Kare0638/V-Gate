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

"""Synchronous and asynchronous clients for the V-Gate API."""

from __future__ import annotations

import time
from typing import Optional, Union

import httpx

from .exceptions import (
    AuthenticationError,
    ConnectionError,
    RateLimitError,
    ServerError,
    VGateError,
)
from .models import (
    ChatCompletion,
    ChatCompletionRequest,
    ChatMessage,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    RateLimitInfo,
)

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MAX_RETRIES = 2


# ── Helpers ─────────────────────────────────────────────────────────────────


def _parse_rate_limit(headers: httpx.Headers) -> RateLimitInfo:
    """Extract rate-limit metadata from response headers."""
    def _int(key: str) -> Optional[int]:
        v = headers.get(key)
        return int(v) if v is not None else None

    def _float(key: str) -> Optional[float]:
        v = headers.get(key)
        return float(v) if v is not None else None

    return RateLimitInfo(
        limit=_int("X-RateLimit-Limit"),
        remaining=_int("X-RateLimit-Remaining"),
        reset=_float("X-RateLimit-Reset"),
        retry_after=_float("Retry-After"),
    )


def _raise_for_status(response: httpx.Response) -> None:
    """Translate HTTP error responses into typed exceptions."""
    if response.is_success:
        return

    status = response.status_code
    try:
        body = response.json()
    except Exception:
        body = {"detail": response.text}

    detail = body.get("detail", response.text)

    if status == 401:
        raise AuthenticationError(detail, status_code=status, body=body)
    if status == 429:
        retry_after = _parse_rate_limit(response.headers).retry_after
        raise RateLimitError(
            detail, retry_after=retry_after, status_code=status, body=body,
        )
    if status >= 500:
        raise ServerError(detail, status_code=status, body=body)
    raise VGateError(detail, status_code=status, body=body)


def _build_headers(api_key: Optional[str]) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


# ── Resource helpers (namespace objects) ────────────────────────────────────


class _SyncChat:
    """Synchronous chat completions resource — accessed as ``client.chat``."""

    def __init__(self, client: VGate):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> ChatCompletion:
        """Create a chat completion.

        Args:
            model: Model identifier on the V-Gate server.
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            temperature: Sampling temperature.
            top_p: Top-p nucleus sampling.
            max_tokens: Maximum tokens to generate.

        Returns:
            A ``ChatCompletion`` object.
        """
        req = ChatCompletionRequest(
            model=model,
            messages=[ChatMessage(**m) for m in messages],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        data = self._client._request("POST", "/v1/chat/completions", json=req.model_dump())
        return ChatCompletion.model_validate(data)


class _SyncEmbeddings:
    """Synchronous embeddings resource — accessed as ``client.embeddings``."""

    def __init__(self, client: VGate):
        self._client = client

    def create(
        self,
        *,
        model: str,
        input: str,
    ) -> EmbeddingResponse:
        """Create an embedding.

        Args:
            model: Model identifier.
            input: Text to embed.

        Returns:
            An ``EmbeddingResponse`` object.
        """
        req = EmbeddingRequest(model=model, input=input)
        data = self._client._request("POST", "/v1/embeddings", json=req.model_dump())
        return EmbeddingResponse.model_validate(data)


class _AsyncChat:
    """Async chat completions resource — accessed as ``client.chat``."""

    def __init__(self, client: AsyncVGate):
        self._client = client

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> ChatCompletion:
        req = ChatCompletionRequest(
            model=model,
            messages=[ChatMessage(**m) for m in messages],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        data = await self._client._request("POST", "/v1/chat/completions", json=req.model_dump())
        return ChatCompletion.model_validate(data)


class _AsyncEmbeddings:
    """Async embeddings resource — accessed as ``client.embeddings``."""

    def __init__(self, client: AsyncVGate):
        self._client = client

    async def create(
        self,
        *,
        model: str,
        input: str,
    ) -> EmbeddingResponse:
        req = EmbeddingRequest(model=model, input=input)
        data = await self._client._request("POST", "/v1/embeddings", json=req.model_dump())
        return EmbeddingResponse.model_validate(data)


# ── Synchronous client ──────────────────────────────────────────────────────


class VGate:
    """Synchronous V-Gate client.

    Example::

        client = VGate(base_url="http://localhost:8000", api_key="sk-...")
        resp = client.chat.create(
            model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            messages=[{"role": "user", "content": "Hi"}],
        )
        print(resp.choices[0].message.content)
        client.close()
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self._http = httpx.Client(
            base_url=self.base_url,
            headers=_build_headers(api_key),
            timeout=timeout,
        )
        self.chat = _SyncChat(self)
        self.embeddings = _SyncEmbeddings(self)

    # -- low-level --------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> dict:
        """Send an HTTP request with retry logic for 429 / 5xx."""
        last_exc: Optional[Exception] = None
        for attempt in range(1 + self.max_retries):
            try:
                response = self._http.request(method, path, **kwargs)
            except httpx.ConnectError as exc:
                raise ConnectionError(f"Cannot connect to {self.base_url}: {exc}") from exc

            if response.is_success:
                return response.json()

            # Decide whether to retry
            if response.status_code == 429:
                info = _parse_rate_limit(response.headers)
                wait = info.retry_after or (2 ** attempt)
                if attempt < self.max_retries:
                    time.sleep(wait)
                    continue
                _raise_for_status(response)  # exhausted retries

            if response.status_code >= 500 and attempt < self.max_retries:
                time.sleep(2 ** attempt)
                continue

            _raise_for_status(response)

        # Should not reach here, but just in case
        raise last_exc or VGateError("Request failed after retries")

    # -- convenience endpoints --------------------------------------------

    def health(self) -> HealthResponse:
        """Check server health (``GET /health``)."""
        try:
            resp = self._http.get("/health")
        except httpx.ConnectError as exc:
            raise ConnectionError(f"Cannot connect to {self.base_url}: {exc}") from exc
        _raise_for_status(resp)
        return HealthResponse.model_validate(resp.json())

    def stats(self) -> dict:
        """Get server statistics (``GET /stats``)."""
        data = self._request("GET", "/stats")
        return data

    def rate_limit_info(self, response_headers: dict) -> RateLimitInfo:
        """Parse rate-limit headers from any raw response headers dict."""
        return _parse_rate_limit(httpx.Headers(response_headers))

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Asynchronous client ─────────────────────────────────────────────────────


class AsyncVGate:
    """Asynchronous V-Gate client.

    Example::

        async with AsyncVGate(base_url="http://localhost:8000", api_key="sk-...") as client:
            resp = await client.chat.create(
                model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
                messages=[{"role": "user", "content": "Hi"}],
            )
            print(resp.choices[0].message.content)
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            headers=_build_headers(api_key),
            timeout=timeout,
        )
        self.chat = _AsyncChat(self)
        self.embeddings = _AsyncEmbeddings(self)

    # -- low-level --------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> dict:
        """Send an HTTP request with retry logic for 429 / 5xx."""
        import asyncio

        for attempt in range(1 + self.max_retries):
            try:
                response = await self._http.request(method, path, **kwargs)
            except httpx.ConnectError as exc:
                raise ConnectionError(f"Cannot connect to {self.base_url}: {exc}") from exc

            if response.is_success:
                return response.json()

            if response.status_code == 429:
                info = _parse_rate_limit(response.headers)
                wait = info.retry_after or (2 ** attempt)
                if attempt < self.max_retries:
                    await asyncio.sleep(wait)
                    continue
                _raise_for_status(response)

            if response.status_code >= 500 and attempt < self.max_retries:
                await asyncio.sleep(2 ** attempt)
                continue

            _raise_for_status(response)

        raise VGateError("Request failed after retries")

    # -- convenience endpoints --------------------------------------------

    async def health(self) -> HealthResponse:
        """Check server health (``GET /health``)."""
        try:
            resp = await self._http.get("/health")
        except httpx.ConnectError as exc:
            raise ConnectionError(f"Cannot connect to {self.base_url}: {exc}") from exc
        _raise_for_status(resp)
        return HealthResponse.model_validate(resp.json())

    async def stats(self) -> dict:
        """Get server statistics (``GET /stats``)."""
        return await self._request("GET", "/stats")

    # -- lifecycle --------------------------------------------------------

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
