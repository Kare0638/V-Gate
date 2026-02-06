"""Tests for synchronous and asynchronous V-Gate clients."""

import pytest
import httpx
import json

from vgate_client import VGate, AsyncVGate
from vgate_client.exceptions import (
    AuthenticationError,
    ConnectionError,
    RateLimitError,
    ServerError,
    VGateError,
)
from vgate_client.models import ChatCompletion, EmbeddingResponse, HealthResponse


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
    headers: dict | None = None,
) -> httpx.Response:
    """Build a fake httpx.Response."""
    content = json.dumps(json_data or {}).encode()
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers=headers or {},
        request=httpx.Request("POST", "http://test/"),
    )


# ── Sync Client Tests ───────────────────────────────────────────────────────


class TestVGateInit:
    def test_default_base_url(self):
        client = VGate()
        assert client.base_url == "http://localhost:8000"
        client.close()

    def test_custom_base_url(self):
        client = VGate(base_url="http://myhost:9000/")
        assert client.base_url == "http://myhost:9000"
        client.close()

    def test_api_key_set_in_headers(self):
        client = VGate(api_key="sk-test-123")
        assert client._http.headers["Authorization"] == "Bearer sk-test-123"
        client.close()

    def test_no_api_key(self):
        client = VGate()
        assert "Authorization" not in client._http.headers
        client.close()

    def test_context_manager(self):
        with VGate() as client:
            assert client.base_url == "http://localhost:8000"


class TestSyncChatCreate:
    def test_success(self, chat_response_payload, monkeypatch):
        client = VGate(api_key="sk-test")
        mock_resp = _mock_response(200, chat_response_payload)
        monkeypatch.setattr(client._http, "request", lambda *a, **kw: mock_resp)

        result = client.chat.create(
            model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert isinstance(result, ChatCompletion)
        assert result.choices[0].message.content == "Hello! How can I help you?"
        assert result.usage.total_tokens == 18
        client.close()

    def test_custom_params(self, chat_response_payload, monkeypatch):
        client = VGate()
        calls = []

        def mock_request(method, path, **kw):
            calls.append(kw.get("json", {}))
            return _mock_response(200, chat_response_payload)

        monkeypatch.setattr(client._http, "request", mock_request)

        client.chat.create(
            model="m",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.1,
            top_p=0.5,
            max_tokens=50,
        )

        sent = calls[0]
        assert sent["temperature"] == 0.1
        assert sent["top_p"] == 0.5
        assert sent["max_tokens"] == 50
        client.close()


class TestSyncEmbeddingsCreate:
    def test_success(self, embedding_response_payload, monkeypatch):
        client = VGate()
        mock_resp = _mock_response(200, embedding_response_payload)
        monkeypatch.setattr(client._http, "request", lambda *a, **kw: mock_resp)

        result = client.embeddings.create(model="m", input="hello world")
        assert isinstance(result, EmbeddingResponse)
        assert len(result.data[0].embedding) == 1536
        client.close()


class TestSyncHealth:
    def test_success(self, health_response_payload, monkeypatch):
        client = VGate()
        mock_resp = _mock_response(200, health_response_payload)
        monkeypatch.setattr(client._http, "get", lambda *a, **kw: mock_resp)

        result = client.health()
        assert isinstance(result, HealthResponse)
        assert result.status == "ok"
        assert result.version == "0.3.2"
        client.close()


class TestSyncStats:
    def test_success(self, stats_response_payload, monkeypatch):
        client = VGate()
        mock_resp = _mock_response(200, stats_response_payload)
        monkeypatch.setattr(client._http, "request", lambda *a, **kw: mock_resp)

        result = client.stats()
        assert result["batcher"]["total_requests"] == 100
        client.close()


class TestSyncErrorHandling:
    def test_401_raises_auth_error(self, monkeypatch):
        client = VGate()
        mock_resp = _mock_response(401, {"detail": "Invalid API key"})
        monkeypatch.setattr(client._http, "request", lambda *a, **kw: mock_resp)

        with pytest.raises(AuthenticationError) as exc_info:
            client.chat.create(model="m", messages=[{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 401
        client.close()

    def test_429_raises_rate_limit_error(self, monkeypatch):
        client = VGate(max_retries=0)
        mock_resp = _mock_response(
            429,
            {"detail": "Rate limit exceeded"},
            headers={"Retry-After": "30", "X-RateLimit-Limit": "100"},
        )
        monkeypatch.setattr(client._http, "request", lambda *a, **kw: mock_resp)

        with pytest.raises(RateLimitError) as exc_info:
            client.chat.create(model="m", messages=[{"role": "user", "content": "hi"}])
        assert exc_info.value.retry_after == 30.0
        client.close()

    def test_500_raises_server_error(self, monkeypatch):
        client = VGate(max_retries=0)
        mock_resp = _mock_response(500, {"detail": "Internal error"})
        monkeypatch.setattr(client._http, "request", lambda *a, **kw: mock_resp)

        with pytest.raises(ServerError) as exc_info:
            client.chat.create(model="m", messages=[{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 500
        client.close()

    def test_422_raises_vgate_error(self, monkeypatch):
        client = VGate()
        mock_resp = _mock_response(422, {"detail": "Validation error"})
        monkeypatch.setattr(client._http, "request", lambda *a, **kw: mock_resp)

        with pytest.raises(VGateError) as exc_info:
            client.chat.create(model="m", messages=[{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 422
        client.close()

    def test_connection_error(self, monkeypatch):
        client = VGate()

        def raise_connect(*a, **kw):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(client._http, "request", raise_connect)

        with pytest.raises(ConnectionError):
            client.chat.create(model="m", messages=[{"role": "user", "content": "hi"}])
        client.close()

    def test_health_connection_error(self, monkeypatch):
        client = VGate()

        def raise_connect(*a, **kw):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(client._http, "get", raise_connect)

        with pytest.raises(ConnectionError):
            client.health()
        client.close()


class TestSyncRetry:
    def test_retry_on_429(self, chat_response_payload, monkeypatch):
        """Should retry on 429, then succeed."""
        client = VGate(max_retries=2)
        call_count = 0

        def mock_request(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_response(429, {"detail": "rate limited"}, {"Retry-After": "0"})
            return _mock_response(200, chat_response_payload)

        monkeypatch.setattr(client._http, "request", mock_request)
        # Patch sleep to avoid waiting
        monkeypatch.setattr("time.sleep", lambda _: None)

        result = client.chat.create(model="m", messages=[{"role": "user", "content": "hi"}])
        assert isinstance(result, ChatCompletion)
        assert call_count == 2
        client.close()

    def test_retry_on_500(self, chat_response_payload, monkeypatch):
        """Should retry on 500, then succeed."""
        client = VGate(max_retries=2)
        call_count = 0

        def mock_request(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_response(500, {"detail": "server error"})
            return _mock_response(200, chat_response_payload)

        monkeypatch.setattr(client._http, "request", mock_request)
        monkeypatch.setattr("time.sleep", lambda _: None)

        result = client.chat.create(model="m", messages=[{"role": "user", "content": "hi"}])
        assert isinstance(result, ChatCompletion)
        assert call_count == 2
        client.close()

    def test_exhausted_retries_raises(self, monkeypatch):
        """Should raise after max retries exceeded."""
        client = VGate(max_retries=1)
        monkeypatch.setattr(
            client._http, "request",
            lambda *a, **kw: _mock_response(500, {"detail": "down"}),
        )
        monkeypatch.setattr("time.sleep", lambda _: None)

        with pytest.raises(ServerError):
            client.chat.create(model="m", messages=[{"role": "user", "content": "hi"}])
        client.close()


# ── Async Client Tests ───────────────────────────────────────────────────────


class TestAsyncVGateInit:
    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with AsyncVGate() as client:
            assert client.base_url == "http://localhost:8000"

    @pytest.mark.asyncio
    async def test_api_key(self):
        client = AsyncVGate(api_key="sk-async-test")
        assert client._http.headers["Authorization"] == "Bearer sk-async-test"
        await client.close()


class TestAsyncChatCreate:
    @pytest.mark.asyncio
    async def test_success(self, chat_response_payload, monkeypatch):
        client = AsyncVGate()

        async def mock_request(*a, **kw):
            return _mock_response(200, chat_response_payload)

        monkeypatch.setattr(client._http, "request", mock_request)

        result = await client.chat.create(
            model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert isinstance(result, ChatCompletion)
        assert result.choices[0].message.content == "Hello! How can I help you?"
        await client.close()


class TestAsyncEmbeddingsCreate:
    @pytest.mark.asyncio
    async def test_success(self, embedding_response_payload, monkeypatch):
        client = AsyncVGate()

        async def mock_request(*a, **kw):
            return _mock_response(200, embedding_response_payload)

        monkeypatch.setattr(client._http, "request", mock_request)

        result = await client.embeddings.create(model="m", input="hello")
        assert isinstance(result, EmbeddingResponse)
        assert len(result.data[0].embedding) == 1536
        await client.close()


class TestAsyncHealth:
    @pytest.mark.asyncio
    async def test_success(self, health_response_payload, monkeypatch):
        client = AsyncVGate()

        async def mock_get(*a, **kw):
            return _mock_response(200, health_response_payload)

        monkeypatch.setattr(client._http, "get", mock_get)

        result = await client.health()
        assert isinstance(result, HealthResponse)
        assert result.status == "ok"
        await client.close()


class TestAsyncErrorHandling:
    @pytest.mark.asyncio
    async def test_401(self, monkeypatch):
        client = AsyncVGate()

        async def mock_request(*a, **kw):
            return _mock_response(401, {"detail": "Invalid API key"})

        monkeypatch.setattr(client._http, "request", mock_request)

        with pytest.raises(AuthenticationError):
            await client.chat.create(model="m", messages=[{"role": "user", "content": "hi"}])
        await client.close()

    @pytest.mark.asyncio
    async def test_429(self, monkeypatch):
        client = AsyncVGate(max_retries=0)

        async def mock_request(*a, **kw):
            return _mock_response(429, {"detail": "Rate limit"}, {"Retry-After": "10"})

        monkeypatch.setattr(client._http, "request", mock_request)

        with pytest.raises(RateLimitError) as exc_info:
            await client.chat.create(model="m", messages=[{"role": "user", "content": "hi"}])
        assert exc_info.value.retry_after == 10.0
        await client.close()

    @pytest.mark.asyncio
    async def test_connection_error(self, monkeypatch):
        client = AsyncVGate()

        async def raise_connect(*a, **kw):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(client._http, "request", raise_connect)

        with pytest.raises(ConnectionError):
            await client.chat.create(model="m", messages=[{"role": "user", "content": "hi"}])
        await client.close()


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_retry_on_429(self, chat_response_payload, monkeypatch):
        client = AsyncVGate(max_retries=2)
        call_count = 0

        async def mock_request(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_response(429, {"detail": "rate limited"}, {"Retry-After": "0"})
            return _mock_response(200, chat_response_payload)

        monkeypatch.setattr(client._http, "request", mock_request)

        import asyncio

        async def noop_sleep(_):
            pass

        monkeypatch.setattr(asyncio, "sleep", noop_sleep)

        result = await client.chat.create(model="m", messages=[{"role": "user", "content": "hi"}])
        assert isinstance(result, ChatCompletion)
        assert call_count == 2
        await client.close()
