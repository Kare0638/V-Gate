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

"""Tests for client.chat.stream() (sync and async SSE consumption)."""

import json

import httpx
import pytest

from vgate_client import AsyncVGate, VGate
from vgate_client.exceptions import (
    AuthenticationError,
    ConnectionError,
    RateLimitError,
    ServerError,
    VGateError,
)
from vgate_client.models import ChatCompletionChunk

_NORMAL_LINES = [
    'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
    "",
    ": this is an SSE comment, not a data line",
    "event: message",
    'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}',
    'data:{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}',
    'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
    "data: [DONE]",
]

_NO_DONE_LINES = [
    'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
    'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}',
    # connection drops here — no "data: [DONE]" line follows
]

_ERROR_MID_STREAM_LINES = [
    'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
    'data: {"error": {"message": "backend exploded", "type": "RuntimeError"}}',
]


def _build_response(status_code, content_type, json_body):
    headers = {}
    if content_type is not None:
        headers["content-type"] = content_type
    return httpx.Response(
        status_code=status_code,
        headers=headers,
        content=json.dumps(json_body or {}).encode(),
        request=httpx.Request("POST", "http://test/"),
    )


class _FakeSyncStreamCM:
    """Stand-in for the context manager httpx.Client.stream() returns."""

    def __init__(self, lines=None, status_code=200, content_type="text/event-stream", json_body=None):
        self.lines = lines or []
        self.status_code = status_code
        self.content_type = content_type
        self.json_body = json_body
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        response = _build_response(self.status_code, self.content_type, self.json_body)
        response.iter_lines = lambda: iter(self.lines)
        return response

    def __exit__(self, *exc_info):
        self.exited = True
        return False


class _FakeAsyncStreamCM:
    """Stand-in for the async context manager httpx.AsyncClient.stream() returns.

    AsyncClient.stream() itself is a plain (non-async) method that returns an
    async context manager, so the monkeypatched replacement must also be a
    plain function/lambda — an `async def` here would need an extra await
    the real client code never performs.
    """

    def __init__(self, lines=None, status_code=200, content_type="text/event-stream", json_body=None):
        self.lines = lines or []
        self.status_code = status_code
        self.content_type = content_type
        self.json_body = json_body
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        response = _build_response(self.status_code, self.content_type, self.json_body)

        async def _aiter_lines():
            for line in self.lines:
                yield line

        response.aiter_lines = _aiter_lines
        return response

    async def __aexit__(self, *exc_info):
        self.exited = True
        return False


# ── Sync ─────────────────────────────────────────────────────────────────────


class TestSyncStream:
    def test_yields_role_content_and_finish_chunks(self, monkeypatch):
        client = VGate()
        monkeypatch.setattr(
            client._http, "stream", lambda *a, **kw: _FakeSyncStreamCM(lines=_NORMAL_LINES)
        )

        chunks = list(
            client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}])
        )

        assert all(isinstance(c, ChatCompletionChunk) for c in chunks)
        assert chunks[0].choices[0].delta.role == "assistant"
        content = "".join(
            c.choices[0].delta.content for c in chunks if c.choices[0].delta.content
        )
        assert content == "Hello"
        assert chunks[-1].choices[0].finish_reason == "stop"
        client.close()

    def test_request_body_sets_stream_true(self, monkeypatch):
        client = VGate()
        captured = {}

        def fake_stream(method, path, **kw):
            captured["json"] = kw.get("json")
            return _FakeSyncStreamCM(lines=_NORMAL_LINES)

        monkeypatch.setattr(client._http, "stream", fake_stream)

        list(client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}]))

        assert captured["json"]["stream"] is True
        client.close()

    def test_context_manager_exits_on_normal_completion(self, monkeypatch):
        client = VGate()
        holder = {}

        def fake_stream(*a, **kw):
            cm = _FakeSyncStreamCM(lines=_NORMAL_LINES)
            holder["cm"] = cm
            return cm

        monkeypatch.setattr(client._http, "stream", fake_stream)

        list(client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}]))

        assert holder["cm"].entered is True
        assert holder["cm"].exited is True
        client.close()

    def test_mid_stream_error_event_raises_server_error(self, monkeypatch):
        client = VGate()
        holder = {}

        def fake_stream(*a, **kw):
            cm = _FakeSyncStreamCM(lines=_ERROR_MID_STREAM_LINES)
            holder["cm"] = cm
            return cm

        monkeypatch.setattr(client._http, "stream", fake_stream)

        with pytest.raises(ServerError, match="backend exploded"):
            list(client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}]))

        assert holder["cm"].exited is True
        client.close()

    @pytest.mark.parametrize(
        "status_code,exc_type",
        [(401, AuthenticationError), (429, RateLimitError), (500, ServerError)],
    )
    def test_http_error_before_any_data_raises(self, monkeypatch, status_code, exc_type):
        client = VGate(max_retries=0)
        monkeypatch.setattr(
            client._http,
            "stream",
            lambda *a, **kw: _FakeSyncStreamCM(
                status_code=status_code,
                content_type="application/json",
                json_body={"detail": "boom"},
            ),
        )

        with pytest.raises(exc_type):
            list(client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}]))
        client.close()

    def test_unexpected_content_type_raises(self, monkeypatch):
        client = VGate()
        monkeypatch.setattr(
            client._http,
            "stream",
            lambda *a, **kw: _FakeSyncStreamCM(content_type="application/json", json_body={}),
        )

        with pytest.raises(VGateError, match="text/event-stream"):
            list(client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}]))
        client.close()

    def test_connection_error(self, monkeypatch):
        client = VGate()

        def raise_connect(*a, **kw):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(client._http, "stream", raise_connect)

        with pytest.raises(ConnectionError):
            list(client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}]))
        client.close()

    def test_missing_done_raises_server_error(self, monkeypatch):
        """A connection that ends without ever sending `data: [DONE]` must
        not look like a clean completion — it should raise, not silently
        stop yielding."""
        client = VGate()
        monkeypatch.setattr(
            client._http, "stream", lambda *a, **kw: _FakeSyncStreamCM(lines=_NO_DONE_LINES)
        )

        chunks = []
        with pytest.raises(ServerError, match=r"\[DONE\]"):
            for chunk in client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}]):
                chunks.append(chunk)

        # The chunks that did arrive before the drop are still delivered.
        assert len(chunks) == 2
        client.close()

    def test_context_manager_closes_connection_on_early_break(self, monkeypatch):
        """Stopping iteration early (via `break` inside a `with` block) must
        still close the underlying SSE connection deterministically."""
        client = VGate()
        holder = {}

        def fake_stream(*a, **kw):
            cm = _FakeSyncStreamCM(lines=_NORMAL_LINES)
            holder["cm"] = cm
            return cm

        monkeypatch.setattr(client._http, "stream", fake_stream)

        seen = []
        with client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}]) as stream:
            for chunk in stream:
                seen.append(chunk)
                if chunk.choices[0].delta.role == "assistant":
                    break

        assert len(seen) == 1
        assert holder["cm"].exited is True
        client.close()

    def test_explicit_close_without_with_closes_connection(self, monkeypatch):
        """Even without the context-manager form, calling .close() directly
        on the returned stream must close the underlying connection."""
        client = VGate()
        holder = {}

        def fake_stream(*a, **kw):
            cm = _FakeSyncStreamCM(lines=_NORMAL_LINES)
            holder["cm"] = cm
            return cm

        monkeypatch.setattr(client._http, "stream", fake_stream)

        stream = client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}])
        next(stream)
        stream.close()

        assert holder["cm"].exited is True
        client.close()


# ── Async ────────────────────────────────────────────────────────────────────


class TestAsyncStream:
    @pytest.mark.asyncio
    async def test_yields_role_content_and_finish_chunks(self, monkeypatch):
        client = AsyncVGate()
        monkeypatch.setattr(
            client._http, "stream", lambda *a, **kw: _FakeAsyncStreamCM(lines=_NORMAL_LINES)
        )

        chunks = [
            c async for c in client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}])
        ]

        assert all(isinstance(c, ChatCompletionChunk) for c in chunks)
        assert chunks[0].choices[0].delta.role == "assistant"
        content = "".join(
            c.choices[0].delta.content for c in chunks if c.choices[0].delta.content
        )
        assert content == "Hello"
        assert chunks[-1].choices[0].finish_reason == "stop"
        await client.close()

    @pytest.mark.asyncio
    async def test_request_body_sets_stream_true(self, monkeypatch):
        client = AsyncVGate()
        captured = {}

        def fake_stream(method, path, **kw):
            captured["json"] = kw.get("json")
            return _FakeAsyncStreamCM(lines=_NORMAL_LINES)

        monkeypatch.setattr(client._http, "stream", fake_stream)

        async for _ in client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}]):
            pass

        assert captured["json"]["stream"] is True
        await client.close()

    @pytest.mark.asyncio
    async def test_context_manager_exits_on_normal_completion(self, monkeypatch):
        client = AsyncVGate()
        holder = {}

        def fake_stream(*a, **kw):
            cm = _FakeAsyncStreamCM(lines=_NORMAL_LINES)
            holder["cm"] = cm
            return cm

        monkeypatch.setattr(client._http, "stream", fake_stream)

        async for _ in client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}]):
            pass

        assert holder["cm"].entered is True
        assert holder["cm"].exited is True
        await client.close()

    @pytest.mark.asyncio
    async def test_mid_stream_error_event_raises_server_error(self, monkeypatch):
        client = AsyncVGate()
        holder = {}

        def fake_stream(*a, **kw):
            cm = _FakeAsyncStreamCM(lines=_ERROR_MID_STREAM_LINES)
            holder["cm"] = cm
            return cm

        monkeypatch.setattr(client._http, "stream", fake_stream)

        with pytest.raises(ServerError, match="backend exploded"):
            async for _ in client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}]):
                pass

        assert holder["cm"].exited is True
        await client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,exc_type",
        [(401, AuthenticationError), (429, RateLimitError), (500, ServerError)],
    )
    async def test_http_error_before_any_data_raises(self, monkeypatch, status_code, exc_type):
        client = AsyncVGate(max_retries=0)
        monkeypatch.setattr(
            client._http,
            "stream",
            lambda *a, **kw: _FakeAsyncStreamCM(
                status_code=status_code,
                content_type="application/json",
                json_body={"detail": "boom"},
            ),
        )

        with pytest.raises(exc_type):
            async for _ in client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}]):
                pass
        await client.close()

    @pytest.mark.asyncio
    async def test_connection_error(self, monkeypatch):
        client = AsyncVGate()

        def raise_connect(*a, **kw):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(client._http, "stream", raise_connect)

        with pytest.raises(ConnectionError):
            async for _ in client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}]):
                pass
        await client.close()

    @pytest.mark.asyncio
    async def test_missing_done_raises_server_error(self, monkeypatch):
        """A connection that ends without ever sending `data: [DONE]` must
        not look like a clean completion — it should raise, not silently
        stop yielding."""
        client = AsyncVGate()
        monkeypatch.setattr(
            client._http, "stream", lambda *a, **kw: _FakeAsyncStreamCM(lines=_NO_DONE_LINES)
        )

        chunks = []
        with pytest.raises(ServerError, match=r"\[DONE\]"):
            async for chunk in client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}]):
                chunks.append(chunk)

        assert len(chunks) == 2
        await client.close()

    @pytest.mark.asyncio
    async def test_context_manager_closes_connection_on_early_break(self, monkeypatch):
        """Stopping iteration early (via `break` inside an `async with` block)
        must still close the underlying SSE connection deterministically."""
        client = AsyncVGate()
        holder = {}

        def fake_stream(*a, **kw):
            cm = _FakeAsyncStreamCM(lines=_NORMAL_LINES)
            holder["cm"] = cm
            return cm

        monkeypatch.setattr(client._http, "stream", fake_stream)

        seen = []
        async with client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}]) as stream:
            async for chunk in stream:
                seen.append(chunk)
                if chunk.choices[0].delta.role == "assistant":
                    break

        assert len(seen) == 1
        assert holder["cm"].exited is True
        await client.close()

    @pytest.mark.asyncio
    async def test_explicit_close_without_with_closes_connection(self, monkeypatch):
        """Even without the context-manager form, calling .aclose() directly
        on the returned stream must close the underlying connection."""
        client = AsyncVGate()
        holder = {}

        def fake_stream(*a, **kw):
            cm = _FakeAsyncStreamCM(lines=_NORMAL_LINES)
            holder["cm"] = cm
            return cm

        monkeypatch.setattr(client._http, "stream", fake_stream)

        stream = client.chat.stream(model="m", messages=[{"role": "user", "content": "hi"}])
        await stream.__anext__()
        await stream.aclose()

        assert holder["cm"].exited is True
        await client.close()
