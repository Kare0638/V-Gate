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

"""
Tests for POST /v1/chat/completions with stream=true (SSE).

Streaming bypasses RequestBatcher and talks to engine.backend.stream_generate()
directly (see main.py's _stream_chat_completion docstring) — this is a
known, intentional MVP limitation, not something these tests should hide.
"""

import json

import pytest

from vgate.config import reset_config


@pytest.fixture(autouse=True)
def _reset():
    reset_config()
    yield
    reset_config()


async def _collect_sse_events(response):
    events = []
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            events.append(line[len("data: "):])
    return events


class TestStreamingChatCompletions:
    @pytest.mark.asyncio
    async def test_stream_true_returns_sse_with_role_content_and_done(self):
        from httpx import ASGITransport, AsyncClient
        from main import app, lifespan

        async with lifespan(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                async with client.stream("POST", "/v1/chat/completions", json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello streaming"}],
                    "max_tokens": 10,
                    "stream": True,
                }) as response:
                    assert response.status_code == 200
                    assert "text/event-stream" in response.headers["content-type"]
                    events = await _collect_sse_events(response)

        assert events[-1] == "[DONE]"

        chunks = [json.loads(e) for e in events[:-1]]
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}

        content_deltas = [
            c["choices"][0]["delta"]["content"]
            for c in chunks
            if "content" in c["choices"][0]["delta"]
        ]
        assert len(content_deltas) > 0
        assert "".join(content_deltas).strip() != ""

        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        # Every chunk shares one completion id (a single logical response)
        assert len({c["id"] for c in chunks}) == 1

    @pytest.mark.asyncio
    async def test_stream_false_is_unaffected(self):
        """Default (stream omitted) must still return a plain JSON response,
        not SSE — this is the Phase 0 non-streaming path, untouched."""
        from httpx import ASGITransport, AsyncClient
        from main import app, lifespan

        async with lifespan(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/v1/chat/completions", json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                })

        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
        assert "choices" in response.json()
