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

import asyncio
import json

import pytest

from vgate.config import reset_config
from vgate.metrics import (
    INFERENCE_ERRORS,
    STREAM_DURATION,
    STREAM_REQUESTS,
    STREAM_TOKENS,
    STREAM_TPOT,
    STREAM_TTFT,
    TOKENS_GENERATED,
)


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


def _hist_count_sum(histogram):
    """Read a Histogram's current (count, sum) via its own .collect(), the
    only public way prometheus_client exposes them (there's no ._count
    attribute — only ._sum — so we pull both from the emitted samples)."""
    count = total = None
    for metric in histogram.collect():
        for sample in metric.samples:
            if sample.name.endswith("_count"):
                count = sample.value
            elif sample.name.endswith("_sum"):
                total = sample.value
    return count, total


class _FakeStreamBackend:
    """Minimal backend stand-in so streaming-metrics tests can control
    exactly what stream_generate() yields, without going through the
    dry-run word-splitter or a real GPU backend. `delays` (seconds to sleep
    before each piece) lets TPOT-weighting tests use real elapsed time
    instead of monkeypatching the global time.monotonic — patching that
    process-wide would also break asyncio's own scheduler and pytest's test
    timing, since they read the same time.monotonic."""

    def __init__(self, pieces=None, error: Exception = None, delays=None):
        self._pieces = pieces or []
        self._error = error
        self._delays = delays or []

    def create_sampling_params(self, temperature, top_p, max_tokens):
        return {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}

    async def stream_generate(self, prompt, sampling_params):
        for i, piece in enumerate(self._pieces):
            if i < len(self._delays):
                await asyncio.sleep(self._delays[i])
            yield piece
        if self._error:
            raise self._error


class _FakeStreamEngine:
    def __init__(self, backend):
        self.backend = backend


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


class TestStreamingMetrics:
    """Streaming has its own vgate_stream_* metrics, kept separate from the
    batcher's TTFT/TPOT (different measurement source, see metrics.py), plus
    correct token-weighted TPOT and completed/error/cancelled bookkeeping.
    Prometheus collectors are process-global, so every assertion here is a
    before/after delta, never an absolute value.
    """

    @pytest.mark.asyncio
    async def test_successful_stream_records_metrics(self):
        from httpx import ASGITransport, AsyncClient
        from main import app, lifespan

        before_ttft_count, _ = _hist_count_sum(STREAM_TTFT)
        before_tpot_count, _ = _hist_count_sum(STREAM_TPOT)
        before_duration_count, _ = _hist_count_sum(STREAM_DURATION)
        before_stream_tokens = STREAM_TOKENS._value.get()
        before_global_tokens = TOKENS_GENERATED._value.get()
        before_completed = STREAM_REQUESTS.labels(status="completed")._value.get()

        async with lifespan(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                async with client.stream("POST", "/v1/chat/completions", json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello metrics test with several words"}],
                    "max_tokens": 10,
                    "stream": True,
                }) as response:
                    events = await _collect_sse_events(response)

        chunks = [json.loads(e) for e in events[:-1]]
        content_deltas = [
            c["choices"][0]["delta"]["content"]
            for c in chunks
            if "content" in c["choices"][0]["delta"]
        ]
        final_num_tokens = len(content_deltas)  # dry-run: one new token per content chunk

        after_ttft_count, _ = _hist_count_sum(STREAM_TTFT)
        after_tpot_count, _ = _hist_count_sum(STREAM_TPOT)
        after_duration_count, _ = _hist_count_sum(STREAM_DURATION)
        after_stream_tokens = STREAM_TOKENS._value.get()
        after_global_tokens = TOKENS_GENERATED._value.get()
        after_completed = STREAM_REQUESTS.labels(status="completed")._value.get()

        assert after_ttft_count == before_ttft_count + 1
        assert after_tpot_count == before_tpot_count + 1
        assert after_duration_count == before_duration_count + 1
        assert after_stream_tokens == before_stream_tokens + final_num_tokens
        assert after_global_tokens == before_global_tokens + final_num_tokens
        assert after_completed == before_completed + 1

    @pytest.mark.asyncio
    async def test_backend_error_mid_stream_records_error_status(self, monkeypatch):
        import main as main_module

        backend = _FakeStreamBackend(
            pieces=[{"delta": "partial", "num_tokens": 2}],
            error=RuntimeError("backend exploded"),
        )
        monkeypatch.setattr(main_module, "engine", _FakeStreamEngine(backend))

        before_error = STREAM_REQUESTS.labels(status="error")._value.get()
        before_completed = STREAM_REQUESTS.labels(status="completed")._value.get()
        before_inference_errors = INFERENCE_ERRORS.labels(error_type="RuntimeError")._value.get()

        request = main_module.ChatCompletionRequest(
            model="m",
            messages=[main_module.ChatMessage(role="user", content="hi")],
            max_tokens=10,
            stream=True,
        )
        events = [
            chunk async for chunk in main_module._stream_chat_completion("prompt", request)
        ]

        assert events[-1] == "data: [DONE]\n\n"
        assert any('"error"' in e for e in events)

        assert STREAM_REQUESTS.labels(status="error")._value.get() == before_error + 1
        assert STREAM_REQUESTS.labels(status="completed")._value.get() == before_completed
        assert (
            INFERENCE_ERRORS.labels(error_type="RuntimeError")._value.get()
            == before_inference_errors + 1
        )

    @pytest.mark.asyncio
    async def test_cancelled_stream_does_not_count_as_error(self, monkeypatch):
        """Simulates a client disconnect: the caller stops consuming and
        closes the generator instead of exhausting it. This must not raise
        (illegal yield during GeneratorExit) and must not be attributed to
        INFERENCE_ERRORS."""
        import main as main_module

        backend = _FakeStreamBackend(pieces=[
            {"delta": "a", "num_tokens": 1},
            {"delta": "b", "num_tokens": 2},
            {"delta": "c", "num_tokens": 3},
        ])
        monkeypatch.setattr(main_module, "engine", _FakeStreamEngine(backend))

        before_cancelled = STREAM_REQUESTS.labels(status="cancelled")._value.get()
        before_completed = STREAM_REQUESTS.labels(status="completed")._value.get()
        before_error = STREAM_REQUESTS.labels(status="error")._value.get()
        before_inference_errors_total = sum(
            s.value for m in INFERENCE_ERRORS.collect() for s in m.samples
            if s.name == "vgate_inference_errors_total"
        )

        request = main_module.ChatCompletionRequest(
            model="m",
            messages=[main_module.ChatMessage(role="user", content="hi")],
            max_tokens=10,
            stream=True,
        )
        gen = main_module._stream_chat_completion("prompt", request)
        await gen.__anext__()  # role chunk
        await gen.__anext__()  # first content chunk
        await gen.aclose()  # simulated disconnect — must not raise

        after_inference_errors_total = sum(
            s.value for m in INFERENCE_ERRORS.collect() for s in m.samples
            if s.name == "vgate_inference_errors_total"
        )

        assert STREAM_REQUESTS.labels(status="cancelled")._value.get() == before_cancelled + 1
        assert STREAM_REQUESTS.labels(status="completed")._value.get() == before_completed
        assert STREAM_REQUESTS.labels(status="error")._value.get() == before_error
        assert after_inference_errors_total == before_inference_errors_total

    @pytest.mark.asyncio
    async def test_disconnect_right_after_role_chunk_counts_as_cancelled(self, monkeypatch):
        """The role chunk yield must be inside the try/except so a disconnect
        that happens before any content arrives is still recorded — it must
        not fall outside all exception handling and vanish uncounted."""
        import main as main_module

        backend = _FakeStreamBackend(pieces=[
            {"delta": "a", "num_tokens": 1},
        ])
        monkeypatch.setattr(main_module, "engine", _FakeStreamEngine(backend))

        before_cancelled = STREAM_REQUESTS.labels(status="cancelled")._value.get()

        request = main_module.ChatCompletionRequest(
            model="m",
            messages=[main_module.ChatMessage(role="user", content="hi")],
            max_tokens=10,
            stream=True,
        )
        gen = main_module._stream_chat_completion("prompt", request)
        await gen.__anext__()  # role chunk only — nothing else consumed yet
        await gen.aclose()  # disconnect before any content delta

        assert STREAM_REQUESTS.labels(status="cancelled")._value.get() == before_cancelled + 1

    @pytest.mark.asyncio
    async def test_disconnect_during_error_event_yield_does_not_raise(self, monkeypatch):
        """A client can disconnect exactly while the mid-stream error event is
        being sent. GeneratorExit at that point doesn't match `except
        Exception`, so it must not reach a `finally` that still tries to
        yield — that's the same illegal-yield RuntimeError this whole
        cancelled/error/completed split exists to avoid."""
        import main as main_module

        backend = _FakeStreamBackend(pieces=[], error=RuntimeError("backend exploded"))
        monkeypatch.setattr(main_module, "engine", _FakeStreamEngine(backend))

        before_error = STREAM_REQUESTS.labels(status="error")._value.get()
        before_inference_errors = INFERENCE_ERRORS.labels(error_type="RuntimeError")._value.get()

        request = main_module.ChatCompletionRequest(
            model="m",
            messages=[main_module.ChatMessage(role="user", content="hi")],
            max_tokens=10,
            stream=True,
        )
        gen = main_module._stream_chat_completion("prompt", request)
        await gen.__anext__()  # role chunk
        await gen.__anext__()  # the error event chunk itself
        await gen.aclose()  # disconnect while that error event was in flight — must not raise

        # The backend failure already happened before the disconnect, so
        # it's still attributed to error/INFERENCE_ERRORS (a documented
        # tie-break), not silently dropped or double-counted as cancelled.
        assert STREAM_REQUESTS.labels(status="error")._value.get() == before_error + 1
        assert (
            INFERENCE_ERRORS.labels(error_type="RuntimeError")._value.get()
            == before_inference_errors + 1
        )

    @pytest.mark.asyncio
    async def test_cancelled_stream_still_counts_last_delivered_chunk_tokens(self, monkeypatch):
        """A chunk that was already handed to the SSE transport before the
        disconnect happened must count toward STREAM_TOKENS/TOKENS_GENERATED
        even though the generator never resumes past that yield."""
        import main as main_module

        backend = _FakeStreamBackend(pieces=[
            {"delta": "abc", "num_tokens": 3},
        ])
        monkeypatch.setattr(main_module, "engine", _FakeStreamEngine(backend))

        before_stream_tokens = STREAM_TOKENS._value.get()
        before_global_tokens = TOKENS_GENERATED._value.get()

        request = main_module.ChatCompletionRequest(
            model="m",
            messages=[main_module.ChatMessage(role="user", content="hi")],
            max_tokens=10,
            stream=True,
        )
        gen = main_module._stream_chat_completion("prompt", request)
        await gen.__anext__()  # role chunk
        await gen.__anext__()  # the one content chunk (num_tokens=3)
        await gen.aclose()  # disconnect right after it was sent

        assert STREAM_TOKENS._value.get() == before_stream_tokens + 3
        assert TOKENS_GENERATED._value.get() == before_global_tokens + 3

    @pytest.mark.asyncio
    async def test_tpot_is_token_weighted_not_chunk_averaged(self, monkeypatch):
        """A single delta can carry more than one token (real vLLM chunks
        aren't always 1 token). TPOT must be decode_time / decode_tokens,
        not decode_time / num_chunks. Uses real elapsed time (via asyncio.sleep)
        rather than a monkeypatched clock, since patching the global
        time.monotonic would also corrupt asyncio's and pytest's own timing.
        """
        import main as main_module

        backend = _FakeStreamBackend(
            pieces=[
                {"delta": "ab", "num_tokens": 3},   # first content delta: TTFT + baseline, no decode contribution
                {"delta": "cde", "num_tokens": 7},  # +4 tokens after a long delay
                {"delta": "f", "num_tokens": 8},    # +1 token after a short delay
            ],
            delays=[0, 0.12, 0.02],
        )
        monkeypatch.setattr(main_module, "engine", _FakeStreamEngine(backend))

        _, before_tpot_sum = _hist_count_sum(STREAM_TPOT)

        request = main_module.ChatCompletionRequest(
            model="m",
            messages=[main_module.ChatMessage(role="user", content="hi")],
            max_tokens=10,
            stream=True,
        )
        async for _ in main_module._stream_chat_completion("prompt", request):
            pass

        _, after_tpot_sum = _hist_count_sum(STREAM_TPOT)
        observed_tpot = after_tpot_sum - before_tpot_sum

        # decode_time ~= 0.12 + 0.02 = 0.14s; decode_tokens = 4 + 1 = 5
        # => token-weighted TPOT ~= 0.028s/token.
        # A naive chunk-time average ((0.12 + 0.02) / 2 chunks) would give
        # ~0.07s/chunk instead — clearly different from the correct value.
        # Generous bounds absorb scheduler jitter while still telling the
        # two computations apart.
        assert 0.01 < observed_tpot < 0.05

