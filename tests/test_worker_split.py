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
Tests for the gateway/worker split.

The end-to-end test wires a real RemoteBackend to a real worker app through an
httpx transport, so the request actually crosses the RemoteBackend ->
/internal/generate boundary rather than being mocked at the seam being tested.
"""

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vgate.backends.base import DryRunBackend, InferenceBackend
from vgate.backends.remote_backend import RemoteBackend, RemoteInferenceError
from vgate.batcher import RequestBatcher
from vgate.config import VGateConfig, WorkerConfig
from vgate.engine import _create_backend
from vgate import worker_api


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def test_role_defaults_to_gateway():
    assert VGateConfig().role == "gateway"


def test_role_rejects_unknown_value():
    with pytest.raises(ValueError, match="role must be one of"):
        VGateConfig(role="coordinator")


def test_worker_endpoints_require_http_scheme():
    with pytest.raises(ValueError, match="must start with http"):
        WorkerConfig(endpoints=["worker-1:8001"])


def test_worker_endpoints_strip_trailing_slash():
    cfg = WorkerConfig(endpoints=["http://worker-1:8001/"])
    assert cfg.endpoints == ["http://worker-1:8001"]


# --------------------------------------------------------------------------
# Backend factory
# --------------------------------------------------------------------------

def test_factory_returns_local_backend_without_endpoints():
    backend = _create_backend("vllm", WorkerConfig())
    assert isinstance(backend, DryRunBackend)  # VGATE_DRY_RUN=true in tests


def test_factory_returns_remote_backend_with_endpoints():
    backend = _create_backend("vllm", WorkerConfig(endpoints=["http://w:8001"]))
    assert isinstance(backend, RemoteBackend)
    backend.shutdown()


def test_remote_backend_satisfies_protocol():
    backend = RemoteBackend(WorkerConfig(endpoints=["http://w:8001"]))
    assert isinstance(backend, InferenceBackend)
    backend.shutdown()


def test_remote_backend_rejects_multiple_endpoints():
    # Multi-worker routing is not implemented yet; failing loudly beats
    # silently using only the first endpoint.
    with pytest.raises(ValueError, match="one worker endpoint"):
        RemoteBackend(WorkerConfig(endpoints=["http://a:8001", "http://b:8001"]))


def test_remote_backend_requires_an_endpoint():
    with pytest.raises(ValueError, match="at least one worker endpoint"):
        RemoteBackend(WorkerConfig())


# --------------------------------------------------------------------------
# Worker app
# --------------------------------------------------------------------------

def _make_worker_app() -> FastAPI:
    """A minimal worker app: just the internal router bound to a dry-run engine."""
    class _Engine:
        backend = DryRunBackend()

    app = FastAPI()
    app.include_router(worker_api.router)
    worker_api.set_engine(_Engine())
    return app


def test_worker_generate_returns_backend_shaped_results():
    with TestClient(_make_worker_app()) as client:
        response = client.post(
            "/internal/generate",
            json={
                "prompts": ["hello", "world"],
                "sampling_params": {"temperature": 0.1, "top_p": 0.9, "max_tokens": 16},
            },
        )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    # Same keys the InferenceBackend protocol promises, so RemoteBackend can
    # hand them straight back to the batcher.
    for result in results:
        assert set(result) >= {"text", "num_tokens", "metrics"}


def test_worker_rejects_empty_prompts():
    with TestClient(_make_worker_app()) as client:
        response = client.post("/internal/generate", json={"prompts": []})
    assert response.status_code == 422


def test_worker_returns_503_before_engine_is_bound():
    app = FastAPI()
    app.include_router(worker_api.router)
    worker_api._engine = None
    try:
        with TestClient(app) as client:
            response = client.post("/internal/generate", json={"prompts": ["x"]})
        assert response.status_code == 503
    finally:
        worker_api.set_engine(type("E", (), {"backend": DryRunBackend()})())


# --------------------------------------------------------------------------
# End-to-end: RemoteBackend -> worker app
# --------------------------------------------------------------------------

def _remote_backend_wired_to(app: FastAPI, **kwargs) -> RemoteBackend:
    """
    RemoteBackend whose HTTP client drives `app` in-process.

    TestClient subclasses httpx.Client and drives an ASGI app through a
    synchronous transport, which is what RemoteBackend.generate() needs --
    httpx.ASGITransport is async-only and cannot back a sync Client.
    """
    backend = RemoteBackend(WorkerConfig(endpoints=["http://testserver"], **kwargs))
    configured_headers = dict(backend._client.headers)
    backend._client.close()

    test_client = TestClient(app)
    test_client.headers.update(configured_headers)
    backend._client = test_client
    return backend


def test_remote_backend_forwards_generate_to_worker():
    backend = _remote_backend_wired_to(_make_worker_app())
    try:
        params = backend.create_sampling_params(temperature=0.5, top_p=0.9, max_tokens=8)
        results = backend.generate(["ping"], params)
    finally:
        backend.shutdown()

    assert len(results) == 1
    assert "text" in results[0]
    assert results[0]["num_tokens"] == 8


def test_remote_backend_raises_when_worker_unreachable():
    backend = RemoteBackend(
        WorkerConfig(endpoints=["http://127.0.0.1:1"], connect_timeout_seconds=0.1)
    )
    try:
        with pytest.raises(RemoteInferenceError, match="unreachable"):
            backend.generate(["ping"], {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4})
    finally:
        backend.shutdown()


def test_remote_backend_raises_on_worker_error_status():
    app = FastAPI()

    @app.post("/internal/generate")
    async def _boom():
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"detail": "Inference failed"})

    backend = _remote_backend_wired_to(app)
    try:
        with pytest.raises(RemoteInferenceError, match="returned 500"):
            backend.generate(["ping"], {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4})
    finally:
        backend.shutdown()


def test_remote_backend_rejects_result_count_mismatch():
    app = FastAPI()

    @app.post("/internal/generate")
    async def _short():
        return {"results": [{"text": "only one", "num_tokens": 1, "metrics": {}}]}

    backend = _remote_backend_wired_to(app)
    try:
        with pytest.raises(RemoteInferenceError, match="for 2 prompts"):
            backend.generate(["a", "b"], {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4})
    finally:
        backend.shutdown()


def test_remote_backend_sends_bearer_token_when_configured():
    seen = {}
    app = FastAPI()

    @app.post("/internal/generate")
    async def _echo():
        return {"results": [{"text": "ok", "num_tokens": 1, "metrics": {}}]}

    @app.middleware("http")
    async def _capture(request, call_next):
        seen["auth"] = request.headers.get("Authorization")
        return await call_next(request)

    backend = _remote_backend_wired_to(app, api_key="sk-worker-test")
    try:
        backend.generate(["x"], {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4})
    finally:
        backend.shutdown()

    assert seen["auth"] == "Bearer sk-worker-test"


def test_gateway_returns_501_for_streaming_with_remote_backend(monkeypatch):
    """
    The 501 must land before any SSE byte is written.

    A 200 followed by an in-band error event reads to a client as "succeeded,
    then broke mid-stream", which is wrong when the backend never supported
    streaming at all.
    """
    import main as main_module

    backend = RemoteBackend(WorkerConfig(endpoints=["http://w:8001"]))
    try:
        with TestClient(main_module.app) as client:
            # Patch inside the context: the app lifespan builds a fresh engine
            # on startup and would overwrite an earlier patch.
            monkeypatch.setattr(main_module, "engine", _Engine(backend))
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
    finally:
        backend.shutdown()

    assert response.status_code == 501
    assert "streaming is not supported" in response.json()["detail"].lower()
    assert "text/event-stream" not in response.headers.get("content-type", "")


def test_gateway_allows_streaming_with_local_backend(monkeypatch):
    """The 501 guard must not fire for backends that do support streaming."""
    import main as main_module

    with TestClient(main_module.app) as client:
        monkeypatch.setattr(main_module, "engine", _Engine(DryRunBackend()))
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "max_tokens": 4,
            },
        )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_remote_backend_streaming_raises_not_implemented():
    backend = RemoteBackend(WorkerConfig(endpoints=["http://w:8001"]))
    try:
        with pytest.raises(NotImplementedError, match="Streaming is not supported"):
            async for _ in backend.stream_generate("hi", {"max_tokens": 4}):
                pass
    finally:
        backend.shutdown()


# --------------------------------------------------------------------------
# Batcher lock behavior
# --------------------------------------------------------------------------

class _Engine:
    def __init__(self, backend):
        self.backend = backend


def test_batcher_serializes_local_backends():
    batcher = RequestBatcher(engine=_Engine(DryRunBackend()))
    assert batcher._serialize_inference is True


def test_batcher_does_not_serialize_concurrent_safe_backends():
    # Otherwise every worker call would queue behind the previous one and
    # adding workers would not increase throughput.
    backend = RemoteBackend(WorkerConfig(endpoints=["http://w:8001"]))
    try:
        batcher = RequestBatcher(engine=_Engine(backend))
        assert batcher._serialize_inference is False
    finally:
        backend.shutdown()


@pytest.mark.asyncio
async def test_concurrent_batches_overlap_with_remote_backend():
    """Two batches should be in flight at once when the backend allows it."""
    concurrent = 0
    peak = 0

    class _SlowConcurrentBackend:
        supports_concurrent_calls = True

        def create_sampling_params(self, temperature, top_p, max_tokens):
            return {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}

        def generate(self, prompts, sampling_params):
            nonlocal concurrent, peak
            concurrent += 1
            peak = max(peak, concurrent)
            import time as _t
            _t.sleep(0.05)
            concurrent -= 1
            return [{"text": p, "token_ids": [], "num_tokens": 1, "metrics": {}} for p in prompts]

    batcher = RequestBatcher(engine=_Engine(_SlowConcurrentBackend()), max_batch_size=1)
    await batcher.start()
    try:
        await asyncio.gather(*(batcher.submit(f"p{i}", max_tokens=4) for i in range(4)))
    finally:
        await batcher.stop()

    assert peak > 1, f"expected overlapping batches, peak concurrency was {peak}"
