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

"""Tests for worker registry, health probing, and routing/retry behavior."""

import asyncio
import threading

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vgate.backends.remote_backend import RemoteBackend, RemoteInferenceError
from vgate.config import WorkerConfig
from vgate.health_checker import WorkerHealthChecker
from vgate.worker_registry import NoHealthyWorkersError, WorkerRegistry

THREE = ["http://a:8001", "http://b:8001", "http://c:8001"]


# --------------------------------------------------------------------------
# Registry: selection
# --------------------------------------------------------------------------

def test_registry_requires_endpoints():
    with pytest.raises(ValueError, match="at least one endpoint"):
        WorkerRegistry([])


def test_pick_cycles_round_robin():
    reg = WorkerRegistry(THREE)
    assert [reg.pick() for _ in range(6)] == THREE + THREE


def test_pick_skips_unhealthy():
    reg = WorkerRegistry(THREE, failure_threshold=1)
    reg.record_failure("http://b:8001")
    picks = {reg.pick() for _ in range(6)}
    assert picks == {"http://a:8001", "http://c:8001"}


def test_pick_honors_exclude():
    reg = WorkerRegistry(THREE)
    first = reg.pick()
    second = reg.pick(exclude={first})
    assert second != first


def test_pick_raises_when_all_unhealthy():
    reg = WorkerRegistry(THREE, failure_threshold=1)
    for ep in THREE:
        reg.record_failure(ep)
    assert reg.has_healthy() is False
    with pytest.raises(NoHealthyWorkersError):
        reg.pick()


def test_pick_raises_when_all_excluded():
    reg = WorkerRegistry(THREE)
    with pytest.raises(NoHealthyWorkersError):
        reg.pick(exclude=set(THREE))


# --------------------------------------------------------------------------
# Registry: health transitions
# --------------------------------------------------------------------------

def test_failure_threshold_tolerates_a_single_blip():
    reg = WorkerRegistry(THREE, failure_threshold=2)
    reg.record_failure("http://a:8001")
    assert "http://a:8001" in reg.healthy_endpoints()
    reg.record_failure("http://a:8001")
    assert "http://a:8001" not in reg.healthy_endpoints()


def test_success_resets_failure_streak():
    reg = WorkerRegistry(THREE, failure_threshold=2)
    reg.record_failure("http://a:8001")
    reg.record_success("http://a:8001")
    reg.record_failure("http://a:8001")
    # The streak restarted, so one more failure should not be enough.
    assert "http://a:8001" in reg.healthy_endpoints()


def test_recovery_requires_sustained_success():
    reg = WorkerRegistry(THREE, failure_threshold=1, success_threshold=2)
    reg.record_failure("http://a:8001")
    assert "http://a:8001" not in reg.healthy_endpoints()

    reg.record_success("http://a:8001")
    assert "http://a:8001" not in reg.healthy_endpoints(), "one probe should not restore"

    reg.record_success("http://a:8001")
    assert "http://a:8001" in reg.healthy_endpoints()


def test_failure_during_recovery_restarts_the_count():
    reg = WorkerRegistry(THREE, failure_threshold=1, success_threshold=2)
    reg.record_failure("http://a:8001")
    reg.record_success("http://a:8001")
    reg.record_failure("http://a:8001")
    reg.record_success("http://a:8001")
    assert "http://a:8001" not in reg.healthy_endpoints()


def test_unknown_endpoint_is_ignored():
    reg = WorkerRegistry(THREE)
    reg.record_failure("http://not-registered:9999")
    reg.record_success("http://not-registered:9999")
    assert reg.healthy_endpoints() == THREE


def test_snapshot_reports_per_worker_state():
    reg = WorkerRegistry(THREE, failure_threshold=1)
    reg.record_failure("http://b:8001")
    snap = {s["endpoint"]: s for s in reg.snapshot()}
    assert snap["http://b:8001"]["healthy"] is False
    assert snap["http://b:8001"]["total_failures"] == 1
    assert snap["http://a:8001"]["healthy"] is True


def test_registry_is_thread_safe():
    """pick() runs in the batcher thread pool while probes run on the loop."""
    reg = WorkerRegistry(THREE)
    errors = []

    def hammer():
        try:
            for _ in range(500):
                reg.pick()
                reg.record_success("http://a:8001")
                reg.record_failure("http://b:8001")
        except Exception as exc:  # noqa: BLE001 - surfacing any race
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


# --------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------

def _backend_with_handler(handler, endpoints=None, **kwargs) -> RemoteBackend:
    """RemoteBackend whose transport is a scripted MockTransport."""
    cfg = WorkerConfig(endpoints=endpoints or ["http://a:8001", "http://b:8001"], **kwargs)
    backend = RemoteBackend(cfg)
    backend._client.close()
    backend._client = httpx.Client(transport=httpx.MockTransport(handler))
    return backend


OK_BODY = {"results": [{"text": "ok", "token_ids": [], "num_tokens": 1, "metrics": {}}]}


def test_connect_error_retries_on_another_worker():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if "a:8001" in str(request.url):
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=OK_BODY)

    backend = _backend_with_handler(handler)
    try:
        results = backend.generate(["x"], {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4})
    finally:
        backend.shutdown()

    assert results == OK_BODY["results"]
    assert len(seen) == 2, "should have tried the second worker"
    assert "b:8001" in seen[-1]


def test_timeout_does_not_retry():
    """A timed-out worker may already be generating; retrying doubles GPU cost."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        raise httpx.ReadTimeout("too slow", request=request)

    backend = _backend_with_handler(handler)
    try:
        with pytest.raises(RemoteInferenceError, match="mid-request"):
            backend.generate(["x"], {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4})
    finally:
        backend.shutdown()

    assert len(seen) == 1, "timeout must not be retried elsewhere"


def test_http_error_does_not_retry():
    """The worker received and rejected it; another worker would too."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(500, json={"detail": "boom"})

    backend = _backend_with_handler(handler)
    try:
        with pytest.raises(RemoteInferenceError, match="returned 500"):
            backend.generate(["x"], {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4})
    finally:
        backend.shutdown()

    assert len(seen) == 1


def test_all_workers_refusing_raises_no_healthy_workers():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    backend = _backend_with_handler(handler)
    try:
        with pytest.raises(NoHealthyWorkersError):
            backend.generate(["x"], {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4})
    finally:
        backend.shutdown()


def test_retry_count_is_bounded_by_worker_count():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        raise httpx.ConnectError("refused", request=request)

    backend = _backend_with_handler(handler, endpoints=THREE)
    try:
        with pytest.raises(NoHealthyWorkersError):
            backend.generate(["x"], {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4})
    finally:
        backend.shutdown()

    assert len(attempts) == 3, "each worker tried exactly once"


def test_success_marks_worker_healthy_again():
    state = {"fail": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["fail"]:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=OK_BODY)

    backend = _backend_with_handler(
        handler, endpoints=["http://a:8001"], failure_threshold=1, success_threshold=1
    )
    try:
        with pytest.raises(NoHealthyWorkersError):
            backend.generate(["x"], {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4})
        assert backend.registry.has_healthy() is False

        # Registry-level recovery, as the health checker would report it.
        state["fail"] = False
        backend.registry.record_success("http://a:8001")
        assert backend.registry.has_healthy() is True

        results = backend.generate(["x"], {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4})
        assert results == OK_BODY["results"]
    finally:
        backend.shutdown()


# --------------------------------------------------------------------------
# Health checker
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_probe_marks_failing_worker_unhealthy():
    reg = WorkerRegistry(["http://a:8001"], failure_threshold=1)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    checker = WorkerHealthChecker(reg, interval_seconds=0.01, timeout_seconds=0.1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await checker.probe_once(client)

    assert reg.has_healthy() is False


@pytest.mark.asyncio
async def test_probe_restores_recovered_worker():
    reg = WorkerRegistry(["http://a:8001"], failure_threshold=1, success_threshold=1)
    reg.record_failure("http://a:8001")
    assert reg.has_healthy() is False

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    checker = WorkerHealthChecker(reg, interval_seconds=0.01, timeout_seconds=0.1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await checker.probe_once(client)

    assert reg.has_healthy() is True


@pytest.mark.asyncio
async def test_probe_treats_connection_error_as_unhealthy():
    reg = WorkerRegistry(["http://a:8001"], failure_threshold=1)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    checker = WorkerHealthChecker(reg, interval_seconds=0.01, timeout_seconds=0.1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await checker.probe_once(client)

    assert reg.has_healthy() is False


@pytest.mark.asyncio
async def test_checker_loop_probes_and_stops_cleanly():
    """The polling loop runs against a mock transport, not the real network."""
    probes = []
    reg = WorkerRegistry(["http://a:8001"], failure_threshold=1)

    def handler(request: httpx.Request) -> httpx.Response:
        probes.append(str(request.url))
        return httpx.Response(500)

    checker = WorkerHealthChecker(
        reg,
        interval_seconds=0.01,
        timeout_seconds=0.1,
        transport=httpx.MockTransport(handler),
    )
    await checker.start()
    await asyncio.sleep(0.08)
    await checker.stop()

    assert checker._task is None
    assert probes, "loop should have probed at least once"
    assert reg.has_healthy() is False, "repeated 500s should remove the worker"


@pytest.mark.asyncio
async def test_checker_sends_api_key():
    seen = {}
    reg = WorkerRegistry(["http://a:8001"])

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200)

    checker = WorkerHealthChecker(reg, api_key="sk-probe")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), headers=checker._headers
    ) as client:
        await checker.probe_once(client)

    assert seen["auth"] == "Bearer sk-probe"
