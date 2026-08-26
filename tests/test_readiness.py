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
Readiness, and specifically why it latches.

/health returned ok unconditionally, so Kubernetes marked a gateway Ready and
routed to it while it held zero healthy workers -- those requests could only be
answered with 503. That is a cold-start problem, and the fix is a startup gate.

It is deliberately NOT a continuous check on the worker pool. Every gateway
replica shares one pool, so failing readiness when the pool is empty makes all
of them unready at once, Kubernetes empties the Service, and clients get a
connection refused rather than a 503 with Retry-After -- less information, no
failover gained. These tests pin that distinction, because the obvious
implementation gets it wrong.
"""

import pytest

import main


@pytest.fixture(autouse=True)
def reset_latch():
    """Each test starts from a process that has never been ready."""
    main._has_been_ready = False
    yield
    main._has_been_ready = False


class FakeRegistry:
    def __init__(self, healthy, known=2):
        self._healthy = healthy
        self._known = known

    def has_healthy(self):
        return self._healthy

    def endpoints(self):
        return [f"http://w{i}:8000" for i in range(self._known)]


class FakeBackend:
    def __init__(self, healthy, known):
        self.registry = FakeRegistry(healthy, known)

    def shutdown(self):
        """Called by the app lifespan on a remote engine."""


class FakeRemoteEngine:
    is_remote = True

    def __init__(self, healthy, known=2):
        self.backend = FakeBackend(healthy, known)


class StubBatcher:
    """Only needs to be non-None here, plus survive lifespan shutdown."""

    async def start(self):
        pass

    async def stop(self):
        pass


def set_gateway(engine, batcher=None):
    batcher = batcher or StubBatcher()
    main.engine = engine
    main.batcher = batcher


def test_not_ready_before_the_batcher_starts():
    original_b, original_e = main.batcher, main.engine
    try:
        main.batcher = None
        ready, reason = main._readiness()
        assert not ready and "batcher" in reason
    finally:
        main.batcher, main.engine = original_b, original_e


def test_not_ready_while_the_pool_has_not_come_up():
    """The cold start this exists to close."""
    original_b, original_e = main.batcher, main.engine
    try:
        set_gateway(FakeRemoteEngine(healthy=False, known=0))
        ready, reason = main._readiness()
        assert not ready
        assert "no workers discovered" in reason
    finally:
        main.batcher, main.engine = original_b, original_e


def test_not_ready_while_known_workers_are_all_unusable():
    original_b, original_e = main.batcher, main.engine
    try:
        set_gateway(FakeRemoteEngine(healthy=False, known=3))
        ready, reason = main._readiness()
        assert not ready
        assert "3 known" in reason, reason
    finally:
        main.batcher, main.engine = original_b, original_e


def test_ready_once_a_worker_is_usable():
    original_b, original_e = main.batcher, main.engine
    try:
        set_gateway(FakeRemoteEngine(healthy=True))
        assert main._readiness()[0]
    finally:
        main.batcher, main.engine = original_b, original_e


def test_readiness_latches_when_the_pool_is_later_lost():
    """
    The design decision, and the one an obvious implementation gets wrong.

    Failing readiness here would take every gateway replica out of the Service
    simultaneously, since they share one pool. Clients would get a connection
    refused instead of 503 with Retry-After: strictly less information, and no
    replica is better placed to take over. Losing the pool is reported through
    the request path, where the answer can carry a retry hint.
    """
    original_b, original_e = main.batcher, main.engine
    try:
        engine = FakeRemoteEngine(healthy=True)
        set_gateway(engine)
        assert main._readiness()[0], "should be ready once a worker is usable"

        engine.backend.registry._healthy = False
        ready, _ = main._readiness()
        assert ready, (
            "readiness un-latched after the pool was lost; that empties the "
            "Service and downgrades a retryable 503 to a connection refused"
        )
    finally:
        main.batcher, main.engine = original_b, original_e


def test_an_in_process_gateway_is_ready_without_any_workers():
    """No pool to wait for when inference is local."""
    original_b, original_e = main.batcher, main.engine
    try:
        set_gateway(type("E", (), {"is_remote": False})())
        assert main._readiness()[0]
    finally:
        main.batcher, main.engine = original_b, original_e


def test_ready_is_exempt_from_authentication():
    """kubelet probes carry no bearer token, so a gated /ready never passes."""
    from vgate.config import SecurityConfig
    assert "/ready" in SecurityConfig().exempt_paths


def test_endpoint_returns_503_while_not_ready():
    from fastapi.testclient import TestClient

    original_b, original_e = main.batcher, main.engine
    try:
        with TestClient(main.app) as client:
            set_gateway(FakeRemoteEngine(healthy=False, known=1))
            main._has_been_ready = False
            response = client.get("/ready")
            assert response.status_code == 503, response.text
            assert response.json()["ready"] is False

            # /health must stay 200: liveness asks a different question, and a
            # process that is up but has no pool must not be restarted for it.
            assert client.get("/health").status_code == 200
    finally:
        main.batcher, main.engine = original_b, original_e
