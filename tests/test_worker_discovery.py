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
Tests for DNS-based worker membership.

Resolvers are injected throughout. The point of these tests is the behaviour
around membership changes -- what survives a re-resolve, what does not, and
what the system does when DNS misbehaves -- none of which needs a real
resolver, and all of which would be untestable if it did.
"""

import asyncio
import socket

import pytest

from vgate.health_checker import WorkerHealthChecker
from vgate.worker_discovery import DnsWorkerDiscovery
from vgate.worker_registry import WorkerRegistry

SVC = "vgate-worker.vgate.svc.cluster.local"


def make_discovery(addresses, ptr=None, **kwargs):
    """Discovery wired to fixed answers. `ptr` maps address -> pod name."""
    ptr = ptr or {}

    def forward(host, port):
        if host != SVC:
            raise socket.gaierror(-2, "Name or service not known")
        return list(addresses)

    def reverse(address):
        if address not in ptr:
            raise socket.herror(1, "Unknown host")
        return ptr[address]

    return DnsWorkerDiscovery(
        SVC, forward_resolver=forward, reverse_resolver=reverse, **kwargs
    )


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

def test_resolves_pods_to_their_stable_names():
    """
    The observed Kubernetes behaviour: forward gives addresses, reverse gives
    the ordinal-stable pod name. The name is what must end up in the endpoint,
    because it is what survives a pod restart.
    """
    d = make_discovery(
        ["10.244.1.4", "10.244.2.3"],
        ptr={
            "10.244.1.4": "vgate-worker-0.vgate-worker.vgate.svc.cluster.local",
            "10.244.2.3": "vgate-worker-1.vgate-worker.vgate.svc.cluster.local",
        },
    )
    assert d.resolve() == [
        "http://vgate-worker-0.vgate-worker.vgate.svc.cluster.local:8000",
        "http://vgate-worker-1.vgate-worker.vgate.svc.cluster.local:8000",
    ]


def test_falls_back_to_the_address_when_reverse_lookup_fails():
    """A cluster with PTR records disabled still works, just with IP identity."""
    d = make_discovery(["10.244.1.4"], ptr={})
    assert d.resolve() == ["http://10.244.1.4:8000"]


def test_unresolvable_name_is_empty_not_an_error():
    """
    A name that does not resolve is the normal state before any worker pod is
    scheduled. Raising here would make gateway startup depend on worker
    scheduling order.
    """
    d = make_discovery([])

    def forward(host, port):
        raise socket.gaierror(-2, "Name or service not known")

    d._forward = forward
    assert d.resolve() == []


def test_ptr_pointing_at_the_service_is_not_treated_as_identity():
    """
    A PTR answering with the Service name gives every pod the same identity,
    which would collapse the pool into one endpoint. That is worse than using
    addresses, so it is rejected.
    """
    d = make_discovery(
        ["10.244.1.4", "10.244.2.3"],
        ptr={"10.244.1.4": SVC, "10.244.2.3": SVC},
    )
    assert d.resolve() == ["http://10.244.1.4:8000", "http://10.244.2.3:8000"]


def test_scheme_and_port_are_configurable():
    d = make_discovery(["10.0.0.1"], ptr={}, port=9000, scheme="https")
    assert d.resolve() == ["https://10.0.0.1:9000"]


def test_ipv6_addresses_are_bracketed():
    """
    `http://fd00::1:8000` cannot be parsed -- the colons in the address are
    indistinguishable from the port separator. Only the fallback path is
    affected, since resolved names never need brackets.
    """
    d = make_discovery(["fd00::1", "fd00::2"], ptr={})
    assert d.resolve() == ["http://[fd00::1]:8000", "http://[fd00::2]:8000"]


def test_ipv6_pod_names_are_not_bracketed():
    d = make_discovery(["fd00::1"], ptr={"fd00::1": "vgate-worker-0.vgate-worker"})
    assert d.resolve() == ["http://vgate-worker-0.vgate-worker:8000"]


def test_forward_resolution_is_not_restricted_to_ipv4():
    """
    An IPv6-only cluster publishes AAAA records and no A records. Pinning the
    lookup to AF_INET would resolve nothing there and leave the pool
    permanently empty with no error to explain it.
    """
    from vgate import worker_discovery

    seen = {}

    def fake_getaddrinfo(host, port, family, socktype):
        seen["family"] = family
        return [(family, socktype, 6, "", ("fd00::1", port, 0, 0))]

    original = worker_discovery.socket.getaddrinfo
    worker_discovery.socket.getaddrinfo = fake_getaddrinfo
    try:
        assert worker_discovery._default_forward("svc", 8000) == ["fd00::1"]
    finally:
        worker_discovery.socket.getaddrinfo = original

    assert seen["family"] == socket.AF_UNSPEC


# --------------------------------------------------------------------------
# Registry membership
# --------------------------------------------------------------------------

def test_set_members_adds_and_removes():
    r = WorkerRegistry(["http://a:8000"])
    added, removed = r.set_members(["http://a:8000", "http://b:8000"])
    assert added == ["http://b:8000"] and removed == []
    assert r.endpoints() == ["http://a:8000", "http://b:8000"]

    added, removed = r.set_members(["http://b:8000"])
    assert added == [] and removed == ["http://a:8000"]
    assert r.endpoints() == ["http://b:8000"]


def test_rediscovery_does_not_reset_a_worker_being_demoted():
    """
    The failure this guards against: membership refreshes every few seconds,
    and a worker needs consecutive failures to leave rotation. If a refresh
    reset that counter, a broken worker would be probed forever and never
    demoted, because the count would restart before reaching the threshold.
    """
    r = WorkerRegistry(["http://a:8000", "http://b:8000"], failure_threshold=3)
    r.record_failure("http://a:8000")
    r.record_failure("http://a:8000")

    r.set_members(["http://a:8000", "http://b:8000", "http://c:8000"])

    r.record_failure("http://a:8000")  # third strike -- must demote
    assert "http://a:8000" not in r.healthy_endpoints()


def test_a_removed_worker_is_no_longer_picked_or_probed():
    r = WorkerRegistry(["http://a:8000", "http://b:8000"])
    r.set_members(["http://b:8000"])
    assert r.endpoints() == ["http://b:8000"]
    assert {r.pick() for _ in range(4)} == {"http://b:8000"}


def test_a_returning_worker_starts_healthy():
    """
    Arrivals are optimistic, matching how configured endpoints are treated.
    A worker that left while unhealthy and comes back must not inherit the old
    verdict, or a replaced pod would start out excluded.
    """
    r = WorkerRegistry(["http://a:8000", "http://b:8000"], failure_threshold=1)
    r.record_failure("http://a:8000")
    assert "http://a:8000" not in r.healthy_endpoints()

    r.set_members(["http://b:8000"])
    r.set_members(["http://a:8000", "http://b:8000"])
    assert "http://a:8000" in r.healthy_endpoints()


def test_round_robin_cycles_through_the_new_member_set():
    """
    After a membership change, selection must use the new set and keep
    rotating through it -- not keep serving the old one, and not stick to a
    single worker.

    An earlier version of this test claimed to check that the round-robin
    cursor is renormalized when the pool shrinks. It passed against an
    implementation that did no renormalization at all, because pick() already
    reduces the cursor modulo the list length. The normalization was removed
    and this checks the property that actually matters instead.
    """
    r = WorkerRegistry([f"http://{n}:8000" for n in "abcd"])
    for _ in range(3):
        r.pick()

    r.set_members(["http://b:8000", "http://c:8000"])
    picks = [r.pick() for _ in range(4)]

    assert set(picks) == {"http://b:8000", "http://c:8000"}
    assert picks[0] != picks[1], "round-robin stopped rotating after a membership change"


def test_discovered_registry_may_start_empty():
    """A gateway must not require its workers to be scheduled first."""
    r = WorkerRegistry([], allow_empty=True)
    assert r.endpoints() == []
    r.set_members(["http://a:8000"])
    assert r.healthy_endpoints() == ["http://a:8000"]


def test_static_registry_still_rejects_an_empty_endpoint_list():
    with pytest.raises(ValueError, match="at least one endpoint"):
        WorkerRegistry([])


# --------------------------------------------------------------------------
# Health checker integration
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refresh_applies_discovered_membership():
    r = WorkerRegistry([], allow_empty=True)
    checker = WorkerHealthChecker(
        registry=r,
        discovery=make_discovery(
            ["10.0.0.1", "10.0.0.2"],
            ptr={"10.0.0.1": "w-0.svc", "10.0.0.2": "w-1.svc"},
        ),
    )
    await checker.refresh_members()
    assert r.endpoints() == ["http://w-0.svc:8000", "http://w-1.svc:8000"]


@pytest.mark.asyncio
async def test_one_empty_resolve_does_not_empty_a_populated_registry():
    """
    A single empty answer is more likely a resolver blip than every worker
    being deleted between two ticks. Acting on it immediately would turn a DNS
    hiccup into a total outage.
    """
    r = WorkerRegistry(["http://a:8000"])
    checker = WorkerHealthChecker(registry=r, discovery=make_discovery([]))
    await checker.refresh_members()
    assert r.endpoints() == ["http://a:8000"]


@pytest.mark.asyncio
async def test_sustained_empty_resolves_do_empty_the_registry():
    """
    Scaling the pool to zero. Without this, the guard against blips keeps the
    last worker in the registry forever and the gateway probes a pod that has
    been deleted -- reintroducing the exact failure discovery exists to remove.

    The distinction between a blip and a real scale-to-zero is that the real
    one keeps saying the same thing.
    """
    r = WorkerRegistry(["http://a:8000"])
    checker = WorkerHealthChecker(
        registry=r, discovery=make_discovery([]), empty_resolve_threshold=3
    )
    for _ in range(2):
        await checker.refresh_members()
        assert r.endpoints() == ["http://a:8000"], "gave up before the threshold"

    await checker.refresh_members()
    assert r.endpoints() == [], "scale-to-zero was never believed"


@pytest.mark.asyncio
async def test_a_successful_resolve_resets_the_empty_streak():
    """
    Intermittent blips must not accumulate across minutes into a false
    scale-to-zero. Only *consecutive* empties count.
    """
    addresses = ["10.0.0.1"]
    discovery = make_discovery(addresses, ptr={"10.0.0.1": "w-0.svc"})
    r = WorkerRegistry(["http://w-0.svc:8000"])
    checker = WorkerHealthChecker(
        registry=r, discovery=discovery, empty_resolve_threshold=3
    )

    empty, full = [], list(addresses)
    for answer in (empty, empty, full, empty, empty):
        addresses[:] = answer
        await checker.refresh_members()

    # Four empties total, but never three in a row.
    assert r.endpoints() == ["http://w-0.svc:8000"]


@pytest.mark.asyncio
async def test_a_resolver_exception_does_not_kill_the_loop():
    """A raising resolver must not take the health checker down with it."""
    r = WorkerRegistry(["http://a:8000"])
    discovery = make_discovery(["10.0.0.1"], ptr={})

    def boom(host, port):
        raise RuntimeError("resolver exploded")

    discovery._forward = boom
    checker = WorkerHealthChecker(registry=r, discovery=discovery)
    await checker.refresh_members()  # must not raise
    assert r.endpoints() == ["http://a:8000"]


@pytest.mark.asyncio
async def test_start_resolves_before_returning():
    """
    Startup must not complete with an empty pool. The first resolve used to
    happen inside the background loop, so `await start()` returned immediately
    and the gateway accepted requests with nothing to route them to, answering
    503 for as long as resolution took.
    """
    r = WorkerRegistry([], allow_empty=True)
    checker = WorkerHealthChecker(
        registry=r,
        interval_seconds=3600,  # the loop must not be what fills this in
        discovery=make_discovery(["10.0.0.1"], ptr={"10.0.0.1": "w-0.svc"}),
    )
    try:
        await checker.start()
        assert r.endpoints() == ["http://w-0.svc:8000"]
    finally:
        await checker.stop()


@pytest.mark.asyncio
async def test_start_does_not_hang_on_a_stalled_resolver():
    """
    Resolving before returning must not make startup depend on DNS answering.
    A resolver that never responds should delay the first request, not the
    process.
    """
    import time

    # Long enough that the 0.2s timeout below cannot pass by luck, short
    # enough that the executor thread does not hold up interpreter shutdown --
    # `wait_for` abandons the wait, it cannot cancel the thread.
    def never_answers(host, port):
        time.sleep(2)
        return []

    discovery = make_discovery([])
    discovery._forward = never_answers
    r = WorkerRegistry(["http://a:8000"])
    checker = WorkerHealthChecker(
        registry=r, discovery=discovery, startup_resolve_timeout=0.2
    )
    try:
        started = asyncio.get_running_loop().time()
        await checker.start()
        assert asyncio.get_running_loop().time() - started < 5
        assert r.endpoints() == ["http://a:8000"]
    finally:
        await checker.stop()


@pytest.mark.asyncio
async def test_refresh_is_a_no_op_without_discovery():
    r = WorkerRegistry(["http://a:8000"])
    checker = WorkerHealthChecker(registry=r, discovery=None)
    await checker.refresh_members()
    assert r.endpoints() == ["http://a:8000"]
