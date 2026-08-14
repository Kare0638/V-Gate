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
Background health probing for worker endpoints.

Probing runs alongside the failure signal that RemoteBackend records from real
requests. Both feed the same registry, and they answer different questions:
request failures notice a dead worker immediately but only when traffic is
flowing, while probes notice recovery on an idle gateway, where no request
would ever retry the worker and let it back in.
"""

import asyncio
from typing import Optional

import httpx

from vgate.logging_config import get_logger
from vgate.worker_discovery import DnsWorkerDiscovery, TransientResolutionError
from vgate.worker_registry import WorkerRegistry

logger = get_logger("vgate.health")


class WorkerHealthChecker:
    """Polls each worker's /health and reports the result to the registry."""

    def __init__(
        self,
        registry: WorkerRegistry,
        interval_seconds: float = 5.0,
        timeout_seconds: float = 2.0,
        api_key: Optional[str] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        discovery: Optional[DnsWorkerDiscovery] = None,
        empty_resolve_threshold: int = 3,
        startup_resolve_timeout: float = 5.0,
    ):
        self.registry = registry
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        # Membership refresh rides this loop rather than running its own task:
        # discovery and probing answer adjacent questions on the same cadence,
        # and a second timer would only add a way for the two to disagree about
        # which workers exist.
        self.discovery = discovery
        # How many consecutive empty resolves it takes to believe the pool is
        # really gone. At the default interval this is ~15s of agreement, which
        # a resolver blip does not survive and a scale-to-zero does.
        self.empty_resolve_threshold = max(1, empty_resolve_threshold)
        self._empty_resolves = 0
        self.startup_resolve_timeout = startup_resolve_timeout
        # Injectable so the polling loop can be exercised without real network
        # calls; None means httpx picks its default transport.
        self._transport = transport
        self._task: Optional[asyncio.Task] = None
        self._running = False
        # Created in start(), which is where a running loop exists to bind it.
        self._first_pass: Optional[asyncio.Event] = None

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._headers = headers

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        self._first_pass = asyncio.Event()
        self._task = asyncio.create_task(self._loop())

        # Wait for one full pass -- resolve, then probe -- before returning.
        #
        # Resolving alone is not enough now that discovered arrivals start out
        # of rotation: startup would complete with a pool that is known but
        # entirely unadmitted, and the gateway would answer 503 until the first
        # scheduled probe an interval later. Probing here admits the workers
        # that are actually up.
        #
        # Bounded, because startup must not hang on a resolver or a worker that
        # never answers. On timeout the loop continues on its normal cadence,
        # so an unreachable dependency delays the first request rather than the
        # process. (A gateway that is Ready while holding zero workers is a
        # separate gap, tracked in ROADMAP.md -- this narrows the window, it
        # does not close it.)
        if self.discovery is not None:
            try:
                await asyncio.wait_for(
                    self._first_pass.wait(), timeout=self.startup_resolve_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "First discovery and probe did not complete before startup; "
                    "continuing and retrying in the background",
                    extra={"extra_data": {
                        "timeout_seconds": self.startup_resolve_timeout,
                        "dns_name": self.discovery.dns_name,
                    }},
                )

        logger.info(
            "Worker health checker started",
            extra={"extra_data": {
                "workers": self.registry.endpoints(),
                "interval_seconds": self.interval_seconds,
            }}
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Worker health checker stopped")

    async def _loop(self) -> None:
        # A separate client from RemoteBackend's: probes must not queue behind
        # in-flight generate calls, which can legitimately take minutes.
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            headers=self._headers,
            transport=self._transport,
        ) as client:
            # One immediate pass before the first sleep, so a freshly started
            # gateway has an admitted pool rather than waiting a full interval
            # with every discovered worker still pending. start() waits on the
            # event this sets.
            await self.refresh_members()
            await self.probe_once(client)
            self._first_pass.set()

            while self._running:
                await asyncio.sleep(self.interval_seconds)
                if not self._running:
                    break
                await self.refresh_members()
                await self.probe_once(client)

    async def refresh_members(self) -> None:
        """Re-resolve worker membership, if discovery is configured."""
        if self.discovery is None:
            return
        # getaddrinfo blocks; on a slow or unreachable resolver it would stall
        # the event loop and with it every in-flight request on this gateway.
        loop = asyncio.get_running_loop()
        try:
            endpoints = await loop.run_in_executor(None, self.discovery.resolve)
        except TransientResolutionError as exc:
            # The resolver could not answer. That is not "there are no
            # workers", and the empty-resolve counter deliberately does not
            # advance here: counting it would let a DNS outage lasting a few
            # ticks discard a pool that never stopped working.
            logger.warning(
                "Worker discovery could not resolve; keeping the current "
                "member set and not counting this as empty",
                extra={"extra_data": {"error": str(exc)}},
            )
            return
        except Exception as exc:  # resolver failures must not kill the loop
            logger.warning(
                "Worker discovery failed; keeping the current member set",
                extra={"extra_data": {"error": str(exc), "error_type": type(exc).__name__}},
            )
            return

        if endpoints:
            self._empty_resolves = 0
            self.registry.set_members(endpoints)
            return

        if not self.registry.endpoints():
            return  # already empty; nothing to decide

        # An empty answer is ambiguous: a resolver blip and a pool scaled to
        # zero look identical in a single reading. Acting on the first one
        # would turn a DNS hiccup into a total outage; never acting on it
        # leaves the last worker in the registry forever, probed after the pod
        # it names has been deleted -- the exact failure discovery exists to
        # remove. Requiring the answer to repeat separates them: a blip does
        # not survive consecutive ticks, and a real scale-to-zero does.
        self._empty_resolves += 1
        if self._empty_resolves < self.empty_resolve_threshold:
            logger.warning(
                "Worker discovery returned no endpoints; keeping the current "
                "member set pending confirmation",
                extra={"extra_data": {
                    "current": self.registry.endpoints(),
                    "consecutive_empty": self._empty_resolves,
                    "threshold": self.empty_resolve_threshold,
                }},
            )
            return

        logger.warning(
            "Worker discovery returned no endpoints on %d consecutive ticks; "
            "treating the pool as empty" % self._empty_resolves,
            extra={"extra_data": {"removed": self.registry.endpoints()}},
        )
        self.registry.set_members([])

    async def probe_once(self, client: httpx.AsyncClient) -> None:
        """Probe every worker once, concurrently."""
        endpoints = self.registry.endpoints()
        results = await asyncio.gather(
            *(self._probe(client, ep) for ep in endpoints),
            return_exceptions=True,
        )
        for endpoint, healthy in zip(endpoints, results):
            # gather with return_exceptions can hand back an exception object;
            # anything that is not an explicit True counts as a failed probe.
            if healthy is True:
                self.registry.record_success(endpoint)
            else:
                self.registry.record_failure(endpoint)

    async def _probe(self, client: httpx.AsyncClient, endpoint: str) -> bool:
        try:
            response = await client.get(f"{endpoint}/health")
        except httpx.RequestError:
            return False
        return response.status_code == 200
