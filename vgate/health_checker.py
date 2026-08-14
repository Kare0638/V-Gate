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
from vgate.worker_discovery import DnsWorkerDiscovery
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
    ):
        self.registry = registry
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        # Membership refresh rides this loop rather than running its own task:
        # discovery and probing answer adjacent questions on the same cadence,
        # and a second timer would only add a way for the two to disagree about
        # which workers exist.
        self.discovery = discovery
        # Injectable so the polling loop can be exercised without real network
        # calls; None means httpx picks its default transport.
        self._transport = transport
        self._task: Optional[asyncio.Task] = None
        self._running = False

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._headers = headers

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
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
            # Resolve before the first sleep so a discovering gateway has a
            # pool by the time the first request arrives, rather than answering
            # 503 for one interval after startup.
            await self.refresh_members()
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
        except Exception as exc:  # resolver failures must not kill the loop
            logger.warning(
                "Worker discovery failed; keeping the current member set",
                extra={"extra_data": {"error": str(exc), "error_type": type(exc).__name__}},
            )
            return

        if not endpoints and self.registry.endpoints():
            # Every name disappearing at once is far more likely to be a DNS
            # hiccup than every worker being deleted between two ticks.
            # Emptying the registry on that reading would turn a resolver blip
            # into a total outage, and the health checker already handles
            # workers that stop answering.
            logger.warning(
                "Worker discovery returned no endpoints; keeping the current member set",
                extra={"extra_data": {"current": self.registry.endpoints()}},
            )
            return

        self.registry.set_members(endpoints)

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
