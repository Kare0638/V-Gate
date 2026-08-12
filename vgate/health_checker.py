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
    ):
        self.registry = registry
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
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
            while self._running:
                await asyncio.sleep(self.interval_seconds)
                if not self._running:
                    break
                await self.probe_once(client)

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
