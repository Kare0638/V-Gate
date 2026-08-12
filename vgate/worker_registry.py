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
Worker registry and health tracking.

State here is touched from two places at once: the background health-check
task on the event loop, and RemoteBackend.generate() running in the batcher's
thread pool. That rules out asyncio.Lock -- a threading.Lock is what actually
protects the state across both.

Membership is static (read from config). This registry tracks which of those
known workers are currently usable, not which workers exist.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from vgate.logging_config import get_logger
from vgate.metrics import WORKER_HEALTHY, WORKER_STATE_CHANGES

logger = get_logger("vgate.registry")


@dataclass
class WorkerState:
    """Health bookkeeping for a single worker endpoint."""

    endpoint: str
    healthy: bool = True
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_change_at: float = field(default_factory=time.monotonic)
    total_failures: int = 0


class NoHealthyWorkersError(RuntimeError):
    """Raised when every known worker is currently marked unhealthy."""


class WorkerRegistry:
    """
    Tracks health of a fixed set of worker endpoints and picks one per request.

    Workers start optimistically healthy so a gateway can serve traffic before
    the first health probe completes; a worker that is actually down is
    demoted by the first failed request or probe.
    """

    def __init__(
        self,
        endpoints: List[str],
        failure_threshold: int = 2,
        success_threshold: int = 2,
    ):
        if not endpoints:
            raise ValueError("WorkerRegistry requires at least one endpoint")

        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold

        self._lock = threading.Lock()
        self._states = {ep: WorkerState(endpoint=ep) for ep in endpoints}
        self._order = list(endpoints)
        self._cursor = 0

        for ep in endpoints:
            WORKER_HEALTHY.labels(worker=ep).set(1)

    # -- selection ---------------------------------------------------------

    def pick(self, exclude: Optional[set] = None) -> str:
        """
        Return the next healthy endpoint, round-robin.

        `exclude` lets a caller skip endpoints it has already tried within one
        request, so a retry lands on a different worker instead of the one
        that just refused the connection.
        """
        exclude = exclude or set()
        with self._lock:
            total = len(self._order)
            for offset in range(total):
                endpoint = self._order[(self._cursor + offset) % total]
                if endpoint in exclude:
                    continue
                if self._states[endpoint].healthy:
                    # Advance past the chosen worker so the next call starts
                    # at the following one.
                    self._cursor = (self._cursor + offset + 1) % total
                    return endpoint

        raise NoHealthyWorkersError(
            f"no healthy worker available ({len(exclude)} excluded this request)"
        )

    # -- health transitions ------------------------------------------------

    def record_failure(self, endpoint: str) -> None:
        """
        Count a failed interaction, demoting the worker at the threshold.

        A threshold above 1 keeps one transient blip from taking a worker out
        of rotation.
        """
        with self._lock:
            state = self._states.get(endpoint)
            if state is None:
                return
            state.total_failures += 1
            state.consecutive_successes = 0
            state.consecutive_failures += 1
            if state.healthy and state.consecutive_failures >= self.failure_threshold:
                state.healthy = False
                state.last_change_at = time.monotonic()
                self._on_change(endpoint, healthy=False)

    def record_success(self, endpoint: str) -> None:
        """Count a successful interaction, restoring the worker at the threshold."""
        with self._lock:
            state = self._states.get(endpoint)
            if state is None:
                return
            state.consecutive_failures = 0
            if state.healthy:
                state.consecutive_successes = 0
                return
            state.consecutive_successes += 1
            if state.consecutive_successes >= self.success_threshold:
                state.healthy = True
                state.last_change_at = time.monotonic()
                self._on_change(endpoint, healthy=True)

    def _on_change(self, endpoint: str, healthy: bool) -> None:
        """Emit metrics and a log line for a health transition. Caller holds the lock."""
        WORKER_HEALTHY.labels(worker=endpoint).set(1 if healthy else 0)
        WORKER_STATE_CHANGES.labels(
            worker=endpoint, transition="recovered" if healthy else "removed"
        ).inc()
        logger.warning(
            "Worker %s" % ("recovered" if healthy else "removed from rotation"),
            extra={"extra_data": {"worker": endpoint, "healthy": healthy}},
        )

    # -- introspection -----------------------------------------------------

    def endpoints(self) -> List[str]:
        return list(self._order)

    def healthy_endpoints(self) -> List[str]:
        with self._lock:
            return [ep for ep in self._order if self._states[ep].healthy]

    def has_healthy(self) -> bool:
        with self._lock:
            return any(self._states[ep].healthy for ep in self._order)

    def snapshot(self) -> List[dict]:
        """Per-worker health for /stats. Bounded, no per-request detail."""
        with self._lock:
            now = time.monotonic()
            return [
                {
                    "endpoint": ep,
                    "healthy": s.healthy,
                    "consecutive_failures": s.consecutive_failures,
                    "total_failures": s.total_failures,
                    "seconds_in_state": round(now - s.last_change_at, 1),
                }
                for ep, s in ((e, self._states[e]) for e in self._order)
            ]
