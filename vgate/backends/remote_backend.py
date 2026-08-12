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
Remote inference backend.

Implements the same InferenceBackend protocol as VLLMBackend and SGLangBackend,
but forwards generation to a separate worker process over HTTP instead of
holding an engine in the gateway process. Because it satisfies the protocol,
RequestBatcher does not need to know whether inference is local or remote.

generate() is synchronous to match the protocol: RequestBatcher already calls
it through run_in_executor, so a blocking HTTP call there does not block the
event loop.
"""

import time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from vgate.config import ModelConfig, WorkerConfig
from vgate.logging_config import get_logger
from vgate.metrics import (
    WORKER_LATENCY, WORKER_REQUESTS, WORKER_RETRIES
)
from vgate.tracing import get_tracer
from vgate.worker_registry import NoHealthyWorkersError, WorkerRegistry

logger = get_logger("vgate.backends.remote")
tracer = get_tracer("vgate.backends.remote")


class RemoteInferenceError(RuntimeError):
    """Raised when a worker cannot be reached or returns an error response."""


class RemoteBackend:
    """Forwards inference to a remote V-Gate worker over HTTP."""

    # Unlike the in-process engines, concurrent calls are safe here: each call
    # is an independent HTTP request and the worker serializes as needed on its
    # own side. RequestBatcher reads this to decide whether to hold its
    # inference lock, which would otherwise serialize every worker call and
    # defeat the point of running more than one worker.
    supports_concurrent_calls = True

    # Streaming over a worker hop is not implemented yet. The gateway checks
    # this before opening an SSE response, so clients get a 501 instead of a
    # 200 followed by an in-band error event.
    supports_streaming = False

    def __init__(self, worker_config: WorkerConfig, registry: Optional[WorkerRegistry] = None):
        if not worker_config.endpoints:
            raise ValueError("RemoteBackend requires at least one worker endpoint")

        self.config = worker_config
        self.registry = registry or WorkerRegistry(
            worker_config.endpoints,
            failure_threshold=worker_config.failure_threshold,
            success_threshold=worker_config.success_threshold,
        )

        headers = {}
        if worker_config.api_key:
            headers["Authorization"] = f"Bearer {worker_config.api_key}"

        self._client = httpx.Client(
            timeout=httpx.Timeout(
                worker_config.timeout_seconds,
                connect=worker_config.connect_timeout_seconds,
            ),
            headers=headers,
        )
        logger.info(
            "Remote backend initialized",
            extra={"extra_data": {
                "endpoints": worker_config.endpoints,
                "timeout_seconds": worker_config.timeout_seconds,
                "authenticated": bool(worker_config.api_key),
            }}
        )

    def load_model(self, model_config: ModelConfig) -> None:
        """No-op: the worker process owns model loading."""

    def create_sampling_params(
        self, temperature: float, top_p: float, max_tokens: int
    ) -> Any:
        # Plain dict so it survives JSON transport; the worker rebuilds
        # engine-native params from it.
        return {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}

    def generate(
        self, prompts: List[str], sampling_params: Any
    ) -> List[Dict[str, Any]]:
        """
        Send a batch to a healthy worker, retrying on connection failures only.

        Retry policy is deliberately narrow, and keyed on what the worker could
        already have started doing:

        - ConnectError: nothing was delivered, so another worker can safely
          take the request.
        - Timeout: the worker may be mid-generation. Retrying would double GPU
          cost for one client request, so this fails instead.
        - Non-200: the worker received and rejected the request. Another worker
          would most likely reject it the same way.
        """
        with tracer.start_as_current_span("remote.generate") as span:
            span.set_attribute("num_prompts", len(prompts))

            tried: set = set()
            last_connect_error: Optional[Exception] = None

            # Bounded by the number of workers: each attempt excludes the ones
            # already tried, so this cannot spin.
            for attempt in range(len(self.registry.endpoints())):
                try:
                    endpoint = self.registry.pick(exclude=tried)
                except NoHealthyWorkersError:
                    break

                if attempt > 0:
                    WORKER_RETRIES.labels(worker=endpoint).inc()
                    span.set_attribute("retried", True)

                started = time.perf_counter()
                try:
                    response = self._client.post(
                        f"{endpoint}/internal/generate",
                        json={"prompts": prompts, "sampling_params": sampling_params},
                    )
                except httpx.ConnectError as exc:
                    # Never delivered — safe to move on to another worker.
                    self.registry.record_failure(endpoint)
                    WORKER_REQUESTS.labels(worker=endpoint, outcome="connect_error").inc()
                    tried.add(endpoint)
                    last_connect_error = exc
                    continue
                except httpx.RequestError as exc:
                    # Timeouts and read errors land here: the worker may already
                    # be generating, so this is not retried elsewhere.
                    self.registry.record_failure(endpoint)
                    WORKER_REQUESTS.labels(worker=endpoint, outcome="request_error").inc()
                    span.set_attribute("error", True)
                    raise RemoteInferenceError(
                        f"worker at {endpoint} failed mid-request: {type(exc).__name__}"
                    ) from exc
                finally:
                    WORKER_LATENCY.labels(worker=endpoint).observe(time.perf_counter() - started)

                if response.status_code != 200:
                    self.registry.record_failure(endpoint)
                    WORKER_REQUESTS.labels(worker=endpoint, outcome="http_error").inc()
                    span.set_attribute("error", True)
                    # Worker error bodies may echo prompt text; keep only a
                    # bounded prefix out of the exception message.
                    raise RemoteInferenceError(
                        f"worker at {endpoint} returned {response.status_code}: "
                        f"{response.text[:200]}"
                    )

                payload = response.json()
                results = payload.get("results")
                if not isinstance(results, list) or len(results) != len(prompts):
                    self.registry.record_failure(endpoint)
                    WORKER_REQUESTS.labels(worker=endpoint, outcome="bad_response").inc()
                    raise RemoteInferenceError(
                        f"worker at {endpoint} returned "
                        f"{len(results) if isinstance(results, list) else 'no'} "
                        f"results for {len(prompts)} prompts"
                    )

                self.registry.record_success(endpoint)
                WORKER_REQUESTS.labels(worker=endpoint, outcome="success").inc()
                span.set_attribute("endpoint", endpoint)
                return results

            span.set_attribute("error", True)
            detail = (
                f"last error: {type(last_connect_error).__name__}"
                if last_connect_error else "none reachable"
            )
            raise NoHealthyWorkersError(
                f"no healthy worker served the request after {len(tried)} attempt(s); {detail}"
            )

    async def stream_generate(
        self, prompt: str, sampling_params: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Not implemented yet.

        Streaming over a worker hop needs cancellation to propagate through
        two connections (client -> gateway -> worker); doing that correctly is
        its own change. Until then this fails loudly rather than silently
        degrading to a non-streaming response.
        """
        raise NotImplementedError(
            "Streaming is not supported with remote workers yet. "
            "Use a gateway with an in-process backend for streaming requests."
        )
        yield  # pragma: no cover  - makes this an async generator

    def shutdown(self) -> None:
        self._client.close()
