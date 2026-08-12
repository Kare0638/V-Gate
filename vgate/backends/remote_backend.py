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

from typing import Any, AsyncIterator, Dict, List

import httpx

from vgate.config import ModelConfig, WorkerConfig
from vgate.logging_config import get_logger
from vgate.tracing import get_tracer

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

    def __init__(self, worker_config: WorkerConfig):
        if not worker_config.endpoints:
            raise ValueError("RemoteBackend requires at least one worker endpoint")

        self.config = worker_config
        # PR1 targets a single worker. Routing across several endpoints, with a
        # registry and health checks, is the next change; until then an extra
        # endpoint would be silently ignored, so refuse it explicitly.
        if len(worker_config.endpoints) > 1:
            raise ValueError(
                "RemoteBackend currently supports exactly one worker endpoint; "
                f"got {len(worker_config.endpoints)}. Multi-worker routing is not implemented yet."
            )
        self.endpoint = worker_config.endpoints[0]

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
                "endpoint": self.endpoint,
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
        with tracer.start_as_current_span("remote.generate") as span:
            span.set_attribute("num_prompts", len(prompts))
            span.set_attribute("endpoint", self.endpoint)

            try:
                response = self._client.post(
                    f"{self.endpoint}/internal/generate",
                    json={"prompts": prompts, "sampling_params": sampling_params},
                )
            except httpx.RequestError as exc:
                span.set_attribute("error", True)
                raise RemoteInferenceError(
                    f"worker at {self.endpoint} unreachable: {type(exc).__name__}"
                ) from exc

            if response.status_code != 200:
                span.set_attribute("error", True)
                # Worker error bodies may echo prompt text; keep only a bounded
                # prefix out of the exception message.
                raise RemoteInferenceError(
                    f"worker at {self.endpoint} returned {response.status_code}: "
                    f"{response.text[:200]}"
                )

            payload = response.json()
            results = payload.get("results")
            if not isinstance(results, list) or len(results) != len(prompts):
                raise RemoteInferenceError(
                    f"worker at {self.endpoint} returned {len(results) if isinstance(results, list) else 'no'} "
                    f"results for {len(prompts)} prompts"
                )
            return results

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
