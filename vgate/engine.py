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

import os
import time
from typing import Optional

from vgate.backends.base import DryRunBackend, InferenceBackend
from vgate.config import ModelConfig, WorkerConfig, get_config
from vgate.tracing import get_tracer

tracer = get_tracer("vgate.engine")

DRY_RUN = os.getenv("VGATE_DRY_RUN", "false").lower() in ("true", "1", "yes")


def _is_remote(worker_config: WorkerConfig) -> bool:
    """Whether this process forwards inference rather than running it."""
    return bool(worker_config.endpoints) or bool(worker_config.discovery.dns_name)


def _create_backend(
    engine_type: str, worker_config: Optional[WorkerConfig] = None
) -> InferenceBackend:
    """
    Factory function to create the appropriate inference backend.

    A gateway configured to reach workers forwards inference instead of running
    it, so that check comes before DRY_RUN and engine_type: those describe how
    the *worker* runs a model, and are irrelevant to a process that only
    forwards. Configured means either a static endpoint list or a discovery
    name — a discovering gateway is remote even before its first resolve, since
    an empty pool is a pool with nothing in it yet, not a local backend.
    Without either, the historical single-process behavior is preserved.
    """
    if worker_config is not None and _is_remote(worker_config):
        from vgate.backends.remote_backend import RemoteBackend
        return RemoteBackend(worker_config)
    if DRY_RUN:
        return DryRunBackend()
    if engine_type == "vllm":
        from vgate.backends.vllm_backend import VLLMBackend
        return VLLMBackend()
    if engine_type == "sglang":
        from vgate.backends.sglang_backend import SGLangBackend
        return SGLangBackend()
    raise ValueError(f"Unknown engine_type: {engine_type!r}")


class VGateEngine:
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        worker_config: Optional[WorkerConfig] = None,
    ):
        """
        Initialize the inference engine with configuration.

        Args:
            model_config: Model configuration. If None, uses global config.
            worker_config: Remote worker settings. If it lists endpoints, this
                engine forwards inference instead of loading a model locally.
                If None, uses global config.
        """
        config = get_config()
        if model_config is None:
            model_config = config.model
        if worker_config is None:
            worker_config = config.worker

        self.is_remote = _is_remote(worker_config)
        self.backend: InferenceBackend = _create_backend(
            model_config.engine_type, worker_config
        )

        if self.is_remote:
            target = worker_config.discovery.dns_name or worker_config.endpoints
            print(f"V-Gate gateway forwarding inference to {target}")
        elif DRY_RUN:
            print("V-Gate starting in DRY-RUN mode (no GPU required)")
        else:
            self.backend.load_model(model_config)

    def chat_completions(self, prompt, max_tokens=256):
        with tracer.start_as_current_span("engine.chat_completions") as span:
            span.set_attribute("prompt_length", len(prompt))
            span.set_attribute("max_tokens", max_tokens)
            span.set_attribute("dry_run", DRY_RUN)

            sampling_params = self.backend.create_sampling_params(
                temperature=0.7, top_p=0.9, max_tokens=max_tokens
            )

            start_time = time.perf_counter()
            results = self.backend.generate([prompt], sampling_params)
            end_time = time.perf_counter()

            result = results[0]
            generated_text = result["text"]
            num_tokens = result["num_tokens"]

            metrics = result.get("metrics", {})
            ttft = metrics.get("ttft", 0.0)
            gen_time = metrics.get("gen_time", end_time - start_time)

            tpot = (gen_time / num_tokens) if num_tokens > 0 else 0

            span.set_attribute("tokens_generated", num_tokens)
            span.set_attribute("ttft_ms", round(ttft * 1000, 2))

            return {
                "text": generated_text,
                "ttft": ttft,
                "tpot": tpot,
                "total_tokens": num_tokens
            }

    def embeddings(self, input_text: str):
        """
        Placeholder for embeddings generation.
        In a real scenario, a dedicated embedding model would be loaded and used here.
        Returns a mock embedding for MVP.
        """
        print(f"VGateEngine: Generating mock embeddings for input: '{input_text}'")
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [i * 0.01 for i in range(1536)],
                    "index": 0,
                }
            ],
            "model": "mock-embedding-model",
            "usage": {"prompt_tokens": len(input_text), "total_tokens": len(input_text)},
        }
