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

import asyncio
import os
import time
from typing import Any, AsyncIterator, Dict, List, Protocol, runtime_checkable

from vgate.config import ModelConfig

# Optional synthetic per-call delay for the dry-run backend, used only by
# benchmarks/run_report.py to give dry-run batching scenarios a non-zero
# compute cost to amortize. Unset (0) by default, so it never affects tests.
_DRYRUN_LATENCY_MS = float(os.getenv("VGATE_DRYRUN_SIMULATED_LATENCY_MS", "0"))


def _simulate_batch_compute(sampling_params: Any) -> None:
    if _DRYRUN_LATENCY_MS <= 0:
        return
    max_tokens = sampling_params.get("max_tokens", 0) if isinstance(sampling_params, dict) else 0
    time.sleep((_DRYRUN_LATENCY_MS + max_tokens * 2) / 1000.0)


@runtime_checkable
class InferenceBackend(Protocol):
    """Protocol that all inference backends must implement."""

    def load_model(self, model_config: ModelConfig) -> None: ...

    def create_sampling_params(
        self, temperature: float, top_p: float, max_tokens: int
    ) -> Any: ...

    def generate(
        self, prompts: List[str], sampling_params: Any
    ) -> List[Dict[str, Any]]: ...

    def stream_generate(
        self, prompt: str, sampling_params: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream a single prompt's completion token-by-token.

        Yields dicts shaped {"delta": str, "num_tokens": int} where
        num_tokens is the cumulative token count so far; the last yielded
        chunk's num_tokens is the total for the request.
        """
        ...

    def shutdown(self) -> None: ...


class DryRunBackend:
    """Mock backend that returns placeholder responses without GPU."""

    # Stateless: each call only reads its arguments, so concurrent calls are
    # safe. Declaring it keeps dry-run benchmarks from measuring an artificial
    # serialization that the real backends do not have.
    supports_concurrent_calls = True

    def load_model(self, model_config: ModelConfig) -> None:
        pass

    def create_sampling_params(
        self, temperature: float, top_p: float, max_tokens: int
    ) -> Any:
        return {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}

    def generate(
        self, prompts: List[str], sampling_params: Any
    ) -> List[Dict[str, Any]]:
        _simulate_batch_compute(sampling_params)
        results = []
        for prompt in prompts:
            results.append({
                "text": f"[dry-run] echo: {prompt[:80]}",
                "token_ids": list(range(8)),
                "num_tokens": 8,
                "metrics": {},
            })
        return results

    async def stream_generate(
        self, prompt: str, sampling_params: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        max_tokens = sampling_params.get("max_tokens", 8) if isinstance(sampling_params, dict) else 8
        words = f"[dry-run] echo: {prompt[:80]}".split()
        num_words = min(len(words), max_tokens) or 1
        for i in range(num_words):
            await asyncio.sleep(0.02)
            delta = words[i] + (" " if i < num_words - 1 else "")
            yield {"delta": delta, "num_tokens": i + 1}

    def shutdown(self) -> None:
        pass
