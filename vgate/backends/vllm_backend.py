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
import uuid
from typing import Any, AsyncIterator, Dict, List

from vgate.config import ModelConfig


def _extract_metrics(output: Any) -> Dict[str, Any]:
    metrics_dict: Dict[str, Any] = {}
    metrics = output.metrics
    if metrics:
        # first_token_latency is precomputed by vLLM; first_token_ts/
        # last_token_ts are monotonic engine-core timestamps, safe to
        # subtract (unlike arrival_time, which is a wall-clock frontend
        # timestamp from a different clock domain).
        metrics_dict["ttft"] = metrics.first_token_latency
        metrics_dict["gen_time"] = metrics.last_token_ts - metrics.first_token_ts
    return metrics_dict


class VLLMBackend:
    """
    Inference backend using vLLM's AsyncLLMEngine.

    A single AsyncLLMEngine instance backs both generate() (the synchronous,
    batch-shaped call used by RequestBatcher) and stream_generate() (used by
    the SSE path). Using one shared engine — rather than a second offline
    LLM() instance — avoids loading the model twice and fighting over the
    same gpu_memory_utilization budget. It also means non-streaming requests
    are submitted to vLLM as independent generate() calls rather than one
    sealed Python-side batch, so they get real continuous batching from
    vLLM's own scheduler for free, ahead of the full RequestBatcher
    redefinition in ROADMAP.md Phase 2 task 9.
    """

    def __init__(self):
        self.engine = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def load_model(self, model_config: ModelConfig) -> None:
        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs

        # load_model() runs synchronously inside main.py's `lifespan`
        # coroutine (via VGateEngine.__init__), so this is the main uvicorn
        # loop. generate() below is called from a worker thread (batcher.py's
        # run_in_executor) and needs this reference to safely hand async
        # engine calls back to the loop that owns the engine.
        self._loop = asyncio.get_event_loop()

        print(f"Loading {model_config.model_id} with {model_config.quantization} quantization (vLLM AsyncLLMEngine)...")
        engine_args = AsyncEngineArgs(
            model=model_config.model_id,
            quantization=model_config.quantization,
            gpu_memory_utilization=model_config.gpu_memory_utilization,
            max_model_len=model_config.max_model_len,
            enforce_eager=model_config.enforce_eager,
            trust_remote_code=model_config.trust_remote_code,
            # AsyncEngineArgs defaults this to False already, unlike the
            # offline LLM() class — kept explicit as a guard against that
            # default flipping again (see vllm_backend.py history: this bit
            # us once, silently zeroing out TTFT/TPOT).
            disable_log_stats=False,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    def create_sampling_params(
        self, temperature: float, top_p: float, max_tokens: int
    ) -> Any:
        from vllm import SamplingParams

        return SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )

    async def _agenerate_one(self, prompt: str, sampling_params: Any) -> Any:
        request_id = str(uuid.uuid4())
        final_output = None
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            if output.finished:
                final_output = output
        return final_output

    def generate(
        self, prompts: List[str], sampling_params: Any
    ) -> List[Dict[str, Any]]:
        """
        Synchronous, batch-shaped entry point for RequestBatcher. Runs on a
        worker thread; bridges to the AsyncLLMEngine on the main loop via
        run_coroutine_threadsafe so there is only ever one engine/CUDA
        context, regardless of which thread calls this.
        """
        async def _run_all():
            return await asyncio.gather(*[
                self._agenerate_one(p, sampling_params) for p in prompts
            ])

        future = asyncio.run_coroutine_threadsafe(_run_all(), self._loop)
        outputs = future.result()

        results = []
        for output in outputs:
            text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids
            results.append({
                "text": text,
                "token_ids": list(token_ids),
                "num_tokens": len(token_ids),
                "metrics": _extract_metrics(output),
            })
        return results

    async def stream_generate(
        self, prompt: str, sampling_params: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        """Runs directly on the caller's loop (main.py's SSE path is itself
        async on the main loop), so no thread-bridging is needed here."""
        request_id = str(uuid.uuid4())
        prev_text_len = 0
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            text = output.outputs[0].text
            delta = text[prev_text_len:]
            prev_text_len = len(text)
            if delta:
                yield {"delta": delta, "num_tokens": len(output.outputs[0].token_ids)}

    def shutdown(self) -> None:
        self.engine = None
