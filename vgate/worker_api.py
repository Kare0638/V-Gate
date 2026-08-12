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
Worker-role API.

A worker owns one engine instance and nothing else: no batching, no cache, no
admission control. Those belong to the gateway, and duplicating them here would
mean every request is batched and cached twice.

The request/response shape is deliberately the wire form of
InferenceBackend.generate(), so RemoteBackend on the gateway side is a
transport for that method rather than a second API with its own semantics.
"""

import asyncio
import time
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from vgate.logging_config import get_logger
from vgate.metrics import INFERENCE_ERRORS
from vgate.tracing import get_tracer

logger = get_logger("vgate.worker")
tracer = get_tracer("vgate.worker")

router = APIRouter(tags=["worker"])

# Set by main.py during lifespan when running with role=worker.
_engine = None
# The in-process engines are not safe to call concurrently, which is why the
# gateway's batcher holds a lock around local backends. That lock lives on the
# gateway, so once inference moves here the worker has to enforce it itself.
_inference_lock: asyncio.Lock | None = None


def set_engine(engine) -> None:
    """Bind the worker's engine. Called once from the app lifespan."""
    global _engine, _inference_lock
    _engine = engine
    _inference_lock = asyncio.Lock()


class SamplingParams(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256


class GenerateRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1)
    sampling_params: SamplingParams = Field(default_factory=SamplingParams)


class GenerateResponse(BaseModel):
    results: List[Dict[str, Any]]


@router.post("/internal/generate", summary="Internal Generate (worker role)")
async def internal_generate(request: GenerateRequest) -> GenerateResponse:
    """
    Run inference for a batch of prompts.

    Not part of the public API surface: the gateway calls this, clients do not.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Worker engine not initialized")

    with tracer.start_as_current_span("worker.generate") as span:
        span.set_attribute("num_prompts", len(request.prompts))

        backend = _engine.backend
        sampling_params = backend.create_sampling_params(
            temperature=request.sampling_params.temperature,
            top_p=request.sampling_params.top_p,
            max_tokens=request.sampling_params.max_tokens,
        )

        started = time.perf_counter()
        try:
            # Serialize engine access, then run the blocking call off the event
            # loop so this worker can still answer /health while generating.
            async with _inference_lock:
                results = await asyncio.get_running_loop().run_in_executor(
                    None, backend.generate, request.prompts, sampling_params
                )
        except Exception as exc:
            span.set_attribute("error", True)
            INFERENCE_ERRORS.labels(error_type=type(exc).__name__).inc()
            logger.error(
                "Worker inference failed",
                extra={"extra_data": {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "num_prompts": len(request.prompts),
                }}
            )
            raise HTTPException(
                status_code=500, detail=f"Inference failed: {type(exc).__name__}"
            ) from exc

        duration = time.perf_counter() - started
        span.set_attribute("duration_s", round(duration, 4))
        logger.info(
            "Worker batch completed",
            extra={"extra_data": {
                "num_prompts": len(request.prompts),
                "duration_ms": round(duration * 1000, 2),
            }}
        )
        return GenerateResponse(results=results)
