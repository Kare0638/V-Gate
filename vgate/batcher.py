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
import functools
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from vgate.cache import ResultCache
from vgate.config import get_config
from vgate.logging_config import get_logger
from vgate.tracing import get_tracer, get_current_trace_id
from vgate.metrics import (
    BATCH_SIZE, BATCH_PROCESSING_TIME, BATCH_QUEUE_TIME,
    PENDING_REQUESTS, TOTAL_BATCHES, TTFT, TPOT,
    TOKENS_GENERATED, INFERENCE_ERRORS, UNIQUE_PROMPTS_PER_BATCH,
    DEDUPLICATED_REQUESTS, DEDUP_RATIO, INFLIGHT_INFERENCES,
    ABANDONED_INFERENCES
)

logger = get_logger("vgate.batcher")
tracer = get_tracer("vgate.batcher")


@dataclass
class _Inflight:
    """
    One in-flight inference and the callers waiting on it.

    `waiters` and `started` exist to decide what to do when every caller has
    walked away (all timed out or disconnected):

    - Not started yet: it is still queued for an admission permit, nothing has
      been spent, and nobody wants the answer. Cancel it, so abandoned work
      does not occupy a permit that a live request could use.
    - Already started: the backend call is running in a thread and cannot be
      interrupted anyway, and its result still populates the cache. Let it
      finish rather than pay for it and throw the answer away.
    """

    task: Optional[asyncio.Task] = None
    waiters: int = 0
    started: bool = False


class RequestBatcher:
    """
    Admission control, in-flight deduplication, and fan-out to the backend.

    Despite the name, this no longer builds batches. It used to collect
    requests into a window and hand the whole window to one backend call,
    which made sense when the backend was an in-process engine that benefited
    from receiving several prompts at once. Two things made that wrong:

    - vLLM and SGLang do continuous batching internally, and do it better --
      they can admit a new request into a batch that is already decoding,
      which a sealed Python-side window cannot.
    - With remote workers, a sealed batch is routed as a unit, so every
      request in it lands on the same worker regardless of load. Load-aware
      routing needs a decision per request, not per window.

    So the three things worth keeping were separated from batch construction:

    - **Deduplication**: concurrent identical requests share one inference.
      This is now stronger than before, because coalescing is not limited to
      requests that happened to land in the same window.
    - **Admission control**: `max_batch_size` bounds concurrent inferences.
    - **Fan-out**: each admitted request calls the backend on its own, so a
      router sees one decision per request.
    """

    def __init__(
        self,
        engine,
        max_batch_size: Optional[int] = None,
        max_wait_time_ms: Optional[float] = None,
    ):
        """
        Args:
            engine: The VGateEngine instance.
            max_batch_size: Maximum concurrent inferences. Retains its
                configuration key, but now bounds concurrency rather than
                window size -- there is no window to size.
            max_wait_time_ms: Accepted and ignored. Kept so existing configs
                and callers keep working; a deprecation warning is logged
                when it is set to a non-default value.
        """
        config = get_config()
        self.engine = engine
        self.max_batch_size = (
            max_batch_size if max_batch_size is not None else config.batch.max_batch_size
        )
        self.max_wait_time_ms = (
            max_wait_time_ms if max_wait_time_ms is not None else config.batch.max_wait_time_ms
        )
        self.cache = ResultCache()

        # In-process engines are not safe to call concurrently. Under the old
        # batching model a separate lock enforced that; with fan-out, every
        # request calls the backend on its own, so the constraint is expressed
        # as the admission limit itself -- one mechanism instead of two.
        # Backends opt into concurrency via supports_concurrent_calls; the
        # default is serial, so local backends stay protected.
        backend = getattr(engine, "backend", None)
        self._serialize_inference = not getattr(backend, "supports_concurrent_calls", False)
        self.max_concurrent_inferences = 1 if self._serialize_inference else self.max_batch_size

        # Bounds concurrent backend calls. Holders are the requests actually
        # running inference; deduplicated followers do not consume a permit.
        self._semaphore = asyncio.Semaphore(self.max_concurrent_inferences)
        # cache_key -> the in-flight entry computing it. Any later request for
        # the same key attaches to that entry instead of starting a second
        # inference.
        self._inflight: Dict[str, "_Inflight"] = {}
        self._lock = asyncio.Lock()
        self._running = False

        # Metrics
        self.total_requests = 0
        self.total_batches = 0
        self.total_batch_size = 0
        self.total_deduplicated = 0
        self.total_queue_time = 0.0
        self.total_queue_samples = 0
        self.total_ttft = 0.0
        self.total_tpot = 0.0
        self.total_inference_samples = 0
        self._waiting = 0

    async def start(self):
        """
        Mark the batcher running.

        There is no background loop any more: a request drives its own
        inference instead of waiting for a timer to seal a window.
        """
        if self._running:
            return
        self._running = True
        logger.info(
            "Batcher started",
            extra={"extra_data": {
                "max_concurrent_inferences": self.max_concurrent_inferences,
                "serialized_backend": self._serialize_inference,
                "mode": "dedup+admission+fanout",
                # Surfaced rather than warned about: the key is still accepted
                # so existing configs load, but it no longer does anything.
                "max_wait_time_ms_ignored": self.max_wait_time_ms,
            }}
        )

    async def stop(self):
        """
        Stop accepting work and let in-flight inferences finish.

        Cancelling them would leave their waiters with unresolved futures, and
        their results are already paid for -- draining also populates the
        cache, so the work is not wasted.
        """
        self._running = False
        async with self._lock:
            pending = [e.task for e in self._inflight.values() if e.task is not None]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        logger.info("Batcher stopped")

    async def submit(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Submit a request, returning its result.

        Checks the cache, coalesces onto an identical in-flight request if one
        exists, and otherwise runs the inference under the concurrency limit.

        Args:
            prompt: The input prompt.
            max_tokens: Max tokens to generate. Uses config default if None.
            temperature: Sampling temperature. Uses config default if None.
            top_p: Top-p sampling. Uses config default if None.
            timeout: Max seconds to wait for a result. Raises
                asyncio.TimeoutError when it elapses. The underlying
                inference is not cancelled -- other waiters may still need
                it, and its result still populates the cache.

        Raises:
            asyncio.TimeoutError: if `timeout` elapses before a result is ready.
            asyncio.CancelledError: if the calling task is cancelled while waiting.
        """
        with tracer.start_as_current_span("batcher.submit") as span:
            config = get_config()
            if max_tokens is None:
                max_tokens = config.inference.max_tokens
            if temperature is None:
                temperature = config.inference.temperature
            if top_p is None:
                top_p = config.inference.top_p

            span.set_attribute("prompt_length", len(prompt))

            cache_key = ResultCache.make_key(prompt, temperature, top_p, max_tokens)

            cached = await self.cache.get(cache_key)
            if cached:
                span.set_attribute("cache_hit", True)
                logger.debug(
                    "Cache hit",
                    extra={"extra_data": {"cache_key": cache_key[:8]}}
                )
                return cached

            span.set_attribute("cache_hit", False)
            self.total_requests += 1

            async with self._lock:
                entry = self._inflight.get(cache_key)
                coalesced = entry is not None
                if entry is None:
                    entry = _Inflight()
                    self._inflight[cache_key] = entry
                    entry.task = asyncio.create_task(
                        self._execute(entry, cache_key, prompt, temperature, top_p, max_tokens)
                    )
                    entry.task.add_done_callback(
                        functools.partial(self._retire, cache_key)
                    )
                entry.waiters += 1

            span.set_attribute("deduplicated", coalesced)
            if coalesced:
                self.total_deduplicated += 1
                DEDUPLICATED_REQUESTS.inc()
                DEDUP_RATIO.set(
                    self.total_deduplicated / self.total_requests
                    if self.total_requests else 0
                )
                logger.debug(
                    "Coalesced onto in-flight request",
                    extra={"extra_data": {"cache_key": cache_key[:8]}}
                )

            # shield: one waiter timing out or disconnecting must not cancel
            # an inference that other waiters are still expecting.
            try:
                if timeout is not None:
                    return await asyncio.wait_for(
                        asyncio.shield(entry.task), timeout=timeout
                    )
                return await asyncio.shield(entry.task)
            finally:
                await self._release_waiter(entry, cache_key)

    async def _release_waiter(self, entry: "_Inflight", cache_key: str) -> None:
        """
        Drop one waiter, and reclaim the inference if it was the last.

        Only work that has not started yet is cancelled -- see _Inflight.
        """
        async with self._lock:
            entry.waiters -= 1
            if entry.waiters > 0 or entry.started or entry.task.done():
                return
            abandoned = entry.task

        abandoned.cancel()
        ABANDONED_INFERENCES.inc()
        logger.info(
            "Cancelled abandoned request before admission",
            extra={"extra_data": {"cache_key": cache_key[:8]}}
        )

    def _retire(self, cache_key: str, task: asyncio.Task) -> None:
        """Drop a finished entry from the in-flight map. Runs as a done-callback."""
        current = self._inflight.get(cache_key)
        if current is not None and current.task is task:
            # Mutating a dict from a done-callback is safe: callbacks run on
            # the event loop thread, same as everything that reads it.
            self._inflight.pop(cache_key, None)

    async def _execute(
        self,
        entry: "_Inflight",
        cache_key: str,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Wait for an admission permit, run one inference, and cache it."""
        queued_at = time.monotonic()
        self._waiting += 1
        PENDING_REQUESTS.set(self._waiting)
        # Tracks whether this request is still counted as waiting, so a
        # cancellation while blocked on the semaphore does not leak the count.
        counted_as_waiting = True
        try:
            async with self._semaphore:
                # Past this point the work is committed: cancelling would not
                # stop the thread-pool call, and the result is worth caching.
                entry.started = True
                self._waiting -= 1
                counted_as_waiting = False
                PENDING_REQUESTS.set(self._waiting)

                queue_time = time.monotonic() - queued_at
                BATCH_QUEUE_TIME.observe(queue_time)
                self.total_queue_time += queue_time
                self.total_queue_samples += 1

                INFLIGHT_INFERENCES.inc()
                try:
                    result = await self._run_inference(
                        prompt, max_tokens, temperature, top_p
                    )
                finally:
                    INFLIGHT_INFERENCES.dec()
        finally:
            if counted_as_waiting:
                self._waiting -= 1
                PENDING_REQUESTS.set(self._waiting)

        await self.cache.put(cache_key, result)
        return result

    async def _run_inference(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Run one inference off the event loop.

        backend.generate() is synchronous per the InferenceBackend protocol,
        so it goes to a thread. OTel context is captured and re-attached
        across that boundary.
        """
        from opentelemetry import context as otel_context

        loop = asyncio.get_event_loop()
        ctx = otel_context.get_current()

        def _inference_with_context():
            token = otel_context.attach(ctx)
            try:
                with tracer.start_as_current_span("batcher.inference") as span:
                    span.set_attribute("num_prompts", 1)
                    result = self._sync_inference(prompt, max_tokens, temperature, top_p)
                    span.set_attribute("total_tokens_generated", result.get("total_tokens", 0))
                    return result
            finally:
                otel_context.detach(token)

        started = time.perf_counter()
        try:
            result = await loop.run_in_executor(None, _inference_with_context)
        except Exception as e:
            INFERENCE_ERRORS.labels(error_type=type(e).__name__).inc()
            logger.error(
                "Inference error",
                extra={"extra_data": {"error": str(e), "error_type": type(e).__name__}}
            )
            raise
        duration = time.perf_counter() - started

        trace_id = get_current_trace_id()
        exemplar = {"trace_id": trace_id} if trace_id else None
        BATCH_PROCESSING_TIME.observe(duration, exemplar=exemplar)

        # One backend call per request now. These two keep their meaning --
        # "prompts handed to one backend call" -- the distribution is simply
        # degenerate while fan-out is one prompt per call.
        self.total_batches += 1
        self.total_batch_size += 1
        TOTAL_BATCHES.inc()
        BATCH_SIZE.observe(1)
        UNIQUE_PROMPTS_PER_BATCH.observe(1)

        if result.get("ttft", 0) > 0:
            TTFT.observe(result["ttft"])
            self.total_ttft += result["ttft"]
        if result.get("tpot", 0) > 0:
            TPOT.observe(result["tpot"])
            self.total_tpot += result["tpot"]
            self.total_inference_samples += 1
        TOKENS_GENERATED.inc(result.get("total_tokens", 0))

        return result

    def _sync_inference(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Synchronous single-prompt inference - runs in the thread pool."""
        backend = self.engine.backend
        sampling_params = backend.create_sampling_params(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )

        start_time = time.perf_counter()
        backend_results = backend.generate([prompt], sampling_params)
        total_time = time.perf_counter() - start_time

        br = backend_results[0]
        num_tokens = br["num_tokens"]
        metrics = br.get("metrics", {})

        ttft = metrics.get("ttft", 0.0)
        gen_time = metrics.get("gen_time", total_time)
        tpot = (gen_time / num_tokens) if num_tokens > 0 else 0

        return {
            "text": br["text"],
            "ttft": ttft,
            "tpot": tpot,
            "total_tokens": num_tokens,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return admission/dedup metrics including cache stats."""
        avg_batch_size = (self.total_batch_size / self.total_batches) if self.total_batches > 0 else 0
        avg_queue_time = (
            self.total_queue_time / self.total_queue_samples if self.total_queue_samples > 0 else 0
        )
        avg_ttft = (
            self.total_ttft / self.total_inference_samples if self.total_inference_samples > 0 else 0
        )
        avg_tpot = (
            self.total_tpot / self.total_inference_samples if self.total_inference_samples > 0 else 0
        )
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "average_batch_size": round(avg_batch_size, 2),
            "pending_requests": self._waiting,
            "inflight_inferences": len(self._inflight),
            "total_deduplicated": self.total_deduplicated,
            "avg_queue_time_s": round(avg_queue_time, 4),
            "avg_ttft_s": round(avg_ttft, 4),
            "avg_tpot_s": round(avg_tpot, 4),
            "cache": self.cache.get_stats(),
        }
