import asyncio
import os
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

DRY_RUN = os.getenv("VGATE_DRY_RUN", "false").lower() in ("true", "1", "yes")

from vgate.cache import ResultCache
from vgate.config import get_config, CacheConfig as ConfigCacheConfig
from vgate.logging_config import get_logger
from vgate.tracing import get_tracer, get_current_trace_id
from vgate.metrics import (
    BATCH_SIZE, BATCH_PROCESSING_TIME, BATCH_QUEUE_TIME,
    PENDING_REQUESTS, TOTAL_BATCHES, TTFT, TPOT,
    TOKENS_GENERATED, INFERENCE_ERRORS, UNIQUE_PROMPTS_PER_BATCH,
    DEDUPLICATED_REQUESTS, DEDUP_RATIO
)

logger = get_logger("vgate.batcher")
tracer = get_tracer("vgate.batcher")


@dataclass
class BatchRequest:
    """A single request waiting to be batched."""
    prompt: str
    max_tokens: int
    temperature: float = 0.7
    top_p: float = 0.9
    cache_key: str = ""
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    created_at: float = field(default_factory=time.time)


class RequestBatcher:
    """
    Dynamic request batcher that aggregates concurrent requests
    and processes them in batches for improved GPU utilization.

    Batching triggers when either condition is met:
    - Batch size reaches max_batch_size
    - Wait time exceeds max_wait_time_ms
    """

    def __init__(
        self,
        engine,
        max_batch_size: Optional[int] = None,
        max_wait_time_ms: Optional[float] = None,
    ):
        """
        Initialize the request batcher.

        Args:
            engine: The VGateEngine instance.
            max_batch_size: Max requests per batch. Uses config default if None.
            max_wait_time_ms: Max wait time before processing. Uses config default if None.
        """
        config = get_config()
        self.engine = engine
        self.max_batch_size = max_batch_size if max_batch_size is not None else config.batch.max_batch_size
        self.max_wait_time_ms = max_wait_time_ms if max_wait_time_ms is not None else config.batch.max_wait_time_ms
        self.cache = ResultCache()

        self._queue: List[BatchRequest] = []
        self._lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()  # Prevent concurrent vLLM calls
        self._batch_task: asyncio.Task = None
        self._running = False

        # Metrics
        self.total_requests = 0
        self.total_batches = 0
        self.total_batch_size = 0
        self.total_deduplicated = 0

    async def start(self):
        """Start the background batching task."""
        if self._running:
            return
        self._running = True
        self._batch_task = asyncio.create_task(self._batch_loop())
        logger.info(
            "Batcher started",
            extra={"extra_data": {
                "max_batch_size": self.max_batch_size,
                "max_wait_time_ms": self.max_wait_time_ms
            }}
        )

    async def stop(self):
        """Stop the batching task and process remaining requests."""
        self._running = False
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        # Process any remaining requests
        await self._process_batch()
        logger.info("Batcher stopped")

    async def submit(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Submit a request for batched processing.
        Returns the result when the batch is processed.
        Checks cache first and returns cached result if available.

        Args:
            prompt: The input prompt.
            max_tokens: Max tokens to generate. Uses config default if None.
            temperature: Sampling temperature. Uses config default if None.
            top_p: Top-p sampling. Uses config default if None.
        """
        with tracer.start_as_current_span("batcher.submit") as span:
            # Apply defaults from config
            config = get_config()
            if max_tokens is None:
                max_tokens = config.inference.max_tokens
            if temperature is None:
                temperature = config.inference.temperature
            if top_p is None:
                top_p = config.inference.top_p

            span.set_attribute("prompt_length", len(prompt))

            cache_key = ResultCache.make_key(prompt, temperature, top_p, max_tokens)

            # Check cache first
            cached = await self.cache.get(cache_key)
            if cached:
                span.set_attribute("cache_hit", True)
                logger.debug(
                    "Cache hit",
                    extra={"extra_data": {"cache_key": cache_key[:8]}}
                )
                return cached

            span.set_attribute("cache_hit", False)

            request = BatchRequest(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                cache_key=cache_key,
                future=asyncio.get_event_loop().create_future(),
                created_at=time.time(),
            )

            async with self._lock:
                self._queue.append(request)
                self.total_requests += 1
                queue_size = len(self._queue)
                PENDING_REQUESTS.set(queue_size)

            # If batch is full, signal immediate processing
            if queue_size >= self.max_batch_size:
                asyncio.create_task(self._process_batch())

            # Wait for result
            result = await request.future
            return result

    async def _batch_loop(self):
        """Background loop that triggers batch processing on timeout."""
        while self._running:
            await asyncio.sleep(self.max_wait_time_ms / 1000.0)

            if self._queue:
                await self._process_batch()

    async def _process_batch(self):
        """Process all pending requests as a batch with deduplication."""
        # Use inference lock to prevent concurrent vLLM calls
        async with self._inference_lock:
            async with self._lock:
                if not self._queue:
                    return

                # Take all pending requests
                batch = self._queue[:]
                self._queue.clear()
                PENDING_REQUESTS.set(0)

            if not batch:
                return

            batch_size = len(batch)
            self.total_batches += 1
            self.total_batch_size += batch_size

            with tracer.start_as_current_span("batcher.process_batch") as span:
                # Record batch metrics
                TOTAL_BATCHES.inc()
                BATCH_SIZE.observe(batch_size)

                span.set_attribute("batch_id", self.total_batches)
                span.set_attribute("batch_size", batch_size)

                # Calculate queue wait times
                current_time = time.time()
                for req in batch:
                    queue_time = current_time - req.created_at
                    BATCH_QUEUE_TIME.observe(queue_time)

                logger.info(
                    "Processing batch",
                    extra={"extra_data": {
                        "batch_id": self.total_batches,
                        "batch_size": batch_size
                    }}
                )

                try:
                    # Group requests by cache_key for deduplication
                    unique_prompts: Dict[str, List[BatchRequest]] = {}
                    for req in batch:
                        if req.cache_key not in unique_prompts:
                            unique_prompts[req.cache_key] = []
                        unique_prompts[req.cache_key].append(req)

                    # Only infer unique prompts
                    prompts_to_infer = [reqs[0].prompt for reqs in unique_prompts.values()]
                    temps_to_infer = [reqs[0].temperature for reqs in unique_prompts.values()]
                    top_ps_to_infer = [reqs[0].top_p for reqs in unique_prompts.values()]
                    max_tokens = max(req.max_tokens for req in batch)

                    dedup_count = batch_size - len(prompts_to_infer)
                    span.set_attribute("unique_prompts", len(prompts_to_infer))
                    span.set_attribute("deduplicated", dedup_count)

                    if dedup_count > 0:
                        self.total_deduplicated += dedup_count
                        DEDUPLICATED_REQUESTS.inc(dedup_count)
                        DEDUP_RATIO.set(dedup_count / batch_size)
                        logger.info(
                            "Deduplicated requests",
                            extra={"extra_data": {
                                "deduplicated": dedup_count,
                                "unique_prompts": len(prompts_to_infer)
                            }}
                        )
                    else:
                        DEDUP_RATIO.set(0)

                    UNIQUE_PROMPTS_PER_BATCH.observe(len(prompts_to_infer))

                    # Process batch through engine
                    batch_start = time.perf_counter()
                    results = await self._run_batch_inference(
                        prompts_to_infer, max_tokens, temps_to_infer[0], top_ps_to_infer[0]
                    )
                    batch_duration = time.perf_counter() - batch_start

                    # Exemplar for metric-trace correlation
                    trace_id = get_current_trace_id()
                    exemplar = {"trace_id": trace_id} if trace_id else None
                    BATCH_PROCESSING_TIME.observe(batch_duration, exemplar=exemplar)

                    # Record inference metrics
                    total_tokens = 0
                    for result in results:
                        if result.get("ttft", 0) > 0:
                            TTFT.observe(result["ttft"])
                        if result.get("tpot", 0) > 0:
                            TPOT.observe(result["tpot"])
                        tokens = result.get("total_tokens", 0)
                        TOKENS_GENERATED.inc(tokens)
                        total_tokens += tokens

                    logger.info(
                        "Batch inference completed",
                        extra={"extra_data": {
                            "batch_id": self.total_batches,
                            "duration_s": round(batch_duration, 3),
                            "prompts": len(prompts_to_infer),
                            "tokens": total_tokens
                        }}
                    )

                    # Distribute results to waiting requests and cache them
                    for (cache_key, reqs), result in zip(unique_prompts.items(), results):
                        # Store in cache
                        await self.cache.put(cache_key, result)
                        # Distribute to all requests with this cache_key
                        for req in reqs:
                            if not req.future.done():
                                req.future.set_result(result)

                except Exception as e:
                    span.set_attribute("error", True)
                    # On error, fail all requests in batch
                    INFERENCE_ERRORS.labels(error_type=type(e).__name__).inc()
                    logger.error(
                        "Batch processing error",
                        extra={"extra_data": {
                            "batch_id": self.total_batches,
                            "error": str(e),
                            "error_type": type(e).__name__
                        }}
                    )
                    for req in batch:
                        if not req.future.done():
                            req.future.set_exception(e)

    async def _run_batch_inference(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """
        Run batched inference through the engine.
        This runs in a thread pool to avoid blocking the event loop.
        OTel context is captured and re-attached in the thread.
        """
        from opentelemetry import context as otel_context

        loop = asyncio.get_event_loop()

        # Capture OTel context before crossing thread boundary
        ctx = otel_context.get_current()

        def _inference_with_context():
            # Re-attach OTel context in the thread pool thread
            token = otel_context.attach(ctx)
            try:
                with tracer.start_as_current_span("batcher.inference") as span:
                    span.set_attribute("num_prompts", len(prompts))
                    results = self._sync_batch_inference(
                        prompts, max_tokens, temperature, top_p
                    )
                    total_tokens = sum(r.get("total_tokens", 0) for r in results)
                    span.set_attribute("total_tokens_generated", total_tokens)
                    return results
            finally:
                otel_context.detach(token)

        results = await loop.run_in_executor(None, _inference_with_context)
        return results

    def _sync_batch_inference(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """Synchronous batch inference - runs in thread pool."""
        if DRY_RUN:
            sampling_params = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        else:
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )

        start_time = time.perf_counter()
        outputs = self.engine.llm.generate(prompts, sampling_params)
        end_time = time.perf_counter()

        total_time = end_time - start_time

        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)

            # Calculate metrics
            metrics = output.metrics
            if metrics:
                ttft = metrics.first_token_time - metrics.arrival_time
                gen_time = metrics.finished_time - metrics.first_token_time
            else:
                ttft = 0.0
                gen_time = total_time / len(prompts)

            tpot = (gen_time / num_tokens) if num_tokens > 0 else 0

            results.append({
                "text": generated_text,
                "ttft": ttft,
                "tpot": tpot,
                "total_tokens": num_tokens
            })

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Return batching metrics including cache stats."""
        avg_batch_size = (self.total_batch_size / self.total_batches) if self.total_batches > 0 else 0
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "average_batch_size": round(avg_batch_size, 2),
            "pending_requests": len(self._queue),
            "total_deduplicated": self.total_deduplicated,
            "cache": self.cache.get_stats(),
        }
