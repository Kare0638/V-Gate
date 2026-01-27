import asyncio
import time
from typing import Any, Dict, List
from dataclasses import dataclass, field


@dataclass
class BatchRequest:
    """A single request waiting to be batched."""
    prompt: str
    max_tokens: int
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
        max_batch_size: int = 8,
        max_wait_time_ms: float = 50.0,
    ):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms

        self._queue: List[BatchRequest] = []
        self._lock = asyncio.Lock()
        self._batch_task: asyncio.Task = None
        self._running = False

        # Metrics
        self.total_requests = 0
        self.total_batches = 0
        self.total_batch_size = 0

    async def start(self):
        """Start the background batching task."""
        if self._running:
            return
        self._running = True
        self._batch_task = asyncio.create_task(self._batch_loop())
        print(f"[Batcher] Started with max_batch_size={self.max_batch_size}, max_wait_time_ms={self.max_wait_time_ms}")

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

    async def submit(self, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
        """
        Submit a request for batched processing.
        Returns the result when the batch is processed.
        """
        request = BatchRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            future=asyncio.get_event_loop().create_future(),
            created_at=time.time(),
        )

        async with self._lock:
            self._queue.append(request)
            self.total_requests += 1
            queue_size = len(self._queue)

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
        """Process all pending requests as a batch."""
        async with self._lock:
            if not self._queue:
                return

            # Take all pending requests
            batch = self._queue[:]
            self._queue.clear()

        if not batch:
            return

        batch_size = len(batch)
        self.total_batches += 1
        self.total_batch_size += batch_size

        print(f"[Batcher] Processing batch #{self.total_batches} with {batch_size} requests")

        try:
            # Collect all prompts
            prompts = [req.prompt for req in batch]
            max_tokens = max(req.max_tokens for req in batch)

            # Process batch through engine
            results = await self._run_batch_inference(prompts, max_tokens)

            # Distribute results to waiting requests
            for i, req in enumerate(batch):
                if not req.future.done():
                    req.future.set_result(results[i])

        except Exception as e:
            # On error, fail all requests in batch
            print(f"[Batcher] Batch processing error: {e}")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    async def _run_batch_inference(self, prompts: List[str], max_tokens: int) -> List[Dict[str, Any]]:
        """
        Run batched inference through the engine.
        This runs in a thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()

        # Run synchronous vLLM inference in thread pool
        results = await loop.run_in_executor(
            None,
            self._sync_batch_inference,
            prompts,
            max_tokens
        )
        return results

    def _sync_batch_inference(self, prompts: List[str], max_tokens: int) -> List[Dict[str, Any]]:
        """Synchronous batch inference - runs in thread pool."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
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

        print(f"[Batcher] Batch inference completed in {total_time:.3f}s for {len(prompts)} prompts")
        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Return batching metrics."""
        avg_batch_size = (self.total_batch_size / self.total_batches) if self.total_batches > 0 else 0
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "average_batch_size": round(avg_batch_size, 2),
            "pending_requests": len(self._queue),
        }
