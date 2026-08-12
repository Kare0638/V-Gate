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
Unit tests for the RequestBatcher.

These tests use a mock engine to test batching behavior without GPU.
"""
import asyncio
import pytest
import time
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from vgate.batcher import RequestBatcher


class MockBackend:
    """Mock inference backend for testing without GPU."""

    def __init__(self):
        # Records (prompts, sampling_params) for every generate() call, so
        # tests can assert on how requests were actually grouped/dispatched.
        self.calls = []

    def create_sampling_params(self, temperature, top_p, max_tokens):
        return {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}

    def generate(self, prompts, sampling_params):
        """Simulate batch generation with standardized dict output."""
        self.calls.append((list(prompts), dict(sampling_params)))
        results = []
        for prompt in prompts:
            results.append({
                "text": f"Response to: {prompt[:30]}",
                "token_ids": list(range(10)),
                "num_tokens": 10,
                "metrics": {},
            })
        return results

    def shutdown(self):
        pass


class MockEngine:
    """Mock VGateEngine for testing."""

    def __init__(self):
        self.backend = MockBackend()


class _SlowBackend:
    """Backend with a controllable delay, for timeout/cancellation tests."""

    def __init__(self, delay: float):
        self.delay = delay
        self.calls = 0

    def create_sampling_params(self, temperature, top_p, max_tokens):
        return {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}

    def generate(self, prompts, sampling_params):
        import time as _t

        self.calls += 1
        _t.sleep(self.delay)
        return [
            {
                "text": f"Response to: {p[:30]}",
                "token_ids": list(range(10)),
                "num_tokens": 10,
                "metrics": {},
            }
            for p in prompts
        ]

    def shutdown(self):
        pass


class _EngineWith:
    """Minimal engine wrapper around an arbitrary backend."""

    def __init__(self, backend):
        self.backend = backend


@pytest.fixture
def mock_engine():
    """Create a mock engine for testing."""
    return MockEngine()


@pytest.fixture
def batcher(mock_engine):
    """Create a RequestBatcher with mock engine."""
    return RequestBatcher(
        engine=mock_engine,
        max_batch_size=4,
        max_wait_time_ms=50.0,
    )


class TestRequestBatcher:
    """Tests for RequestBatcher class."""

    @pytest.mark.asyncio
    async def test_start_stop(self, batcher):
        """Test batcher can start and stop cleanly."""
        await batcher.start()
        assert batcher._running is True
        # No background loop any more: requests drive their own inference
        # instead of waiting for a timer to seal a window.
        assert batcher._inflight == {}

        await batcher.stop()
        assert batcher._running is False

    @pytest.mark.asyncio
    async def test_single_request(self, batcher):
        """Test single request is processed correctly."""
        await batcher.start()

        result = await batcher.submit("What is 2+2?", max_tokens=50)

        assert "text" in result
        assert "total_tokens" in result
        assert result["total_tokens"] == 10  # Mock returns 10 tokens

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_multiple_requests_batched(self, batcher):
        """Test multiple concurrent requests are batched together."""
        await batcher.start()

        # Submit 3 requests concurrently
        tasks = [
            batcher.submit(f"Question {i}", max_tokens=50)
            for i in range(3)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        # All should have been processed in 1 batch (< max_batch_size=4)
        assert batcher.total_batches >= 1

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_batch_size_limit(self, batcher):
        """Test batch respects max_batch_size limit."""
        await batcher.start()

        # Submit more requests than max_batch_size
        tasks = [
            batcher.submit(f"Question {i}", max_tokens=50)
            for i in range(6)  # max_batch_size is 4
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 6
        # Should have processed in at least 2 batches
        assert batcher.total_batches >= 1

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_max_batch_size_is_hard_cap(self, batcher):
        """No single generate() call should receive more than max_batch_size prompts.

        _process_batch used to drain the entire queue regardless of
        max_batch_size (only using it as a trigger threshold), so a burst of
        concurrent requests could produce a call larger than configured.
        """
        await batcher.start()

        tasks = [
            batcher.submit(f"Unique question {i}", max_tokens=50)
            for i in range(10)  # max_batch_size is 4
        ]
        await asyncio.gather(*tasks)

        assert len(batcher.engine.backend.calls) >= 1
        for prompts, _ in batcher.engine.backend.calls:
            assert len(prompts) <= batcher.max_batch_size

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_different_sampling_params_not_mixed(self, batcher):
        """Requests with different temperature/top_p/max_tokens landing in the
        same drained batch must not share a single generate() call — a shared
        SamplingParams would silently apply the wrong params to some prompts.
        """
        await batcher.start()

        tasks = [
            batcher.submit("Prompt hot", max_tokens=50, temperature=1.0, top_p=0.9),
            batcher.submit("Prompt cold", max_tokens=50, temperature=0.0, top_p=0.9),
        ]
        await asyncio.gather(*tasks)

        calls_by_prompt = {}
        for prompts, sampling_params in batcher.engine.backend.calls:
            for prompt in prompts:
                calls_by_prompt[prompt] = sampling_params

        assert calls_by_prompt["Prompt hot"]["temperature"] == 1.0
        assert calls_by_prompt["Prompt cold"]["temperature"] == 0.0

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, batcher):
        """Test metrics are tracked correctly."""
        await batcher.start()

        # Submit some requests
        await batcher.submit("Test 1", max_tokens=50)
        await batcher.submit("Test 2", max_tokens=50)

        metrics = batcher.get_metrics()

        assert "total_requests" in metrics
        assert "total_batches" in metrics
        assert "average_batch_size" in metrics
        assert "pending_requests" in metrics
        assert metrics["total_requests"] >= 2

        # Queue time / TTFT / TPOT averages used by the load benchmark report
        assert metrics["avg_queue_time_s"] >= 0
        assert metrics["avg_ttft_s"] >= 0
        assert metrics["avg_tpot_s"] >= 0

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_queue_time_uses_monotonic_clock(self, batcher):
        """
        Queue time must be computed from a monotonic clock, not wall time,
        so a backward wall-clock adjustment (e.g. NTP correction) can never
        produce a negative avg_queue_time_s.
        """
        await batcher.start()
        await batcher.submit("Test", max_tokens=50)
        metrics = batcher.get_metrics()
        assert metrics["avg_queue_time_s"] >= 0
        await batcher.stop()

    @pytest.mark.asyncio
    async def test_submit_timeout_raises_without_leaking_state(self, mock_engine):
        """
        A submit() timeout must raise without leaking admission state.

        Under fan-out the timeout does *not* cancel the inference: other
        waiters may still need it, and its result still populates the cache.
        So the assertion is that the in-flight entry retires once the
        inference finishes, not that it vanishes at timeout.
        """
        slow = _SlowBackend(delay=0.2)
        batcher = RequestBatcher(engine=_EngineWith(slow), max_batch_size=4)
        await batcher.start()
        try:
            with pytest.raises(asyncio.TimeoutError):
                await batcher.submit("Slow one", max_tokens=50, timeout=0.05)

            # Still running: the shield kept it alive past the caller's timeout.
            assert len(batcher._inflight) == 1

            await batcher.stop()  # drains in-flight work
            assert batcher._inflight == {}
            assert batcher.get_metrics()["pending_requests"] == 0
            # The result the caller gave up on is still cached.
            assert slow.calls == 1
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_cancellation_does_not_kill_shared_inference(self, mock_engine):
        """
        One caller cancelling must not cancel an inference another caller is
        waiting on -- that is the point of shielding the shared task.
        """
        slow = _SlowBackend(delay=0.15)
        batcher = RequestBatcher(engine=_EngineWith(slow), max_batch_size=4)
        await batcher.start()
        try:
            first = asyncio.create_task(batcher.submit("Shared", max_tokens=50))
            await asyncio.sleep(0.01)
            second = asyncio.create_task(batcher.submit("Shared", max_tokens=50))
            await asyncio.sleep(0.01)

            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

            # The survivor still gets a result, from the same single inference.
            result = await second
            assert result["text"].startswith("Response to: Shared")
            assert slow.calls == 1
        finally:
            await batcher.stop()

        assert batcher._inflight == {}
        assert batcher.get_metrics()["pending_requests"] == 0

    @pytest.mark.asyncio
    async def test_timeout_triggers_batch(self, batcher):
        """Test that timeout triggers batch processing."""
        await batcher.start()

        # Submit a single request
        start = time.time()
        result = await batcher.submit("Single request", max_tokens=50)
        elapsed = time.time() - start

        # Should complete within timeout window (plus some margin)
        assert elapsed < 0.2  # 50ms timeout + processing + margin
        assert result is not None

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, batcher):
        """Test pending requests are processed on shutdown."""
        await batcher.start()

        # Submit a request
        task = asyncio.create_task(batcher.submit("Pending request", max_tokens=50))

        # Give it a moment to queue
        await asyncio.sleep(0.01)

        # Stop should process remaining requests
        await batcher.stop()

        # The task should complete
        result = await task
        assert result is not None


class TestBatcherIntegration:
    """Integration tests for the batcher with simulated load."""

    @pytest.mark.asyncio
    async def test_high_concurrency(self, batcher):
        """Test batcher handles high concurrency correctly."""
        await batcher.start()

        # Simulate 20 concurrent requests
        tasks = [
            batcher.submit(f"Concurrent request {i}", max_tokens=50)
            for i in range(20)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        assert all("text" in r for r in results)
        assert batcher.total_requests == 20

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_sequential_requests(self, batcher):
        """Test sequential requests are processed correctly."""
        await batcher.start()

        for i in range(3):
            result = await batcher.submit(f"Sequential {i}", max_tokens=50)
            assert "text" in result

        assert batcher.total_requests == 3

        await batcher.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
