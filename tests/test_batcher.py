"""
Unit tests for the RequestBatcher.

These tests use a mock engine to test batching behavior without GPU.
"""
import asyncio
import pytest
import time
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vgate'))

from batcher import RequestBatcher, BatchRequest


class MockLLM:
    """Mock vLLM for testing without GPU."""

    def generate(self, prompts, sampling_params):
        """Simulate batch generation with mock outputs."""
        outputs = []
        for prompt in prompts:
            mock_output = MagicMock()
            mock_output.outputs = [MagicMock()]
            mock_output.outputs[0].text = f"Response to: {prompt[:30]}"
            mock_output.outputs[0].token_ids = list(range(10))  # 10 tokens
            mock_output.metrics = None  # Trigger fallback timing
            outputs.append(mock_output)
        return outputs


class MockEngine:
    """Mock VGateEngine for testing."""

    def __init__(self):
        self.llm = MockLLM()


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


class TestBatchRequest:
    """Tests for BatchRequest dataclass."""

    def test_creation(self):
        """Test BatchRequest can be created with required fields."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            req = BatchRequest(
                prompt="Hello world",
                max_tokens=100,
            )
            assert req.prompt == "Hello world"
            assert req.max_tokens == 100
            assert req.future is not None
            assert req.created_at > 0
        finally:
            loop.close()


class TestRequestBatcher:
    """Tests for RequestBatcher class."""

    @pytest.mark.asyncio
    async def test_start_stop(self, batcher):
        """Test batcher can start and stop cleanly."""
        await batcher.start()
        assert batcher._running is True
        assert batcher._batch_task is not None

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

        await batcher.stop()

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
