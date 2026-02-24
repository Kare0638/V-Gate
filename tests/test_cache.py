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
Unit tests for the ResultCache.

Tests cover cache operations, LRU eviction, and batch deduplication.
"""
import asyncio
import pytest
from unittest.mock import MagicMock

from vgate.cache import ResultCache
from vgate.config import CacheConfig
from vgate.batcher import RequestBatcher


class TestCacheConfig:
    """Tests for CacheConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()
        assert config.maxsize == 1000
        assert config.enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CacheConfig(maxsize=500, enabled=False)
        assert config.maxsize == 500
        assert config.enabled is False


class TestResultCache:
    """Tests for ResultCache class."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache returns stored value on hit."""
        cache = ResultCache()
        key = ResultCache.make_key("Hello", 0.7, 0.9, 256)
        await cache.put(key, {"text": "World"})

        result = await cache.get(key)

        assert result == {"text": "World"}
        assert cache.hits == 1
        assert cache.misses == 0

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache returns None on miss."""
        cache = ResultCache()
        key = ResultCache.make_key("Hello", 0.7, 0.9, 256)

        result = await cache.get(key)

        assert result is None
        assert cache.hits == 0
        assert cache.misses == 1

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ResultCache(CacheConfig(maxsize=2))

        await cache.put("a", {"v": 1})
        await cache.put("b", {"v": 2})
        await cache.put("c", {"v": 3})  # Should evict "a"

        assert await cache.get("a") is None  # Evicted
        assert await cache.get("b") is not None
        assert await cache.get("c") is not None

    @pytest.mark.asyncio
    async def test_lru_access_order(self):
        """Test LRU updates order on access."""
        cache = ResultCache(CacheConfig(maxsize=2))

        await cache.put("a", {"v": 1})
        await cache.put("b", {"v": 2})
        await cache.get("a")  # Access "a", making it recently used
        await cache.put("c", {"v": 3})  # Should evict "b", not "a"

        assert await cache.get("a") is not None
        assert await cache.get("b") is None  # Evicted
        assert await cache.get("c") is not None

    def test_make_key_consistency(self):
        """Test make_key produces consistent keys for same input."""
        key1 = ResultCache.make_key("Hello", 0.7, 0.9, 256)
        key2 = ResultCache.make_key("Hello", 0.7, 0.9, 256)
        assert key1 == key2

    def test_make_key_uniqueness(self):
        """Test make_key produces different keys for different inputs."""
        key1 = ResultCache.make_key("Hello", 0.7, 0.9, 256)
        key2 = ResultCache.make_key("World", 0.7, 0.9, 256)
        key3 = ResultCache.make_key("Hello", 0.8, 0.9, 256)
        key4 = ResultCache.make_key("Hello", 0.7, 0.95, 256)
        key5 = ResultCache.make_key("Hello", 0.7, 0.9, 512)

        assert len({key1, key2, key3, key4, key5}) == 5  # All unique

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test get_stats returns correct statistics."""
        cache = ResultCache(CacheConfig(maxsize=100))

        await cache.put("a", {"v": 1})
        await cache.get("a")  # Hit
        await cache.get("b")  # Miss

        stats = cache.get_stats()

        assert stats["size"] == 1
        assert stats["maxsize"] == 100
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_update_existing_key(self):
        """Test updating an existing key moves it to end."""
        cache = ResultCache(CacheConfig(maxsize=2))

        await cache.put("a", {"v": 1})
        await cache.put("b", {"v": 2})
        await cache.put("a", {"v": 10})  # Update "a", move to end
        await cache.put("c", {"v": 3})  # Should evict "b", not "a"

        result_a = await cache.get("a")
        result_b = await cache.get("b")
        result_c = await cache.get("c")

        assert result_a == {"v": 10}
        assert result_b is None  # Evicted
        assert result_c is not None


class MockLLM:
    """Mock vLLM for testing without GPU."""

    def __init__(self):
        self.call_count = 0
        self.prompts_seen = []

    def generate(self, prompts, sampling_params):
        """Track prompts and return mock outputs."""
        self.call_count += 1
        self.prompts_seen.extend(prompts)

        outputs = []
        for prompt in prompts:
            mock_output = MagicMock()
            mock_output.outputs = [MagicMock()]
            mock_output.outputs[0].text = f"Response to: {prompt[:30]}"
            mock_output.outputs[0].token_ids = list(range(10))
            mock_output.metrics = None
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
def cached_batcher(mock_engine):
    """Create a RequestBatcher with cache enabled."""
    return RequestBatcher(
        engine=mock_engine,
        max_batch_size=4,
        max_wait_time_ms=50.0,
    )


class TestBatcherCache:
    """Tests for cache integration in RequestBatcher."""

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_inference(self, cached_batcher):
        """Test cache hit returns without calling engine."""
        await cached_batcher.start()

        # First request - cache miss
        result1 = await cached_batcher.submit("Hello world", max_tokens=50)
        call_count_after_first = cached_batcher.engine.llm.call_count

        # Second identical request - should hit cache
        result2 = await cached_batcher.submit("Hello world", max_tokens=50)
        call_count_after_second = cached_batcher.engine.llm.call_count

        assert result1["text"] == result2["text"]
        assert call_count_after_second == call_count_after_first  # No new inference

        await cached_batcher.stop()

    @pytest.mark.asyncio
    async def test_cache_stats_in_metrics(self, cached_batcher):
        """Test cache stats are included in batcher metrics."""
        await cached_batcher.start()

        await cached_batcher.submit("Test prompt", max_tokens=50)
        await cached_batcher.submit("Test prompt", max_tokens=50)  # Cache hit

        metrics = cached_batcher.get_metrics()

        assert "cache" in metrics
        assert metrics["cache"]["hits"] == 1
        assert metrics["cache"]["misses"] == 1
        assert metrics["cache"]["size"] == 1

        await cached_batcher.stop()

    @pytest.mark.asyncio
    async def test_different_params_different_cache_keys(self, cached_batcher):
        """Test different parameters produce different cache entries."""
        await cached_batcher.start()

        # Same prompt, different temperatures
        await cached_batcher.submit("Hello", max_tokens=50, temperature=0.7)
        await cached_batcher.submit("Hello", max_tokens=50, temperature=0.8)

        metrics = cached_batcher.get_metrics()
        assert metrics["cache"]["size"] == 2

        await cached_batcher.stop()


class TestBatchDedup:
    """Tests for batch deduplication."""

    @pytest.mark.asyncio
    async def test_batch_dedup_same_prompt(self, cached_batcher):
        """Test identical prompts in same batch share inference."""
        await cached_batcher.start()

        # Submit 3 identical prompts concurrently
        tasks = [
            cached_batcher.submit("Duplicate prompt", max_tokens=50)
            for _ in range(3)
        ]
        results = await asyncio.gather(*tasks)

        # All should get the same result
        assert all(r["text"] == results[0]["text"] for r in results)

        # Only 1 unique prompt should have been inferred
        assert len(cached_batcher.engine.llm.prompts_seen) == 1

        await cached_batcher.stop()

    @pytest.mark.asyncio
    async def test_batch_dedup_mixed_prompts(self, cached_batcher):
        """Test batch with mix of duplicate and unique prompts."""
        await cached_batcher.start()

        # Submit mix: 2 x "A", 2 x "B", 1 x "C" = 5 requests, 3 unique
        tasks = [
            cached_batcher.submit("Prompt A", max_tokens=50),
            cached_batcher.submit("Prompt B", max_tokens=50),
            cached_batcher.submit("Prompt A", max_tokens=50),
            cached_batcher.submit("Prompt B", max_tokens=50),
            cached_batcher.submit("Prompt C", max_tokens=50),
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        # Should have inferred only 3 unique prompts
        unique_prompts = set(cached_batcher.engine.llm.prompts_seen)
        assert len(unique_prompts) == 3

        await cached_batcher.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
