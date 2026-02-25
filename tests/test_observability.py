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
Unit tests for V-Gate observability features.

Tests cover structured logging and Prometheus metrics.
"""
import asyncio
import pytest
import logging
import json
from io import StringIO
from unittest.mock import MagicMock, patch

from vgate.logging_config import (
    JSONFormatter, ConsoleFormatter, setup_logging, get_logger, LogContext
)
from vgate.metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, BATCH_SIZE, CACHE_HITS, CACHE_MISSES,
    TOKENS_GENERATED, init_app_info
)
from vgate.cache import ResultCache
from vgate.config import CacheConfig
from vgate.batcher import RequestBatcher


class TestJSONFormatter:
    """Tests for JSON log formatter."""

    def test_format_basic_message(self):
        """Test basic message formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_format_with_extra_data(self):
        """Test formatting with extra data."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.extra_data = {"batch_id": 5, "batch_size": 8}

        output = formatter.format(record)
        data = json.loads(output)

        assert data["batch_id"] == 5
        assert data["batch_size"] == 8

    def test_format_with_request_id(self):
        """Test formatting with request ID."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.request_id = "abc123"

        output = formatter.format(record)
        data = json.loads(output)

        assert data["request_id"] == "abc123"


class TestConsoleFormatter:
    """Tests for console log formatter."""

    def test_format_basic_message(self):
        """Test basic console formatting."""
        formatter = ConsoleFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)

        assert "INFO" in output
        assert "test.logger" in output
        assert "Test message" in output

    def test_format_with_extra_data(self):
        """Test console formatting with extra data."""
        formatter = ConsoleFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.extra_data = {"key": "value"}

        output = formatter.format(record)

        assert "key=value" in output


class TestSetupLogging:
    """Tests for logging setup."""

    def test_setup_json_logging(self):
        """Test JSON logging setup."""
        logger = setup_logging(level="DEBUG", json_format=True, logger_name="test.json")

        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)

    def test_setup_console_logging(self):
        """Test console logging setup."""
        logger = setup_logging(level="INFO", json_format=False, logger_name="test.console")

        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, ConsoleFormatter)


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger(self):
        """Test getting a logger by name."""
        logger = get_logger("test.module")

        assert logger.name == "test.module"
        assert isinstance(logger, logging.Logger)


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_log_with_context(self):
        """Test logging with context."""
        logger = setup_logging(level="DEBUG", json_format=True, logger_name="test.context")

        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.handlers = [handler]

        ctx = LogContext(logger, request_id="test123")
        ctx.info("Test message", extra_key="extra_value")

        output = stream.getvalue()
        data = json.loads(output)

        assert data["message"] == "Test message"
        assert data["request_id"] == "test123"
        assert data["extra_key"] == "extra_value"


class TestPrometheusMetrics:
    """Tests for Prometheus metrics."""

    def test_request_count_labels(self):
        """Test REQUEST_COUNT has correct labels."""
        assert "endpoint" in REQUEST_COUNT._labelnames
        assert "method" in REQUEST_COUNT._labelnames
        assert "status" in REQUEST_COUNT._labelnames

    def test_request_latency_buckets(self):
        """Test REQUEST_LATENCY has histogram buckets."""
        # Histogram should have buckets defined
        assert hasattr(REQUEST_LATENCY, '_upper_bounds')

    def test_batch_size_histogram(self):
        """Test BATCH_SIZE histogram."""
        # Record some observations
        BATCH_SIZE.observe(4)
        BATCH_SIZE.observe(8)
        BATCH_SIZE.observe(2)

        # Verify it's a histogram
        assert hasattr(BATCH_SIZE, '_upper_bounds')

    def test_cache_counters(self):
        """Test cache hit/miss counters."""
        initial_hits = CACHE_HITS._value.get()
        initial_misses = CACHE_MISSES._value.get()

        CACHE_HITS.inc()
        CACHE_MISSES.inc()
        CACHE_MISSES.inc()

        assert CACHE_HITS._value.get() == initial_hits + 1
        assert CACHE_MISSES._value.get() == initial_misses + 2

    def test_init_app_info(self):
        """Test application info initialization."""
        init_app_info(version="1.0.0", model="test-model")
        # Info metric should be set (no exception raised)


class MockBackend:
    """Mock inference backend for testing without GPU."""

    def create_sampling_params(self, temperature, top_p, max_tokens):
        return {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}

    def generate(self, prompts, sampling_params):
        """Simulate batch generation with standardized dict output."""
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


@pytest.fixture
def mock_engine():
    """Create a mock engine for testing."""
    return MockEngine()


@pytest.fixture
def batcher_with_metrics(mock_engine):
    """Create a RequestBatcher for metrics testing."""
    return RequestBatcher(
        engine=mock_engine,
        max_batch_size=4,
        max_wait_time_ms=50.0,
    )


class TestBatcherMetricsIntegration:
    """Tests for batcher metrics integration."""

    @pytest.mark.asyncio
    async def test_batch_metrics_recorded(self, batcher_with_metrics):
        """Test that batch metrics are recorded."""
        await batcher_with_metrics.start()

        # Submit some requests
        await batcher_with_metrics.submit("Test prompt", max_tokens=50)

        # Get metrics
        metrics = batcher_with_metrics.get_metrics()

        assert metrics["total_requests"] >= 1
        assert metrics["total_batches"] >= 1
        assert "total_deduplicated" in metrics

        await batcher_with_metrics.stop()

    @pytest.mark.asyncio
    async def test_cache_metrics_integration(self, batcher_with_metrics):
        """Test cache metrics are properly integrated."""
        await batcher_with_metrics.start()

        # First request - cache miss
        await batcher_with_metrics.submit("Unique prompt", max_tokens=50)

        # Second identical request - cache hit
        await batcher_with_metrics.submit("Unique prompt", max_tokens=50)

        metrics = batcher_with_metrics.get_metrics()

        assert metrics["cache"]["hits"] >= 1
        assert metrics["cache"]["misses"] >= 1

        await batcher_with_metrics.stop()


class TestCacheMetricsIntegration:
    """Tests for cache Prometheus metrics."""

    @pytest.mark.asyncio
    async def test_cache_prometheus_metrics(self):
        """Test cache updates Prometheus metrics."""
        cache = ResultCache(CacheConfig(maxsize=10))

        # Get initial values
        initial_hits = CACHE_HITS._value.get()
        initial_misses = CACHE_MISSES._value.get()

        # Cache miss
        key = ResultCache.make_key("test", 0.7, 0.9, 256)
        await cache.get(key)

        # Cache put + hit
        await cache.put(key, {"text": "result"})
        await cache.get(key)

        # Verify Prometheus counters incremented
        assert CACHE_HITS._value.get() > initial_hits
        assert CACHE_MISSES._value.get() > initial_misses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
