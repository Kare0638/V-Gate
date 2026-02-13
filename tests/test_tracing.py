"""
Unit tests for V-Gate tracing module.
"""
import json
import logging

from vgate.config import VGateConfig
from vgate.logging_config import JSONFormatter
from vgate.tracing import (
    get_current_trace_id,
    get_tracer,
    init_tracing,
    is_tracing_enabled,
    shutdown_tracing,
)


class TestTracingModule:
    """Tests for tracing initialization and helpers."""

    def test_init_tracing_disabled(self):
        """Tracing should remain disabled when config disables it."""
        config = VGateConfig(tracing={"enabled": False})

        init_tracing(config)

        assert is_tracing_enabled() is False

    def test_init_tracing_enabled_and_shutdown(self):
        """Tracing should initialize and then shut down cleanly."""
        config = VGateConfig(
            version="9.9.9-test",
            tracing={
                "enabled": True,
                "service_name": "vgate-test",
                "otlp_endpoint": "http://collector:4317",
                "otlp_insecure": True,
                "sample_rate": 0.5,
            },
        )

        init_tracing(config)
        assert is_tracing_enabled() is True

        shutdown_tracing()
        assert is_tracing_enabled() is False

    def test_get_current_trace_id_without_span(self):
        """No active span should return empty trace ID."""
        assert get_current_trace_id() == ""

    def test_get_current_trace_id_with_active_span(self):
        """Active span should return a 32-hex trace ID."""
        tracer = get_tracer("tests.tracing")

        with tracer.start_as_current_span("test-span"):
            trace_id = get_current_trace_id()

        assert len(trace_id) == 32
        int(trace_id, 16)  # Validate hex format
        assert get_current_trace_id() == ""

    def test_shutdown_tracing_is_idempotent(self):
        """Calling shutdown twice should not fail."""
        shutdown_tracing()
        shutdown_tracing()
        assert is_tracing_enabled() is False


class TestTracingLogCorrelation:
    """Tests for logging and tracing correlation."""

    def test_json_formatter_includes_trace_and_span_ids(self):
        """JSON logs should include trace and span IDs in active span."""
        tracer = get_tracer("tests.logging")
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="tests.logging",
            level=logging.INFO,
            pathname="test_tracing.py",
            lineno=1,
            msg="hello tracing",
            args=(),
            exc_info=None,
        )

        with tracer.start_as_current_span("log-span"):
            data = json.loads(formatter.format(record))

        assert "trace_id" in data
        assert "span_id" in data
        assert len(data["trace_id"]) == 32
        assert len(data["span_id"]) == 16
