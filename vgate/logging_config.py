"""
Structured logging configuration for V-Gate.

Provides JSON-formatted logs with timestamps, log levels, and contextual data.
"""
import logging
import json
import sys
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _get_otel_ids() -> tuple:
    """
    Extract trace_id and span_id from the current OTel span context.

    Returns:
        Tuple of (trace_id, span_id) as hex strings, or ("", "") if unavailable.
    """
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.trace_id != 0:
            return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
    except ImportError:
        pass
    return "", ""


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add OTel trace/span IDs when available
        trace_id, span_id = _get_otel_ids()
        if trace_id:
            log_data["trace_id"] = trace_id
            log_data["span_id"] = span_id

        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        if hasattr(record, "extra_data") and record.extra_data:
            log_data.update(record.extra_data)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with colors."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build base message
        message = f"{color}[{timestamp}] {record.levelname:8}{self.RESET} {record.name}: {record.getMessage()}"

        # Add extra data if present
        if hasattr(record, "extra_data") and record.extra_data:
            extra_str = " | ".join(f"{k}={v}" for k, v in record.extra_data.items())
            message += f" | {extra_str}"

        # Append OTel trace/span IDs when available
        trace_id, span_id = _get_otel_ids()
        if trace_id:
            message += f" | trace_id={trace_id} span_id={span_id}"

        return message


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    logger_name: str = "vgate"
) -> logging.Logger:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, use JSON format; otherwise use console format
        logger_name: Name of the logger to configure

    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Set formatter based on preference
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(ConsoleFormatter())

    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding extra data to log records."""

    def __init__(self, logger: logging.Logger, **kwargs):
        self.logger = logger
        self.extra_data = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def info(self, message: str, **kwargs):
        """Log info message with context."""
        extra = {**self.extra_data, **kwargs}
        self.logger.info(message, extra={"extra_data": extra})

    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        extra = {**self.extra_data, **kwargs}
        self.logger.debug(message, extra={"extra_data": extra})

    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        extra = {**self.extra_data, **kwargs}
        self.logger.warning(message, extra={"extra_data": extra})

    def error(self, message: str, **kwargs):
        """Log error message with context."""
        extra = {**self.extra_data, **kwargs}
        self.logger.error(message, extra={"extra_data": extra})


def get_default_log_level() -> str:
    """Get the default log level from configuration."""
    from vgate.config import get_config
    return get_config().logging.level


def get_default_json_format() -> bool:
    """Get the default JSON format setting from configuration."""
    from vgate.config import get_config
    return get_config().logging.json_format


# For backward compatibility, these are now functions
# Use get_default_log_level() and get_default_json_format() instead
DEFAULT_LOG_LEVEL = os.getenv("VGATE_LOGGING__LEVEL", os.getenv("VGATE_LOG_LEVEL", "INFO"))
DEFAULT_JSON_FORMAT = os.getenv("VGATE_LOGGING__JSON_FORMAT", os.getenv("VGATE_LOG_JSON", "true")).lower() == "true"
