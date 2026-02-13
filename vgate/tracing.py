"""
V-Gate Distributed Tracing Module.

Provides OpenTelemetry integration for distributed tracing across
Gateway -> Batcher -> Engine components.

When tracing is disabled (default), the OTel API returns no-op tracers/spans
automatically -- zero conditional checks needed at call sites.

All OTel SDK imports are deferred (inside function bodies) so the module
loads without the OTel SDK when tracing is disabled.
"""
from typing import Optional

from vgate.logging_config import get_logger

logger = get_logger("vgate.tracing")

# Module-level state
_tracer_provider = None
_tracing_enabled = False


def init_tracing(config=None) -> None:
    """
    Initialize OpenTelemetry tracing.

    Sets up TracerProvider with OTLP exporter, BatchSpanProcessor,
    and TraceIdRatioBased sampler. No-op when config.tracing.enabled is False.

    Args:
        config: VGateConfig instance. Uses global config if None.
    """
    global _tracer_provider, _tracing_enabled

    if config is None:
        from vgate.config import get_config
        config = get_config()

    if not config.tracing.enabled:
        logger.info("Tracing disabled")
        return

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({
        "service.name": config.tracing.service_name,
        "service.version": config.version,
    })

    sampler = TraceIdRatioBased(config.tracing.sample_rate)

    _tracer_provider = TracerProvider(
        resource=resource,
        sampler=sampler,
    )

    exporter = OTLPSpanExporter(
        endpoint=config.tracing.otlp_endpoint,
        insecure=config.tracing.otlp_insecure,
    )

    _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_tracer_provider)

    _tracing_enabled = True

    logger.info(
        "Tracing initialized",
        extra={"extra_data": {
            "service_name": config.tracing.service_name,
            "otlp_endpoint": config.tracing.otlp_endpoint,
            "sample_rate": config.tracing.sample_rate,
        }}
    )


def get_tracer(name: str):
    """
    Get an OpenTelemetry tracer by name.

    Returns a no-op tracer when tracing is not initialized,
    which produces no-op spans with zero overhead.

    Args:
        name: Tracer name (e.g., "vgate.batcher").
    """
    from opentelemetry import trace
    return trace.get_tracer(name)


def shutdown_tracing() -> None:
    """Flush pending spans and shut down the tracer provider."""
    global _tracer_provider, _tracing_enabled

    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        logger.info("Tracing shut down")

    _tracer_provider = None
    _tracing_enabled = False


def get_current_trace_id() -> str:
    """
    Extract the 32-hex trace_id from the current span context.

    Returns:
        32-character hex trace_id string, or "" if no active span.
    """
    from opentelemetry import trace

    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx and ctx.trace_id != 0:
        return format(ctx.trace_id, "032x")
    return ""


def is_tracing_enabled() -> bool:
    """Return whether tracing has been initialized."""
    return _tracing_enabled
