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
Pytest configuration for V-Gate tests.
"""
import importlib.util
import itertools
import sys
import types
from contextvars import ContextVar

import pytest

# Configure pytest-asyncio
pytest_plugins = ['pytest_asyncio']


def _install_otel_stub_if_missing() -> None:
    """Install a minimal OpenTelemetry stub when dependency is unavailable."""
    if importlib.util.find_spec("opentelemetry") is not None:
        return
    if "opentelemetry" in sys.modules:
        return

    current_span_var: ContextVar = ContextVar("otel_current_span", default=None)
    current_ctx_var: ContextVar = ContextVar("otel_context", default={})
    trace_id_counter = itertools.count(1)
    span_id_counter = itertools.count(1)
    state = {"provider": None}

    class SpanContext:
        def __init__(self, trace_id: int = 0, span_id: int = 0):
            self.trace_id = trace_id
            self.span_id = span_id

    class Span:
        def __init__(self, trace_id: int = 0, span_id: int = 0):
            self._ctx = SpanContext(trace_id=trace_id, span_id=span_id)
            self.attributes = {}

        def get_span_context(self):
            return self._ctx

        def set_attribute(self, key, value) -> None:
            self.attributes[key] = value

    class SpanManager:
        def __init__(self, name: str):
            self.name = name
            self._span = Span(
                trace_id=next(trace_id_counter),
                span_id=next(span_id_counter),
            )
            self._token = None

        def __enter__(self):
            self._token = current_span_var.set(self._span)
            return self._span

        def __exit__(self, exc_type, exc, tb):
            if self._token is not None:
                current_span_var.reset(self._token)
            return False

    class Tracer:
        def __init__(self, name: str):
            self.name = name

        def start_as_current_span(self, name: str):
            return SpanManager(name)

    class TracerProvider:
        def __init__(self, resource=None, sampler=None):
            self.resource = resource
            self.sampler = sampler
            self.span_processors = []
            self.shutdown_called = False

        def add_span_processor(self, span_processor) -> None:
            self.span_processors.append(span_processor)

        def shutdown(self) -> None:
            self.shutdown_called = True

    class BatchSpanProcessor:
        def __init__(self, exporter):
            self.exporter = exporter

    class OTLPSpanExporter:
        def __init__(self, endpoint=None, insecure=None):
            self.endpoint = endpoint
            self.insecure = insecure

    class TraceIdRatioBased:
        def __init__(self, rate: float):
            self.rate = rate

    class Resource(dict):
        @classmethod
        def create(cls, attributes):
            return cls(attributes)

    class FastAPIInstrumentor:
        @staticmethod
        def instrument_app(app) -> None:
            app.state.otel_instrumented = True

    def get_current_span():
        span = current_span_var.get()
        if span is None:
            return Span()
        return span

    def get_tracer(name: str):
        return Tracer(name)

    def set_tracer_provider(provider) -> None:
        state["provider"] = provider

    def get_current():
        return current_ctx_var.get()

    def attach(ctx):
        return current_ctx_var.set(ctx)

    def detach(token) -> None:
        current_ctx_var.reset(token)

    otel_mod = types.ModuleType("opentelemetry")
    otel_mod.__path__ = []
    trace_mod = types.ModuleType("opentelemetry.trace")
    context_mod = types.ModuleType("opentelemetry.context")

    sdk_mod = types.ModuleType("opentelemetry.sdk")
    sdk_mod.__path__ = []
    sdk_trace_mod = types.ModuleType("opentelemetry.sdk.trace")
    sdk_trace_export_mod = types.ModuleType("opentelemetry.sdk.trace.export")
    sdk_trace_sampling_mod = types.ModuleType("opentelemetry.sdk.trace.sampling")
    sdk_resources_mod = types.ModuleType("opentelemetry.sdk.resources")

    exporter_mod = types.ModuleType("opentelemetry.exporter")
    exporter_mod.__path__ = []
    exporter_otlp_mod = types.ModuleType("opentelemetry.exporter.otlp")
    exporter_otlp_mod.__path__ = []
    exporter_otlp_proto_mod = types.ModuleType("opentelemetry.exporter.otlp.proto")
    exporter_otlp_proto_mod.__path__ = []
    exporter_otlp_proto_grpc_mod = types.ModuleType("opentelemetry.exporter.otlp.proto.grpc")
    exporter_otlp_proto_grpc_mod.__path__ = []
    exporter_trace_mod = types.ModuleType("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")

    instrumentation_mod = types.ModuleType("opentelemetry.instrumentation")
    instrumentation_mod.__path__ = []
    instrumentation_fastapi_mod = types.ModuleType("opentelemetry.instrumentation.fastapi")

    trace_mod.get_tracer = get_tracer
    trace_mod.get_current_span = get_current_span
    trace_mod.set_tracer_provider = set_tracer_provider

    context_mod.get_current = get_current
    context_mod.attach = attach
    context_mod.detach = detach

    sdk_trace_mod.TracerProvider = TracerProvider
    sdk_trace_export_mod.BatchSpanProcessor = BatchSpanProcessor
    sdk_trace_sampling_mod.TraceIdRatioBased = TraceIdRatioBased
    sdk_resources_mod.Resource = Resource

    exporter_trace_mod.OTLPSpanExporter = OTLPSpanExporter

    instrumentation_fastapi_mod.FastAPIInstrumentor = FastAPIInstrumentor

    otel_mod.trace = trace_mod
    otel_mod.context = context_mod
    sdk_mod.trace = sdk_trace_mod
    sdk_mod.resources = sdk_resources_mod
    sdk_trace_mod.export = sdk_trace_export_mod
    sdk_trace_mod.sampling = sdk_trace_sampling_mod

    sys.modules["opentelemetry"] = otel_mod
    sys.modules["opentelemetry.trace"] = trace_mod
    sys.modules["opentelemetry.context"] = context_mod
    sys.modules["opentelemetry.sdk"] = sdk_mod
    sys.modules["opentelemetry.sdk.trace"] = sdk_trace_mod
    sys.modules["opentelemetry.sdk.trace.export"] = sdk_trace_export_mod
    sys.modules["opentelemetry.sdk.trace.sampling"] = sdk_trace_sampling_mod
    sys.modules["opentelemetry.sdk.resources"] = sdk_resources_mod
    sys.modules["opentelemetry.exporter"] = exporter_mod
    sys.modules["opentelemetry.exporter.otlp"] = exporter_otlp_mod
    sys.modules["opentelemetry.exporter.otlp.proto"] = exporter_otlp_proto_mod
    sys.modules["opentelemetry.exporter.otlp.proto.grpc"] = exporter_otlp_proto_grpc_mod
    sys.modules["opentelemetry.exporter.otlp.proto.grpc.trace_exporter"] = exporter_trace_mod
    sys.modules["opentelemetry.instrumentation"] = instrumentation_mod
    sys.modules["opentelemetry.instrumentation.fastapi"] = instrumentation_fastapi_mod


def _install_sglang_stub_if_missing() -> None:
    """Install a minimal SGLang stub when dependency is unavailable."""
    if importlib.util.find_spec("sglang") is not None:
        return
    if "sglang" in sys.modules:
        return

    class MockEngine:
        def __init__(self, model_path=None, mem_fraction_static=None, tp_size=None):
            self.model_path = model_path

        def generate(self, prompts, sampling_params):
            results = []
            for prompt in prompts:
                results.append({
                    "text": f"[sglang-stub] echo: {prompt[:80]}",
                    "meta_info": {
                        "completion_tokens_ids": list(range(8)),
                    },
                })
            return results

        def shutdown(self):
            pass

    sglang_mod = types.ModuleType("sglang")
    sglang_mod.Engine = MockEngine
    sys.modules["sglang"] = sglang_mod


_install_otel_stub_if_missing()
_install_sglang_stub_if_missing()


@pytest.fixture(autouse=True)
def reset_tracing_state():
    """Reset tracing global state between tests."""
    from vgate.tracing import shutdown_tracing

    shutdown_tracing()
    yield
    shutdown_tracing()
