"""
src/telemetry/tracing.py
-------------------------
OpenTelemetry instrumentation for the API and pipeline.

Exports traces to an OTLP collector (e.g. Jaeger, Grafana Tempo).
Set OTEL_EXPORTER_OTLP_ENDPOINT to point to your collector.

Usage
-----
Call configure_tracing() once at startup in api/main.py and flows.py.
Use the @trace_span decorator on any function you want to instrument.
"""

from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable

logger = logging.getLogger(__name__)

_TRACER = None
_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "churn-prediction")
_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")


def configure_tracing() -> bool:
    """
    Initialise OpenTelemetry tracing. Returns True if successful,
    False if opentelemetry packages are not installed or no endpoint set.
    """
    global _TRACER

    if not _ENDPOINT:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set — tracing disabled.")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": _SERVICE_NAME})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=_ENDPOINT, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer(_SERVICE_NAME)
        logger.info("OpenTelemetry tracing enabled → %s", _ENDPOINT)
        return True

    except ImportError:
        logger.warning(
            "opentelemetry packages not installed. " "Run: pip install opentelemetry-sdk opentelemetry-exporter-otlp"
        )
        return False


def get_tracer():
    """Return the active tracer, or a no-op tracer if not configured."""
    global _TRACER
    if _TRACER is None:
        try:
            from opentelemetry import trace

            _TRACER = trace.get_tracer(_SERVICE_NAME)
        except ImportError:
            return _NoOpTracer()
    return _TRACER


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None):
    """Context manager to wrap a block of code in an OTel span."""
    tracer = get_tracer()
    try:
        with tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, str(v))
            yield span
    except AttributeError:
        # No-op fallback
        yield None


def trace_span_decorator(name: str | None = None, attributes: dict | None = None):
    """Decorator to wrap a function in an OTel span."""

    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with trace_span(span_name, attributes):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class _NoOpTracer:
    """Silent fallback when OTel is not configured."""

    def start_as_current_span(self, name, **kwargs):
        return _NoOpSpan()


class _NoOpSpan:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, *args):
        pass
