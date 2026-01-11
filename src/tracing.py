"""
OpenTelemetry Distributed Tracing for Quantum Traffic Optimizer.

This module provides distributed tracing capabilities using OpenTelemetry.
It enables request tracing across service boundaries for debugging and
performance analysis.

Features:
- Automatic trace propagation
- Span creation for key operations
- Integration with Jaeger, Zipkin, or OTLP collectors
- Correlation with logging

Usage:
    from src.tracing import get_tracer, trace_span

    tracer = get_tracer(__name__)

    @trace_span("my_operation")
    async def my_function():
        ...
"""

import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry, make it optional
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.b3 import B3MultiFormat
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    Status = None
    StatusCode = None


# =============================================================================
# Tracer Configuration
# =============================================================================


class TracingManager:
    """
    Manages OpenTelemetry tracing configuration.

    Handles tracer initialization, span creation, and instrumentation.
    """

    def __init__(self):
        self._initialized = False
        self._tracer: Optional[Any] = None
        self._settings: Optional[Settings] = None

    @property
    def is_available(self) -> bool:
        """Check if OpenTelemetry is available."""
        return OTEL_AVAILABLE

    @property
    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        if self._settings is None:
            self._settings = get_settings()
        return self.is_available and getattr(self._settings, "TRACING_ENABLED", False)

    def initialize(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize OpenTelemetry tracing.

        Args:
            settings: Application settings.
        """
        if self._initialized:
            return

        if not self.is_available:
            logger.info("OpenTelemetry not available, tracing disabled")
            return

        self._settings = settings or get_settings()

        if not getattr(self._settings, "TRACING_ENABLED", False):
            logger.info("Tracing disabled in settings")
            return

        try:
            # Create resource with service information
            resource = Resource.create({
                "service.name": self._settings.APP_NAME,
                "service.version": self._settings.APP_VERSION,
                "deployment.environment": self._settings.ENVIRONMENT,
            })

            # Create tracer provider
            provider = TracerProvider(resource=resource)

            # Configure exporter based on settings
            exporter_type = getattr(self._settings, "TRACING_EXPORTER", "console")

            if exporter_type == "otlp":
                endpoint = getattr(self._settings, "OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
                exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            elif exporter_type == "console":
                exporter = ConsoleSpanExporter()
            else:
                logger.warning(f"Unknown exporter type: {exporter_type}, using console")
                exporter = ConsoleSpanExporter()

            # Add span processor
            provider.add_span_processor(BatchSpanProcessor(exporter))

            # Set global tracer provider
            trace.set_tracer_provider(provider)

            # Set propagator for distributed tracing
            set_global_textmap(B3MultiFormat())

            self._tracer = trace.get_tracer(
                self._settings.APP_NAME,
                self._settings.APP_VERSION
            )

            self._initialized = True
            logger.info(f"OpenTelemetry tracing initialized with {exporter_type} exporter")

        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")

    def get_tracer(self, name: str) -> Any:
        """
        Get a tracer instance.

        Args:
            name: Tracer name (usually module name).

        Returns:
            Tracer instance or NoOpTracer if not available.
        """
        if not self.is_enabled or not self._initialized:
            return NoOpTracer()

        return trace.get_tracer(name)

    def instrument_fastapi(self, app) -> None:
        """Instrument FastAPI application."""
        if not self.is_enabled or not self._initialized:
            return

        try:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI instrumented for tracing")
        except Exception as e:
            logger.warning(f"Failed to instrument FastAPI: {e}")

    def instrument_sqlalchemy(self, engine) -> None:
        """Instrument SQLAlchemy engine."""
        if not self.is_enabled or not self._initialized:
            return

        try:
            SQLAlchemyInstrumentor().instrument(engine=engine)
            logger.info("SQLAlchemy instrumented for tracing")
        except Exception as e:
            logger.warning(f"Failed to instrument SQLAlchemy: {e}")

    def instrument_redis(self) -> None:
        """Instrument Redis client."""
        if not self.is_enabled or not self._initialized:
            return

        try:
            RedisInstrumentor().instrument()
            logger.info("Redis instrumented for tracing")
        except Exception as e:
            logger.warning(f"Failed to instrument Redis: {e}")

    def instrument_httpx(self) -> None:
        """Instrument HTTPX client."""
        if not self.is_enabled or not self._initialized:
            return

        try:
            HTTPXClientInstrumentor().instrument()
            logger.info("HTTPX instrumented for tracing")
        except Exception as e:
            logger.warning(f"Failed to instrument HTTPX: {e}")


# Global tracing manager instance
tracing_manager = TracingManager()


# =============================================================================
# NoOp Tracer for when OpenTelemetry is not available
# =============================================================================


class NoOpSpan:
    """No-operation span for when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class NoOpTracer:
    """No-operation tracer for when OpenTelemetry is not available."""

    def start_as_current_span(self, name: str, **kwargs) -> NoOpSpan:
        return NoOpSpan()

    def start_span(self, name: str, **kwargs) -> NoOpSpan:
        return NoOpSpan()


# =============================================================================
# Helper Functions
# =============================================================================


def get_tracer(name: str) -> Any:
    """
    Get a tracer instance.

    Args:
        name: Tracer name (usually __name__).

    Returns:
        Tracer instance.
    """
    return tracing_manager.get_tracer(name)


def get_current_span() -> Any:
    """Get the current active span."""
    if not tracing_manager.is_enabled:
        return NoOpSpan()
    return trace.get_current_span()


@contextmanager
def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Context manager for creating a traced span.

    Args:
        name: Span name.
        attributes: Optional span attributes.

    Yields:
        Span object.

    Example:
        with trace_span("my_operation", {"key": "value"}) as span:
            span.set_attribute("result", 42)
    """
    if not tracing_manager.is_enabled:
        yield NoOpSpan()
        return

    tracer = get_tracer(__name__)
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def trace_function(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """
    Decorator for tracing function execution.

    Args:
        name: Optional span name. Defaults to function name.
        attributes: Optional span attributes.

    Example:
        @trace_function("process_data")
        async def process_data(data):
            ...
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with trace_span(span_name, attributes) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace_span(span_name, attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# Initialization
# =============================================================================


def init_tracing(settings: Optional[Settings] = None) -> None:
    """Initialize tracing on application startup."""
    tracing_manager.initialize(settings)


def instrument_app(app) -> None:
    """Instrument a FastAPI application."""
    tracing_manager.instrument_fastapi(app)


def instrument_all() -> None:
    """Instrument all supported libraries."""
    tracing_manager.instrument_httpx()
    tracing_manager.instrument_redis()
