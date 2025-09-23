"""OpenTelemetry instrumentation for golden testset operations.

This module provides comprehensive tracing for:
1. Database operations
2. Generation workflows
3. Validation pipelines
4. Phoenix uploads
5. Cost tracking
"""

import asyncio
import functools
import logging
import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
# Optional instrumentation packages (not required for core functionality)
try:
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    HAS_AIOHTTP_INST = True
except ImportError:
    HAS_AIOHTTP_INST = False

try:
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    HAS_ASYNCPG_INST = True
except ImportError:
    HAS_ASYNCPG_INST = False
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class TracingConfig:
    """Configuration for OpenTelemetry tracing."""

    def __init__(self):
        self.service_name = os.getenv("OTEL_SERVICE_NAME", "golden-testset-manager")
        self.otlp_endpoint = os.getenv("PHOENIX_OTLP_ENDPOINT", "http://localhost:4317")
        self.phoenix_endpoint = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006")
        self.enable_tracing = os.getenv("ENABLE_TRACING", "true").lower() == "true"
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.version = "1.0.0"


class GoldenTestsetTracer:
    """Manages OpenTelemetry tracing for golden testset operations."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.config = TracingConfig()
            self.tracer = None
            self.meter = None
            self.metrics = {}

            if self.config.enable_tracing or self.config.enable_metrics:
                self._setup_telemetry()

            self._initialized = True

    def _setup_telemetry(self):
        """Setup OpenTelemetry tracing and metrics."""
        # Create resource
        resource = Resource.create(
            {
                "service.name": self.config.service_name,
                "service.version": self.config.version,
                "environment": self.config.environment,
                "telemetry.sdk.language": "python",
                "telemetry.sdk.name": "opentelemetry",
            }
        )

        if self.config.enable_tracing:
            self._setup_tracing(resource)

        if self.config.enable_metrics:
            self._setup_metrics(resource)

        # Auto-instrument libraries
        self._setup_instrumentation()

        logger.info(
            f"OpenTelemetry initialized: tracing={self.config.enable_tracing}, "
            f"metrics={self.config.enable_metrics}"
        )

    def _setup_tracing(self, resource: Resource):
        """Setup tracing with OTLP exporter."""
        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Configure OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=self.config.otlp_endpoint,
            insecure=True,
        )

        # Add batch processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(span_processor)

        # Set global tracer provider
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(__name__, self.config.version)

    def _setup_metrics(self, resource: Resource):
        """Setup metrics with OTLP exporter."""
        # Configure metric exporter
        metric_exporter = OTLPMetricExporter(
            endpoint=self.config.otlp_endpoint,
            insecure=True,
        )

        # Create metric reader
        metric_reader = PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=10000,  # 10 seconds
        )

        # Create meter provider
        provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader],
        )

        # Set global meter provider
        metrics.set_meter_provider(provider)
        self.meter = metrics.get_meter(__name__, self.config.version)

        # Create metrics
        self._create_metrics()

    def _create_metrics(self):
        """Create application metrics."""
        if not self.meter:
            return

        # Counters
        self.metrics["testset_created"] = self.meter.create_counter(
            "golden_testset.created", unit="1", description="Number of testsets created"
        )

        self.metrics["validation_failed"] = self.meter.create_counter(
            "golden_testset.validation.failed",
            unit="1",
            description="Number of validation failures",
        )

        self.metrics["api_calls"] = self.meter.create_counter(
            "golden_testset.api.calls", unit="1", description="Number of API calls made"
        )

        # Histograms
        self.metrics["generation_duration"] = self.meter.create_histogram(
            "golden_testset.generation.duration",
            unit="s",
            description="Duration of testset generation",
        )

        self.metrics["validation_duration"] = self.meter.create_histogram(
            "golden_testset.validation.duration",
            unit="s",
            description="Duration of validation pipeline",
        )

        self.metrics["upload_duration"] = self.meter.create_histogram(
            "golden_testset.upload.duration",
            unit="s",
            description="Duration of Phoenix upload",
        )

        self.metrics["token_usage"] = self.meter.create_histogram(
            "golden_testset.tokens.used", unit="1", description="Number of tokens used"
        )

        self.metrics["cost_per_generation"] = self.meter.create_histogram(
            "golden_testset.cost", unit="USD", description="Cost per generation in USD"
        )

    def _setup_instrumentation(self):
        """Setup automatic instrumentation for libraries."""
        # Instrument AsyncPG for database operations if available
        if HAS_ASYNCPG_INST:
            AsyncPGInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
            logger.info("AsyncPG instrumentation enabled")

        # Instrument aiohttp for HTTP requests if available
        if HAS_AIOHTTP_INST:
            AioHttpClientInstrumentor().instrument(
                tracer_provider=trace.get_tracer_provider()
            )
            logger.info("AioHTTP client instrumentation enabled")

        if not HAS_ASYNCPG_INST and not HAS_AIOHTTP_INST:
            logger.warning("No instrumentation packages available, manual tracing only")

    @contextmanager
    def span(
        self,
        name: str,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ):
        """Create a new span context manager.

        Args:
            name: Span name
            kind: Span kind
            attributes: Span attributes

        Yields:
            Span object
        """
        if not self.tracer:
            yield None
            return

        with self.tracer.start_as_current_span(
            name, kind=kind, attributes=attributes or {}
        ) as span:
            yield span

    def trace_async(
        self,
        name: str | None = None,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ):
        """Decorator for tracing async functions.

        Args:
            name: Span name (defaults to function name)
            kind: Span kind
            attributes: Additional span attributes
        """

        def decorator(func: F) -> F:
            span_name = name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.tracer:
                    return await func(*args, **kwargs)

                with self.tracer.start_as_current_span(
                    span_name, kind=kind, attributes=attributes or {}
                ) as span:
                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            return wrapper

        return decorator

    def trace_sync(
        self,
        name: str | None = None,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ):
        """Decorator for tracing sync functions.

        Args:
            name: Span name (defaults to function name)
            kind: Span kind
            attributes: Additional span attributes
        """

        def decorator(func: F) -> F:
            span_name = name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.tracer:
                    return func(*args, **kwargs)

                with self.tracer.start_as_current_span(
                    span_name, kind=kind, attributes=attributes or {}
                ) as span:
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            return wrapper

        return decorator

    def record_metric(
        self,
        metric_name: str,
        value: int | float,
        attributes: dict[str, Any] | None = None,
    ):
        """Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            attributes: Metric attributes
        """
        if metric_name not in self.metrics:
            logger.warning(f"Unknown metric: {metric_name}")
            return

        metric = self.metrics[metric_name]
        attrs = attributes or {}

        if hasattr(metric, "add"):
            # Counter
            metric.add(value, attrs)
        elif hasattr(metric, "record"):
            # Histogram
            metric.record(value, attrs)

    def get_current_span(self) -> Span | None:
        """Get the current active span.

        Returns:
            Current span or None
        """
        if not self.tracer:
            return None

        return trace.get_current_span()

    def add_span_event(self, name: str, attributes: dict[str, Any] | None = None):
        """Add an event to the current span.

        Args:
            name: Event name
            attributes: Event attributes
        """
        span = self.get_current_span()
        if span:
            span.add_event(name, attributes=attributes or {})

    def set_span_attribute(self, key: str, value: Any):
        """Set an attribute on the current span.

        Args:
            key: Attribute key
            value: Attribute value
        """
        span = self.get_current_span()
        if span:
            span.set_attribute(key, value)

    def set_span_status(self, status: Status):
        """Set the status of the current span.

        Args:
            status: Span status
        """
        span = self.get_current_span()
        if span:
            span.set_status(status)

    def inject_context(self, carrier: dict[str, Any]):
        """Inject trace context into a carrier for propagation.

        Args:
            carrier: Dictionary to inject context into
        """
        inject(carrier)

    def extract_context(self, carrier: dict[str, Any]):
        """Extract trace context from a carrier.

        Args:
            carrier: Dictionary to extract context from
        """
        return extract(carrier)


# Global tracer instance
tracer = GoldenTestsetTracer()


# Convenience decorators
def trace_async(
    name: str | None = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
):
    """Decorator for tracing async functions."""
    return tracer.trace_async(name, kind, attributes)


def trace_sync(
    name: str | None = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
):
    """Decorator for tracing sync functions."""
    return tracer.trace_sync(name, kind, attributes)


# Specialized tracing decorators
def trace_database(name: str | None = None):
    """Decorator for database operations."""
    return trace_async(
        name=name, kind=trace.SpanKind.CLIENT, attributes={"db.system": "postgresql"}
    )


def trace_api_call(service: str, operation: str):
    """Decorator for external API calls."""
    return trace_async(
        name=f"{service}.{operation}",
        kind=trace.SpanKind.CLIENT,
        attributes={"service.name": service, "operation": operation},
    )


def trace_validation(validation_type: str):
    """Decorator for validation operations."""
    return trace_async(
        name=f"validation.{validation_type}",
        attributes={"validation.type": validation_type},
    )


def trace_generation(generation_type: str = "testset"):
    """Decorator for generation operations."""
    return trace_async(
        name=f"generation.{generation_type}",
        attributes={"generation.type": generation_type},
    )


# Context managers for manual tracing
@contextmanager
def trace_operation(name: str, attributes: dict[str, Any] | None = None):
    """Context manager for tracing operations.

    Args:
        name: Operation name
        attributes: Operation attributes
    """
    with tracer.span(name, attributes=attributes) as span:
        yield span


@contextmanager
def trace_batch_operation(
    operation: str, batch_size: int, attributes: dict[str, Any] | None = None
):
    """Context manager for tracing batch operations.

    Args:
        operation: Operation name
        batch_size: Size of the batch
        attributes: Additional attributes
    """
    attrs = {"batch.size": batch_size}
    if attributes:
        attrs.update(attributes)

    with tracer.span(f"batch.{operation}", attributes=attrs) as span:
        yield span


# Metrics recording helpers
def record_testset_created(version: str, size: int):
    """Record testset creation metric."""
    tracer.record_metric("testset_created", 1, {"version": version, "size": size})


def record_validation_failed(validation_type: str, reason: str):
    """Record validation failure metric."""
    tracer.record_metric(
        "validation_failed", 1, {"type": validation_type, "reason": reason}
    )


def record_api_call(service: str, operation: str, status: str):
    """Record API call metric."""
    tracer.record_metric(
        "api_calls", 1, {"service": service, "operation": operation, "status": status}
    )


def record_generation_duration(
    duration_seconds: float, generation_type: str = "testset"
):
    """Record generation duration metric."""
    tracer.record_metric(
        "generation_duration", duration_seconds, {"type": generation_type}
    )


def record_token_usage(tokens: int, model: str):
    """Record token usage metric."""
    tracer.record_metric("token_usage", tokens, {"model": model})


def record_generation_cost(cost_usd: float, model: str):
    """Record generation cost metric."""
    tracer.record_metric("cost_per_generation", cost_usd, {"model": model})


async def test_tracing():
    """Test tracing functionality."""

    @trace_async()
    async def sample_operation():
        tracer.add_span_event("Starting operation")
        await asyncio.sleep(0.1)
        tracer.set_span_attribute("result", "success")
        return "completed"

    @trace_database("test_query")
    async def database_operation():
        await asyncio.sleep(0.05)
        return ["row1", "row2"]

    @trace_api_call("openai", "generate")
    async def api_operation():
        await asyncio.sleep(0.2)
        record_api_call("openai", "generate", "success")
        record_token_usage(1500, "gpt-4.1-mini")
        return {"response": "generated text"}

    logger.info("Testing tracing functionality...")

    # Run test operations
    with trace_operation("test_suite"):
        result1 = await sample_operation()
        logger.info(f"Sample operation: {result1}")

        result2 = await database_operation()
        logger.info(f"Database operation: {len(result2)} rows")

        result3 = await api_operation()
        logger.info(f"API operation: {result3}")

        # Record test metrics
        record_testset_created("1.0.0", 10)
        record_generation_duration(5.2)
        record_generation_cost(0.15, "gpt-4.1-mini")

    logger.info("Tracing test completed")


if __name__ == "__main__":
    import sys

    # Run test
    asyncio.run(test_tracing())
    logger.info(f"View traces at: {tracer.config.phoenix_endpoint}")
    sys.exit(0)
