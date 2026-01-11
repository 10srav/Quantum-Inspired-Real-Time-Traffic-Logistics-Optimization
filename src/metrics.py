"""
Prometheus Metrics and Monitoring for Quantum Traffic Optimizer.

This module provides:
- Prometheus metrics collection
- Application health monitoring
- Performance tracking
- Custom business metrics
"""

import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

from .config import Settings, get_settings

# Try to import prometheus_client, make it optional
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# =========================
# Metric Definitions
# =========================

if PROMETHEUS_AVAILABLE:
    # Application Info
    APP_INFO = Info(
        "quantum_traffic_optimizer",
        "Quantum Traffic Optimizer application information"
    )

    # Request Metrics
    REQUEST_COUNT = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"]
    )

    REQUEST_LATENCY = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )

    REQUEST_IN_PROGRESS = Gauge(
        "http_requests_in_progress",
        "Number of HTTP requests currently in progress",
        ["method", "endpoint"]
    )

    # Optimization Metrics
    OPTIMIZATION_COUNT = Counter(
        "optimization_requests_total",
        "Total optimization requests",
        ["traffic_level", "algorithm", "status"]
    )

    OPTIMIZATION_DURATION = Histogram(
        "optimization_duration_seconds",
        "Time spent on route optimization",
        ["algorithm", "n_deliveries"],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
    )

    OPTIMIZATION_IMPROVEMENT = Histogram(
        "optimization_improvement_percentage",
        "Percentage improvement over greedy baseline",
        ["algorithm"],
        buckets=[0, 5, 10, 15, 20, 25, 30, 40, 50]
    )

    DELIVERIES_PER_REQUEST = Histogram(
        "deliveries_per_request",
        "Number of deliveries per optimization request",
        buckets=[1, 2, 3, 5, 7, 10, 15, 20]
    )

    # Route Metrics
    ROUTES_CACHED = Gauge(
        "routes_cached_total",
        "Number of routes currently in cache"
    )

    ROUTE_DISTANCE = Histogram(
        "route_total_distance_meters",
        "Total route distance in meters",
        buckets=[500, 1000, 2000, 5000, 10000, 20000, 50000]
    )

    ROUTE_ETA = Histogram(
        "route_total_eta_minutes",
        "Total route ETA in minutes",
        buckets=[5, 10, 15, 30, 45, 60, 90, 120]
    )

    # QAOA Metrics
    QAOA_ITERATIONS = Histogram(
        "qaoa_iterations",
        "Number of QAOA iterations",
        buckets=[10, 50, 100, 200, 500, 1000]
    )

    QAOA_LAYERS = Gauge(
        "qaoa_layers_configured",
        "Number of QAOA layers configured"
    )

    # System Metrics
    WEBSOCKET_CONNECTIONS = Gauge(
        "websocket_connections_active",
        "Number of active WebSocket connections"
    )

    GRAPH_LOADED = Gauge(
        "osm_graph_loaded",
        "Whether OSM graph is loaded (1=yes, 0=no)"
    )

    GRAPH_NODES = Gauge(
        "osm_graph_nodes_total",
        "Number of nodes in the OSM graph"
    )

    GRAPH_EDGES = Gauge(
        "osm_graph_edges_total",
        "Number of edges in the OSM graph"
    )

    # Error Metrics
    ERROR_COUNT = Counter(
        "errors_total",
        "Total errors by type",
        ["error_type", "endpoint"]
    )

    # Rate Limiting Metrics
    RATE_LIMIT_HITS = Counter(
        "rate_limit_hits_total",
        "Number of rate limit hits",
        ["endpoint"]
    )

    # Authentication Metrics
    AUTH_ATTEMPTS = Counter(
        "auth_attempts_total",
        "Authentication attempts",
        ["method", "status"]
    )

    # Cache Metrics
    CACHE_HITS = Counter(
        "cache_hits_total",
        "Cache hits",
        ["cache_type"]
    )

    CACHE_MISSES = Counter(
        "cache_misses_total",
        "Cache misses",
        ["cache_type"]
    )


# =========================
# Metric Helper Functions
# =========================

def set_app_info(settings: Settings) -> None:
    """Set application info metric."""
    if not PROMETHEUS_AVAILABLE or not settings.METRICS_ENABLED:
        return

    APP_INFO.info({
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "qaoa_layers": str(settings.QAOA_LAYERS),
        "qaoa_enabled": str(settings.USE_QAOA_IN_API),
    })


def track_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration: float
) -> None:
    """Track HTTP request metrics."""
    if not PROMETHEUS_AVAILABLE:
        return

    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status_code=str(status_code)
    ).inc()

    REQUEST_LATENCY.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)


def track_optimization(
    traffic_level: str,
    algorithm: str,
    n_deliveries: int,
    duration: float,
    improvement: Optional[float],
    total_distance: float,
    total_eta: float,
    success: bool = True
) -> None:
    """Track optimization metrics."""
    if not PROMETHEUS_AVAILABLE:
        return

    status = "success" if success else "failure"
    OPTIMIZATION_COUNT.labels(
        traffic_level=traffic_level,
        algorithm=algorithm,
        status=status
    ).inc()

    OPTIMIZATION_DURATION.labels(
        algorithm=algorithm,
        n_deliveries=str(min(n_deliveries, 20))  # Cap for label cardinality
    ).observe(duration)

    DELIVERIES_PER_REQUEST.observe(n_deliveries)
    ROUTE_DISTANCE.observe(total_distance)
    ROUTE_ETA.observe(total_eta)

    if improvement is not None:
        OPTIMIZATION_IMPROVEMENT.labels(algorithm=algorithm).observe(improvement)


def track_error(error_type: str, endpoint: str) -> None:
    """Track error metrics."""
    if not PROMETHEUS_AVAILABLE:
        return

    ERROR_COUNT.labels(
        error_type=error_type,
        endpoint=endpoint
    ).inc()


def track_rate_limit(endpoint: str) -> None:
    """Track rate limit hit."""
    if not PROMETHEUS_AVAILABLE:
        return

    RATE_LIMIT_HITS.labels(endpoint=endpoint).inc()


def track_auth(method: str, success: bool) -> None:
    """Track authentication attempt."""
    if not PROMETHEUS_AVAILABLE:
        return

    AUTH_ATTEMPTS.labels(
        method=method,
        status="success" if success else "failure"
    ).inc()


def track_cache(cache_type: str, hit: bool) -> None:
    """Track cache hit/miss."""
    if not PROMETHEUS_AVAILABLE:
        return

    if hit:
        CACHE_HITS.labels(cache_type=cache_type).inc()
    else:
        CACHE_MISSES.labels(cache_type=cache_type).inc()


def update_routes_cached(count: int) -> None:
    """Update cached routes gauge."""
    if not PROMETHEUS_AVAILABLE:
        return

    ROUTES_CACHED.set(count)


def update_websocket_connections(count: int) -> None:
    """Update WebSocket connections gauge."""
    if not PROMETHEUS_AVAILABLE:
        return

    WEBSOCKET_CONNECTIONS.set(count)


def update_graph_metrics(loaded: bool, nodes: int = 0, edges: int = 0) -> None:
    """Update graph metrics."""
    if not PROMETHEUS_AVAILABLE:
        return

    GRAPH_LOADED.set(1 if loaded else 0)
    if loaded:
        GRAPH_NODES.set(nodes)
        GRAPH_EDGES.set(edges)


def update_qaoa_layers(layers: int) -> None:
    """Update QAOA layers gauge."""
    if not PROMETHEUS_AVAILABLE:
        return

    QAOA_LAYERS.set(layers)


# =========================
# Metric Decorators
# =========================

def timed(metric_name: Optional[str] = None):
    """
    Decorator to time function execution.

    Args:
        metric_name: Optional custom metric name.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if PROMETHEUS_AVAILABLE:
                    name = metric_name or func.__name__
                    OPTIMIZATION_DURATION.labels(
                        algorithm=name,
                        n_deliveries="n/a"
                    ).observe(duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if PROMETHEUS_AVAILABLE:
                    name = metric_name or func.__name__
                    OPTIMIZATION_DURATION.labels(
                        algorithm=name,
                        n_deliveries="n/a"
                    ).observe(duration)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def counted(counter_name: str, labels: Optional[Dict[str, str]] = None):
    """
    Decorator to count function calls.

    Args:
        counter_name: Name for the counter.
        labels: Optional labels for the counter.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if PROMETHEUS_AVAILABLE:
                REQUEST_COUNT.labels(**labels or {}).inc()
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if PROMETHEUS_AVAILABLE:
                REQUEST_COUNT.labels(**labels or {}).inc()
            return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =========================
# Metrics Endpoint
# =========================

def get_metrics() -> bytes:
    """
    Generate Prometheus metrics output.

    Returns:
        Prometheus metrics in text format.
    """
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus client not installed\n"

    return generate_latest()


def get_metrics_content_type() -> str:
    """Get content type for metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        return "text/plain"
    return CONTENT_TYPE_LATEST


# =========================
# Health Check
# =========================

class HealthChecker:
    """
    Application health checker.

    Aggregates health status from multiple components.
    """

    def __init__(self):
        self.checks: Dict[str, Callable[[], bool]] = {}

    def register(self, name: str, check: Callable[[], bool]) -> None:
        """Register a health check function."""
        self.checks[name] = check

    async def check_all(self) -> Dict[str, Any]:
        """
        Run all health checks.

        Returns:
            Health status dictionary.
        """
        results = {}
        all_healthy = True

        for name, check in self.checks.items():
            try:
                import asyncio
                if asyncio.iscoroutinefunction(check):
                    healthy = await check()
                else:
                    healthy = check()
                results[name] = {
                    "status": "healthy" if healthy else "unhealthy",
                    "healthy": healthy
                }
                if not healthy:
                    all_healthy = False
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "healthy": False,
                    "error": str(e)
                }
                all_healthy = False

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": results
        }


# Global health checker instance
health_checker = HealthChecker()


# =========================
# Metrics Middleware
# =========================

class MetricsMiddleware:
    """
    Middleware to collect HTTP request metrics.

    Tracks request count, latency, and in-progress requests.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not PROMETHEUS_AVAILABLE:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")

        # Normalize path for metrics (remove IDs)
        endpoint = self._normalize_path(path)

        # Track in-progress
        REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()

        start_time = time.time()
        status_code = 500  # Default if not set

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.time() - start_time

            REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

    def _normalize_path(self, path: str) -> str:
        """
        Normalize path by replacing IDs with placeholders.

        This prevents high cardinality in metrics.
        """
        import re

        # Replace UUIDs
        path = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "{id}",
            path
        )

        # Replace numeric IDs
        path = re.sub(r"/\d+", "/{id}", path)

        # Replace route IDs (8-char alphanumeric)
        path = re.sub(r"/[a-zA-Z0-9]{8}$", "/{route_id}", path)

        return path
