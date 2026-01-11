"""
Structured Logging Configuration for Quantum Traffic Optimizer.

This module provides structured logging with:
- JSON formatting for production
- Correlation IDs for request tracing
- Context propagation
- Log level configuration
"""

import json
import logging
import sys
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from .config import Settings, get_settings


# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
request_context_var: ContextVar[Dict[str, Any]] = ContextVar("request_context", default={})


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set the correlation ID for the current context.

    Args:
        correlation_id: Optional ID to set. Generates new one if not provided.

    Returns:
        The correlation ID that was set.
    """
    if correlation_id is None:
        correlation_id = str(uuid4())
    correlation_id_var.set(correlation_id)
    return correlation_id


def set_request_context(**kwargs) -> None:
    """Set additional request context for logging."""
    ctx = request_context_var.get()
    ctx.update(kwargs)
    request_context_var.set(ctx)


def clear_context() -> None:
    """Clear logging context."""
    correlation_id_var.set(None)
    request_context_var.set({})


class JsonFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Outputs logs in JSON format suitable for log aggregation
    systems like ELK, Splunk, or CloudWatch.
    """

    def __init__(self, settings: Optional[Settings] = None):
        super().__init__()
        self.settings = settings or get_settings()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log data
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if present
        correlation_id = get_correlation_id()
        if correlation_id:
            log_data["correlation_id"] = correlation_id

        # Add request context
        request_context = request_context_var.get()
        if request_context:
            log_data["context"] = request_context

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Add extra fields from record
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName"
            }
        }
        if extra_fields:
            log_data["extra"] = extra_fields

        # Add environment info
        log_data["environment"] = self.settings.ENVIRONMENT
        log_data["app_version"] = self.settings.APP_VERSION

        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter for development.

    Includes correlation IDs and colored output for terminals.
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as colored text."""
        # Get correlation ID
        correlation_id = get_correlation_id()
        cid_part = f"[{correlation_id[:8]}] " if correlation_id else ""

        # Format timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Format level with color
        level = record.levelname
        if self.use_colors:
            color = self.COLORS.get(level, "")
            level = f"{color}{level:8}{self.RESET}"
        else:
            level = f"{level:8}"

        # Format message
        message = record.getMessage()

        # Build log line
        log_line = f"{timestamp} {level} {cid_part}[{record.name}] {message}"

        # Add exception info
        if record.exc_info:
            log_line += "\n" + "".join(traceback.format_exception(*record.exc_info))

        return log_line


class CorrelationIdFilter(logging.Filter):
    """
    Logging filter that adds correlation ID to all records.

    This ensures correlation IDs are available even for
    standard library logging calls.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to record."""
        record.correlation_id = get_correlation_id() or "-"
        return True


def setup_logging(settings: Optional[Settings] = None) -> None:
    """
    Configure application logging.

    Args:
        settings: Application settings. Uses default if not provided.
    """
    settings = settings or get_settings()

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))

    # Set formatter based on configuration
    if settings.LOG_FORMAT == "json":
        formatter = JsonFormatter(settings)
    else:
        # Use colors in development
        use_colors = settings.is_development and sys.stdout.isatty()
        formatter = TextFormatter(use_colors=use_colors)

    console_handler.setFormatter(formatter)
    console_handler.addFilter(CorrelationIdFilter())

    root_logger.addHandler(console_handler)

    # Set log levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Log startup message
    logging.info(
        "Logging configured",
        extra={
            "log_level": settings.LOG_LEVEL,
            "log_format": settings.LOG_FORMAT,
            "environment": settings.ENVIRONMENT
        }
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name (usually __name__).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.addFilter(CorrelationIdFilter())
    return logger


# =========================
# Logging Context Manager
# =========================

class LogContext:
    """
    Context manager for scoped logging context.

    Usage:
        with LogContext(user_id="123", action="optimize"):
            logger.info("Starting optimization")
            # ... do work ...
            logger.info("Completed")
    """

    def __init__(self, correlation_id: Optional[str] = None, **kwargs):
        """
        Initialize logging context.

        Args:
            correlation_id: Optional correlation ID to use.
            **kwargs: Additional context fields.
        """
        self.correlation_id = correlation_id
        self.context = kwargs
        self._previous_correlation_id = None
        self._previous_context = None

    def __enter__(self) -> "LogContext":
        """Enter context and set logging variables."""
        self._previous_correlation_id = get_correlation_id()
        self._previous_context = request_context_var.get().copy()

        if self.correlation_id:
            set_correlation_id(self.correlation_id)
        elif not self._previous_correlation_id:
            set_correlation_id()

        if self.context:
            set_request_context(**self.context)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore previous state."""
        if self._previous_correlation_id:
            correlation_id_var.set(self._previous_correlation_id)
        if self._previous_context is not None:
            request_context_var.set(self._previous_context)


# =========================
# Request Logging Middleware
# =========================

class RequestLoggingMiddleware:
    """
    Middleware for request/response logging.

    Logs:
    - Incoming requests with correlation IDs
    - Response status and timing
    - Errors and exceptions
    """

    def __init__(self, app):
        self.app = app
        self.logger = get_logger("quantum_traffic.requests")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract or generate correlation ID
        headers = dict(scope.get("headers", []))
        correlation_id = (
            headers.get(b"x-correlation-id", b"").decode() or
            headers.get(b"x-request-id", b"").decode() or
            str(uuid4())
        )

        # Set context
        set_correlation_id(correlation_id)
        set_request_context(
            method=scope.get("method", ""),
            path=scope.get("path", ""),
            client_ip=self._get_client_ip(scope)
        )

        # Track timing
        import time
        start_time = time.time()

        # Log request
        self.logger.info(
            f"{scope.get('method', '')} {scope.get('path', '')}",
            extra={"type": "request_start"}
        )

        # Track response status
        response_status = None

        async def send_wrapper(message):
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message.get("status", 0)
                # Add correlation ID to response headers
                headers = list(message.get("headers", []))
                headers.append((b"x-correlation-id", correlation_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            self.logger.exception(
                f"Request failed: {str(e)}",
                extra={"type": "request_error"}
            )
            raise
        finally:
            # Log response
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                f"Response {response_status} in {duration_ms:.2f}ms",
                extra={
                    "type": "request_complete",
                    "status_code": response_status,
                    "duration_ms": duration_ms
                }
            )
            # Clear context
            clear_context()

    def _get_client_ip(self, scope) -> str:
        """Extract client IP from scope."""
        headers = dict(scope.get("headers", []))

        # Check forwarded headers
        for header in [b"x-forwarded-for", b"x-real-ip"]:
            if header in headers:
                return headers[header].decode().split(",")[0].strip()

        # Fall back to client info
        client = scope.get("client")
        if client:
            return client[0]
        return "unknown"


# =========================
# Audit Logging
# =========================

class AuditLogger:
    """
    Specialized logger for security audit events.

    Records authentication, authorization, and data access events.
    """

    def __init__(self):
        self.logger = get_logger("quantum_traffic.audit")

    def log_auth_success(
        self,
        user_id: str,
        auth_method: str,
        ip_address: Optional[str] = None
    ) -> None:
        """Log successful authentication."""
        self.logger.info(
            f"Authentication successful for {user_id}",
            extra={
                "type": "auth_success",
                "user_id": user_id,
                "auth_method": auth_method,
                "ip_address": ip_address
            }
        )

    def log_auth_failure(
        self,
        reason: str,
        ip_address: Optional[str] = None,
        attempted_user: Optional[str] = None
    ) -> None:
        """Log failed authentication."""
        self.logger.warning(
            f"Authentication failed: {reason}",
            extra={
                "type": "auth_failure",
                "reason": reason,
                "ip_address": ip_address,
                "attempted_user": attempted_user
            }
        )

    def log_rate_limit(
        self,
        ip_address: str,
        limit: int,
        window: str
    ) -> None:
        """Log rate limit exceeded."""
        self.logger.warning(
            f"Rate limit exceeded for {ip_address}",
            extra={
                "type": "rate_limit",
                "ip_address": ip_address,
                "limit": limit,
                "window": window
            }
        )

    def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str
    ) -> None:
        """Log data access event."""
        self.logger.info(
            f"Data access: {action} {resource_type}/{resource_id}",
            extra={
                "type": "data_access",
                "user_id": user_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "action": action
            }
        )


# Global audit logger instance
audit_logger = AuditLogger()
