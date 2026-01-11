"""
Custom Exceptions and Error Handling for Quantum Traffic Optimizer.

This module provides:
- Custom exception classes
- Error response models
- Exception handlers for FastAPI
- Error categorization
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .logging_config import get_correlation_id, get_logger

logger = get_logger(__name__)


# =========================
# Error Response Models
# =========================

class ErrorDetail(BaseModel):
    """Detailed error information."""
    field: Optional[str] = Field(default=None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(default=None, description="Error code")


class ErrorResponse(BaseModel):
    """
    Standard error response format.

    All API errors follow this structure for consistency.
    """
    error: str = Field(..., description="Error type/category")
    message: str = Field(..., description="Human-readable error message")
    status_code: int = Field(..., description="HTTP status code")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation ID")
    details: Optional[list[ErrorDetail]] = Field(default=None, description="Detailed error info")
    timestamp: Optional[str] = Field(default=None, description="Error timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "status_code": 400,
                "correlation_id": "abc123",
                "details": [
                    {
                        "field": "deliveries",
                        "message": "At least one delivery is required",
                        "code": "min_length"
                    }
                ]
            }
        }


# =========================
# Custom Exceptions
# =========================

class QuantumTrafficException(Exception):
    """
    Base exception for Quantum Traffic Optimizer.

    All custom exceptions should inherit from this class.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[list[ErrorDetail]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)

    def to_response(self) -> ErrorResponse:
        """Convert exception to ErrorResponse."""
        from datetime import datetime, timezone
        return ErrorResponse(
            error=self.error_code,
            message=self.message,
            status_code=self.status_code,
            correlation_id=get_correlation_id(),
            details=self.details,
            timestamp=datetime.now(timezone.utc).isoformat()
        )


class ValidationException(QuantumTrafficException):
    """Raised when request validation fails."""

    def __init__(self, message: str, details: Optional[list[ErrorDetail]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )


class InvalidLocationException(QuantumTrafficException):
    """Raised when a location is outside the valid bounding box."""

    def __init__(self, lat: float, lng: float, bbox: tuple):
        super().__init__(
            message=f"Location ({lat}, {lng}) is outside the valid area",
            error_code="INVALID_LOCATION",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=[
                ErrorDetail(
                    field="location",
                    message=f"Latitude must be between {bbox[0]} and {bbox[1]}, "
                            f"Longitude must be between {bbox[2]} and {bbox[3]}",
                    code="out_of_bounds"
                )
            ]
        )


class OptimizationException(QuantumTrafficException):
    """Raised when route optimization fails."""

    def __init__(self, message: str, details: Optional[list[ErrorDetail]] = None):
        super().__init__(
            message=message,
            error_code="OPTIMIZATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


class OptimizationTimeoutException(QuantumTrafficException):
    """Raised when optimization exceeds the time limit."""

    def __init__(self, timeout: float):
        super().__init__(
            message=f"Optimization timed out after {timeout} seconds",
            error_code="OPTIMIZATION_TIMEOUT",
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            details=[
                ErrorDetail(
                    message="Try reducing the number of deliveries or using a simpler algorithm",
                    code="timeout"
                )
            ]
        )


class RouteNotFoundException(QuantumTrafficException):
    """Raised when a route is not found."""

    def __init__(self, route_id: str):
        super().__init__(
            message=f"Route '{route_id}' not found",
            error_code="ROUTE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND
        )


class GraphNotLoadedException(QuantumTrafficException):
    """Raised when the OSM graph is not available."""

    def __init__(self):
        super().__init__(
            message="Road network graph is not loaded",
            error_code="GRAPH_NOT_LOADED",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=[
                ErrorDetail(
                    message="The service is initializing. Please try again shortly.",
                    code="service_unavailable"
                )
            ]
        )


class AuthenticationException(QuantumTrafficException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=status.HTTP_401_UNAUTHORIZED
        )


class AuthorizationException(QuantumTrafficException):
    """Raised when authorization fails."""

    def __init__(self, message: str = "Access denied"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=status.HTTP_403_FORBIDDEN
        )


class RateLimitException(QuantumTrafficException):
    """Raised when rate limit is exceeded."""

    def __init__(self, retry_after: int = 60):
        super().__init__(
            message="Rate limit exceeded. Please slow down.",
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=[
                ErrorDetail(
                    message=f"Retry after {retry_after} seconds",
                    code="rate_limited"
                )
            ]
        )
        self.retry_after = retry_after


class ServiceUnavailableException(QuantumTrafficException):
    """Raised when a required service is unavailable."""

    def __init__(self, service: str):
        super().__init__(
            message=f"Service '{service}' is currently unavailable",
            error_code="SERVICE_UNAVAILABLE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


class DatabaseException(QuantumTrafficException):
    """Raised when a database operation fails."""

    def __init__(self, message: str = "Database operation failed"):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class CacheException(QuantumTrafficException):
    """Raised when a cache operation fails."""

    def __init__(self, message: str = "Cache operation failed"):
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# =========================
# Exception Handlers
# =========================

def create_error_response(
    error_code: str,
    message: str,
    status_code: int,
    details: Optional[list[ErrorDetail]] = None
) -> JSONResponse:
    """Create a JSON error response."""
    from datetime import datetime, timezone

    response = ErrorResponse(
        error=error_code,
        message=message,
        status_code=status_code,
        correlation_id=get_correlation_id(),
        details=details,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

    return JSONResponse(
        status_code=status_code,
        content=response.model_dump()
    )


async def quantum_traffic_exception_handler(
    request: Request,
    exc: QuantumTrafficException
) -> JSONResponse:
    """Handle QuantumTrafficException and subclasses."""
    logger.warning(
        f"Application error: {exc.error_code} - {exc.message}",
        extra={
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )

    response = exc.to_response()

    # Add retry-after header for rate limits
    headers = {}
    if isinstance(exc, RateLimitException):
        headers["Retry-After"] = str(exc.retry_after)

    return JSONResponse(
        status_code=exc.status_code,
        content=response.model_dump(),
        headers=headers if headers else None
    )


async def http_exception_handler(
    request: Request,
    exc: HTTPException
) -> JSONResponse:
    """Handle standard FastAPI HTTPException."""
    logger.warning(
        f"HTTP error: {exc.status_code} - {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )

    # Map status codes to error codes
    error_code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        422: "UNPROCESSABLE_ENTITY",
        429: "TOO_MANY_REQUESTS",
        500: "INTERNAL_SERVER_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
        504: "GATEWAY_TIMEOUT",
    }

    error_code = error_code_map.get(exc.status_code, f"HTTP_{exc.status_code}")

    return create_error_response(
        error_code=error_code,
        message=str(exc.detail),
        status_code=exc.status_code
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    logger.warning(
        f"Validation error on {request.url.path}",
        extra={
            "errors": exc.errors(),
            "path": request.url.path
        }
    )

    # Convert Pydantic errors to our format
    details = []
    for error in exc.errors():
        field_path = ".".join(str(loc) for loc in error.get("loc", []))
        details.append(ErrorDetail(
            field=field_path,
            message=error.get("msg", "Validation failed"),
            code=error.get("type", "validation_error")
        ))

    return create_error_response(
        error_code="VALIDATION_ERROR",
        message="Request validation failed",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details=details
    )


async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception(
        f"Unexpected error: {str(exc)}",
        extra={
            "exception_type": type(exc).__name__,
            "path": request.url.path
        }
    )

    # Don't expose internal error details in production
    from .config import get_settings
    settings = get_settings()

    if settings.is_production:
        message = "An unexpected error occurred"
    else:
        message = str(exc)

    return create_error_response(
        error_code="INTERNAL_ERROR",
        message=message,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )


def register_exception_handlers(app) -> None:
    """
    Register all exception handlers with the FastAPI app.

    Args:
        app: FastAPI application instance.
    """
    app.add_exception_handler(QuantumTrafficException, quantum_traffic_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
