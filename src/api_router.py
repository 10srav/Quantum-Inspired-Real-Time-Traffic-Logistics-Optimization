"""
API Router with Versioning for Quantum Traffic Optimizer.

This module provides:
- API versioning (v1)
- Route organization
- OpenAPI documentation enhancement
"""

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .models import ErrorResponse, HealthResponse
from .security import User, check_rate_limit, optional_auth, require_auth

# =========================
# API Router Configuration
# =========================

# Create versioned router
api_v1_router = APIRouter(prefix="/api/v1", tags=["v1"])


# =========================
# Health & Info Endpoints
# =========================

@api_v1_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the API and its dependencies.",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy", "model": ErrorResponse}
    }
)
async def health_check(
    settings: Annotated[Settings, Depends(get_settings)]
):
    """
    Health check endpoint for monitoring.

    Returns the service status, version, and component health.
    """
    from .metrics import health_checker

    # Run health checks
    health_status = await health_checker.check_all()

    status_code = 200 if health_status["status"] == "healthy" else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": health_status["status"],
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "checks": health_status["checks"]
        }
    )


@api_v1_router.get(
    "/health/live",
    summary="Liveness Probe",
    description="Kubernetes liveness probe endpoint.",
)
async def liveness():
    """
    Liveness probe for Kubernetes.

    Returns 200 if the service is running.
    """
    return {"status": "alive"}


@api_v1_router.get(
    "/health/ready",
    summary="Readiness Probe",
    description="Kubernetes readiness probe endpoint.",
)
async def readiness(
    settings: Annotated[Settings, Depends(get_settings)]
):
    """
    Readiness probe for Kubernetes.

    Returns 200 if the service is ready to accept traffic.
    Checks database, cache, and graph availability.
    """
    from .metrics import health_checker

    health_status = await health_checker.check_all()

    if health_status["status"] == "healthy":
        return {"status": "ready", "checks": health_status["checks"]}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "checks": health_status["checks"]}
        )


@api_v1_router.get(
    "/info",
    summary="API Information",
    description="Get information about the API.",
)
async def api_info(
    settings: Annotated[Settings, Depends(get_settings)]
):
    """
    Get API information.

    Returns version, environment, and feature flags.
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "features": {
            "websocket": settings.FEATURE_WEBSOCKET_ENABLED,
            "map_generation": settings.FEATURE_MAP_GENERATION,
            "qaoa": settings.USE_QAOA_IN_API,
        },
        "limits": {
            "max_deliveries": settings.MAX_DELIVERIES,
            "rate_limit_per_minute": settings.RATE_LIMIT_PER_MINUTE,
        }
    }


# =========================
# Routes List Endpoints
# =========================

@api_v1_router.get(
    "/routes",
    summary="List Routes",
    description="List all cached routes with pagination.",
    dependencies=[Depends(check_rate_limit)]
)
async def list_routes(
    user: Annotated[Optional[User], Depends(optional_auth)],
    skip: int = Query(0, ge=0, description="Number of routes to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum routes to return"),
    sort_by: str = Query("created_at", pattern="^(created_at|total_distance|total_eta)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    traffic_level: Optional[str] = Query(None, pattern="^(low|medium|high)$"),
):
    """
    List cached routes with pagination and filtering.

    Args:
        skip: Number of routes to skip (for pagination).
        limit: Maximum number of routes to return.
        sort_by: Field to sort by.
        order: Sort order (asc/desc).
        traffic_level: Filter by traffic level.

    Returns:
        Paginated list of routes.
    """
    # Import here to avoid circular dependency
    from .main import state

    routes = []
    for route_id, data in state.routes.items():
        # Filter by traffic level
        if traffic_level and data.get("traffic_level") != traffic_level:
            continue

        routes.append({
            "route_id": route_id,
            "n_deliveries": len(data.get("deliveries", [])),
            "traffic_level": data.get("traffic_level"),
            "total_distance": data.get("total_distance"),
            "total_eta": data.get("total_eta"),
        })

    # Sort routes
    reverse = order == "desc"
    if sort_by in ["total_distance", "total_eta"]:
        routes.sort(key=lambda x: x.get(sort_by, 0) or 0, reverse=reverse)

    # Apply pagination
    total = len(routes)
    routes = routes[skip:skip + limit]

    return {
        "routes": routes,
        "pagination": {
            "skip": skip,
            "limit": limit,
            "total": total,
            "has_more": skip + limit < total
        }
    }


# =========================
# Protected Endpoints Example
# =========================

@api_v1_router.get(
    "/protected",
    summary="Protected Endpoint",
    description="Example endpoint requiring authentication.",
    dependencies=[Depends(check_rate_limit)]
)
async def protected_endpoint(
    user: Annotated[User, Depends(require_auth)]
):
    """
    Example protected endpoint.

    Requires valid authentication (JWT or API key).
    """
    return {
        "message": "You have access!",
        "user_id": user.id,
        "scopes": user.scopes
    }


# =========================
# Metrics Endpoint
# =========================

@api_v1_router.get(
    "/metrics",
    summary="Prometheus Metrics",
    description="Get Prometheus metrics for monitoring.",
    include_in_schema=False  # Hide from public docs
)
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    from fastapi.responses import Response
    from .metrics import get_metrics, get_metrics_content_type

    return Response(
        content=get_metrics(),
        media_type=get_metrics_content_type()
    )


# =========================
# Router Factory
# =========================

def create_api_router(settings: Settings) -> APIRouter:
    """
    Create and configure the API router.

    Args:
        settings: Application settings.

    Returns:
        Configured APIRouter instance.
    """
    router = APIRouter()

    # Include v1 router
    router.include_router(api_v1_router)

    return router
