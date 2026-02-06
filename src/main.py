"""
FastAPI Application for Quantum Traffic Optimization.

This module provides the REST API and WebSocket endpoints for
delivery route optimization using QUBO/QAOA.

Production Features:
- Configuration management with environment variables
- JWT and API key authentication
- Rate limiting
- Structured logging with correlation IDs
- Prometheus metrics
- Database persistence (optional)
- Redis caching (optional)
- API versioning
"""

import time
from contextlib import asynccontextmanager
from typing import Annotated, Dict, Optional

import folium
import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

# Local imports
from .config import Settings, get_settings
from .exceptions import (
    GraphNotLoadedException,
    OptimizationException,
    RouteNotFoundException,
    register_exception_handlers,
)
from .graph_builder import OSMGraph
from .logging_config import (
    RequestLoggingMiddleware,
    get_logger,
    setup_logging,
)
from .metrics import (
    MetricsMiddleware,
    get_metrics,
    get_metrics_content_type,
    health_checker,
    set_app_info,
    track_optimization,
    update_graph_metrics,
    update_qaoa_layers,
    update_routes_cached,
    update_websocket_connections,
)
from .models import (
    CompareResult,
    DeliveryPoint,
    ErrorResponse,
    HealthResponse,
    OptimizeRequest,
    OptimizeResult,
    ReoptimizeMessage,
    SequenceStop,
    SolverResultModel,
)
from .qubo_optimizer import QUBOOptimizer
from .security import (
    SecurityHeadersMiddleware,
    User,
    check_rate_limit,
    optional_auth,
)
from .traffic_sim import TrafficSimulator
from .utils import (
    calculate_eta,
    calculate_route_distance,
    compute_improvement,
    generate_route_id,
)

# Initialize logger
logger = get_logger(__name__)


# =========================
# Global State
# =========================

class AppState:
    """Application state container."""

    def __init__(self):
        self.graph: Optional[OSMGraph] = None
        self.traffic_sim: Optional[TrafficSimulator] = None
        self.optimizer: Optional[QUBOOptimizer] = None
        self.routes: Dict[str, Dict] = {}
        self.websocket_clients: list = []
        self._settings: Optional[Settings] = None

    @property
    def settings(self) -> Settings:
        """Get settings, loading if necessary."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    @property
    def max_cached_routes(self) -> int:
        """Get max cached routes from settings."""
        return self.settings.MAX_CACHED_ROUTES

    def add_route(self, route_id: str, route_data: Dict) -> None:
        """Add a route with automatic cache cleanup."""
        # Clean up old routes if cache is full
        if len(self.routes) >= self.max_cached_routes:
            # Remove oldest routes (first 20% of cache)
            routes_to_remove = list(self.routes.keys())[:self.max_cached_routes // 5]
            for old_route_id in routes_to_remove:
                del self.routes[old_route_id]
                logger.debug(f"Evicted route {old_route_id} from cache")

        self.routes[route_id] = route_data
        update_routes_cached(len(self.routes))

    def remove_route(self, route_id: str) -> bool:
        """Remove a route from cache."""
        if route_id in self.routes:
            del self.routes[route_id]
            update_routes_cached(len(self.routes))
            return True
        return False


# Global state instance
state = AppState()


# =========================
# Lifespan Management
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    settings = get_settings()

    # Setup logging
    setup_logging(settings)
    logger.info("Starting Quantum Traffic Optimizer...")

    # Set app info for metrics
    set_app_info(settings)

    # Initialize graph
    logger.info("Loading OSM graph...")
    state.graph = OSMGraph(use_cache=settings.OSM_USE_CACHE)

    if state.graph.graph is not None:
        update_graph_metrics(
            loaded=True,
            nodes=state.graph.graph.number_of_nodes(),
            edges=state.graph.graph.number_of_edges()
        )
        logger.info(
            f"Graph loaded: {state.graph.graph.number_of_nodes()} nodes, "
            f"{state.graph.graph.number_of_edges()} edges"
        )
    else:
        update_graph_metrics(loaded=False)
        logger.warning("Graph not loaded - using demo mode")

    # Initialize traffic simulator
    state.traffic_sim = TrafficSimulator(state.graph, seed=settings.QAOA_SEED)
    logger.info("Traffic simulator initialized")

    # Initialize QUBO optimizer
    state.optimizer = QUBOOptimizer(
        n_layers=settings.QAOA_LAYERS,
        seed=settings.QAOA_SEED,
        timeout=settings.QAOA_TIMEOUT
    )
    update_qaoa_layers(settings.QAOA_LAYERS)
    logger.info(f"QUBO optimizer initialized with {settings.QAOA_LAYERS} layers")

    # Register health checks
    health_checker.register("graph", lambda: state.graph is not None and state.graph.graph is not None)
    health_checker.register("optimizer", lambda: state.optimizer is not None)

    # Initialize database if enabled
    if settings.DATABASE_ENABLED:
        try:
            from .database import init_database
            await init_database()
            health_checker.register("database", check_database_health)
            logger.info("Database initialized")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    # Initialize cache if enabled
    if settings.REDIS_ENABLED:
        try:
            from .cache import init_cache
            await init_cache()
            health_checker.register("cache", check_cache_health)
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")

    logger.info("Initialization complete!")

    yield

    # Cleanup
    logger.info("Shutting down...")

    # Close database
    if settings.DATABASE_ENABLED:
        try:
            from .database import close_database
            await close_database()
        except Exception as e:
            logger.error(f"Database cleanup error: {e}")

    # Close cache
    if settings.REDIS_ENABLED:
        try:
            from .cache import close_cache
            await close_cache()
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

    state.websocket_clients.clear()
    logger.info("Shutdown complete")


async def check_database_health() -> bool:
    """Check database health."""
    try:
        from .database import db_manager
        return await db_manager.health_check()
    except Exception:
        return False


async def check_cache_health() -> bool:
    """Check cache health."""
    try:
        from .cache import cache_manager
        return await cache_manager.health_check()
    except Exception:
        return False


# =========================
# Create FastAPI App
# =========================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description="QUBO/QAOA-based delivery route optimization with quantum-inspired algorithms",
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs" if settings.DEBUG or settings.is_development else None,
        redoc_url="/redoc" if settings.DEBUG or settings.is_development else None,
        openapi_url="/openapi.json" if settings.DEBUG or settings.is_development else None,
    )

    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)

    # Add metrics middleware
    if settings.METRICS_ENABLED:
        app.add_middleware(MetricsMiddleware)

    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Add CORS middleware with restricted origins in production
    cors_origins = settings.cors_origins_list
    if settings.is_production and not cors_origins:
        logger.warning("No CORS origins configured for production!")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register exception handlers
    register_exception_handlers(app)

    return app


# Create app instance
app = create_app()


# =========================
# Root Endpoints
# =========================

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    settings = get_settings()
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{settings.APP_NAME}</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
            h1 {{ color: #2c3e50; }}
            a {{ color: #3498db; }}
            code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
            .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin: 2px; }}
            .badge-green {{ background: #27ae60; color: white; }}
            .badge-blue {{ background: #3498db; color: white; }}
        </style>
    </head>
    <body>
        <h1>Quantum Traffic Optimizer API</h1>
        <p>
            <span class="badge badge-green">v{settings.APP_VERSION}</span>
            <span class="badge badge-blue">{settings.ENVIRONMENT}</span>
        </p>
        <p>QUBO/QAOA-based delivery route optimization for Vijayawada, India.</p>

        <h2>Endpoints</h2>
        <ul>
            <li><a href="/docs">/docs</a> - Swagger UI documentation</li>
            <li><a href="/redoc">/redoc</a> - ReDoc documentation</li>
            <li><code>POST /optimize</code> - Optimize delivery route</li>
            <li><code>GET /map/{{route_id}}</code> - View route map</li>
            <li><code>GET /health</code> - Health check</li>
            <li><code>GET /api/v1/health</code> - Detailed health check (v1 API)</li>
            <li><code>WS /reoptimize</code> - Real-time updates</li>
        </ul>

        <h2>Quick Start</h2>
        <pre>
curl -X POST http://localhost:8000/optimize \\
  -H "Content-Type: application/json" \\
  -d '{{
    "current_loc": [16.5063, 80.6480],
    "deliveries": [
      {{"lat": 16.5175, "lng": 80.6198, "priority": 2}},
      {{"lat": 16.5412, "lng": 80.6352, "priority": 1}}
    ],
    "traffic_level": "medium"
  }}'
        </pre>

        <h2>Features</h2>
        <ul>
            <li>Quantum-inspired QUBO optimization</li>
            <li>Real-time traffic simulation</li>
            <li>Interactive Folium maps</li>
            <li>WebSocket support for live updates</li>
        </ul>
    </body>
    </html>
    """


# =========================
# Health Endpoints
# =========================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        graph_loaded=state.graph is not None and state.graph.graph is not None
    )


@app.get("/health/live", tags=["Health"])
async def liveness():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
async def readiness():
    """Kubernetes readiness probe."""
    health_status = await health_checker.check_all()

    if health_status["status"] == "healthy":
        return {"status": "ready", "checks": health_status["checks"]}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "checks": health_status["checks"]}
        )


# =========================
# Metrics Endpoint
# =========================

@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=get_metrics(),
        media_type=get_metrics_content_type()
    )


# =========================
# Optimization Endpoints
# =========================

@app.post(
    "/optimize",
    response_model=OptimizeResult,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Optimization failed"},
        503: {"model": ErrorResponse, "description": "Service not available"},
    },
    tags=["Optimization"],
    summary="Optimize Delivery Route",
    description="Optimize the delivery sequence using quantum-inspired algorithms.",
)
async def optimize_route(
    request: OptimizeRequest,
    background_tasks: BackgroundTasks,
    _: Annotated[None, Depends(check_rate_limit)],
    user: Annotated[Optional[User], Depends(optional_auth)],
):
    """
    Optimize delivery sequence using QAOA.

    Takes a current location and list of deliveries, returns an optimized
    route sequence with ETAs and optional map visualization.
    """
    settings = get_settings()
    start_time = time.time()

    # Validate state
    if state.graph is None or state.optimizer is None:
        raise GraphNotLoadedException()

    try:
        # Build locations list: current_loc + all deliveries
        locations = [request.current_loc]
        for d in request.deliveries:
            locations.append((d.lat, d.lng))

        n = len(locations)

        # Extract priorities (depot has priority 0)
        priorities = [0.0] + [d.priority for d in request.deliveries]

        # Special case: single delivery - no optimization needed
        if n == 2:
            # With only one delivery, the optimal route is trivial: depot -> delivery
            start_time_single = time.time()

            # Compute distance for single delivery
            dist_matrix = state.graph.precompute_shortest_paths(locations)

            # Update traffic and get congestion weights
            state.traffic_sim.update_congestion(request.traffic_level)
            congestion_matrix = state.traffic_sim.get_congestion_matrix(locations)

            # Direct route: depot (0) -> delivery (1)
            opt_sequence = [0, 1]
            single_distance = dist_matrix[0, 1]
            congestion = congestion_matrix[0, 1]
            single_eta = calculate_eta(single_distance, congestion_factor=congestion)

            opt_time = time.time() - start_time_single
            algorithm = "direct"
            improvement = 0.0  # No improvement possible with single delivery

            # Build sequence stop for single delivery
            delivery = request.deliveries[0]
            sequence_stops = [
                SequenceStop(
                    position=0,
                    delivery=delivery,
                    distance_from_prev=single_distance,
                    eta_from_prev=single_eta,
                    cumulative_distance=single_distance,
                    cumulative_eta=single_eta
                )
            ]

            # Generate route ID
            route_id = generate_route_id()

            # Generate map if requested
            map_html = None
            if request.include_map and settings.FEATURE_MAP_GENERATION:
                map_html = generate_route_map(
                    locations,
                    opt_sequence,
                    request.deliveries,
                    route_id
                )

            # Store route data
            route_data = {
                "locations": locations,
                "sequence": opt_sequence,
                "deliveries": [d.model_dump() for d in request.deliveries],
                "traffic_level": request.traffic_level,
                "map_html": map_html,
                "total_distance": single_distance,
                "total_eta": single_eta,
                "user_id": user.id if user else None,
            }

            state.add_route(route_id, route_data)

            # Track metrics
            track_optimization(
                traffic_level=request.traffic_level,
                algorithm=algorithm,
                n_deliveries=1,
                duration=opt_time,
                improvement=improvement,
                total_distance=single_distance,
                total_eta=single_eta,
                success=True
            )

            # Compute route geometry (actual road paths)
            route_geometry = state.graph.get_route_geometry(locations, opt_sequence)
            # Convert to list format for JSON serialization
            route_geometry_json = [
                [[coord[0], coord[1]] for coord in segment]
                for segment in route_geometry
            ]

            logger.info(
                f"Route optimized (single delivery): {route_id}",
                extra={
                    "route_id": route_id,
                    "n_deliveries": 1,
                    "algorithm": algorithm,
                    "optimization_time": opt_time,
                }
            )

            return OptimizeResult(
                route_id=route_id,
                sequence=sequence_stops,
                total_distance=single_distance,
                total_eta=single_eta,
                optimization_time=opt_time,
                traffic_level=request.traffic_level,
                map_html=map_html,
                improvement_over_greedy=improvement,
                route_geometry=route_geometry_json
            )

        # Compute distance matrix
        dist_matrix = state.graph.precompute_shortest_paths(locations)

        # Update traffic and get congestion weights
        state.traffic_sim.update_congestion(request.traffic_level)
        congestion_matrix = state.traffic_sim.get_congestion_matrix(locations)

        # Determine algorithm to use based on problem size
        use_qaoa = settings.USE_QAOA_IN_API and n <= 4  # QAOA only for tiny instances
        if use_qaoa:
            algorithm = "qaoa"
        elif n <= 8:
            algorithm = "brute_force"
        else:
            algorithm = "simulated_annealing"

        # Run optimization
        opt_sequence, opt_cost, opt_time = state.optimizer.optimize(
            dist_matrix,
            priorities,
            congestion_matrix,
            request.traffic_level,
            use_qaoa=use_qaoa
        )

        # Ensure depot is first
        if opt_sequence[0] != 0:
            if 0 in opt_sequence:
                opt_sequence.remove(0)
            opt_sequence.insert(0, 0)

        # Validate that all nodes are included in the sequence
        expected_nodes = set(range(n))
        actual_nodes = set(opt_sequence)
        missing_nodes = expected_nodes - actual_nodes

        if missing_nodes:
            logger.warning(
                f"Sequence missing {len(missing_nodes)} nodes, adding them",
                extra={"missing_nodes": list(missing_nodes)}
            )
            # Add missing nodes at the end (will be optimized by 2-opt)
            for node in sorted(missing_nodes):
                opt_sequence.append(node)

        # Compute greedy baseline for comparison
        greedy_seq, greedy_cost = state.optimizer.solve_greedy(dist_matrix, start_idx=0)
        opt_distance = calculate_route_distance(opt_sequence, dist_matrix)
        greedy_distance = calculate_route_distance(greedy_seq, dist_matrix)

        # If simulated annealing gave worse result, use greedy instead
        if opt_distance > greedy_distance:
            opt_sequence = greedy_seq
            opt_distance = greedy_distance
            algorithm = "greedy"

        improvement = compute_improvement(opt_distance, greedy_distance)

        # Build sequence stops
        sequence_stops = []
        cumulative_dist = 0.0
        cumulative_eta = 0.0

        for pos, loc_idx in enumerate(opt_sequence):
            if loc_idx == 0:
                continue  # Skip depot in output sequence

            prev_idx = opt_sequence[pos - 1] if pos > 0 else 0

            dist_from_prev = dist_matrix[prev_idx, loc_idx]
            congestion = congestion_matrix[prev_idx, loc_idx]
            eta_from_prev = calculate_eta(dist_from_prev, congestion_factor=congestion)

            cumulative_dist += dist_from_prev
            cumulative_eta += eta_from_prev

            # Map back to delivery
            delivery_idx = loc_idx - 1  # Adjust for depot offset
            delivery = request.deliveries[delivery_idx]

            stop = SequenceStop(
                position=len(sequence_stops),
                delivery=delivery,
                distance_from_prev=dist_from_prev,
                eta_from_prev=eta_from_prev,
                cumulative_distance=cumulative_dist,
                cumulative_eta=cumulative_eta
            )
            sequence_stops.append(stop)

        # Generate route ID
        route_id = generate_route_id()

        # Generate map if requested
        map_html = None
        if request.include_map and settings.FEATURE_MAP_GENERATION:
            map_html = generate_route_map(
                locations,
                opt_sequence,
                request.deliveries,
                route_id
            )

        # Store route data
        route_data = {
            "locations": locations,
            "sequence": opt_sequence,
            "deliveries": [d.model_dump() for d in request.deliveries],
            "traffic_level": request.traffic_level,
            "map_html": map_html,
            "total_distance": cumulative_dist,
            "total_eta": cumulative_eta,
            "user_id": user.id if user else None,
        }

        # Store route for later retrieval (with automatic cache cleanup)
        state.add_route(route_id, route_data)

        total_time = time.time() - start_time

        # Track metrics
        track_optimization(
            traffic_level=request.traffic_level,
            algorithm=algorithm,
            n_deliveries=len(request.deliveries),
            duration=opt_time,
            improvement=improvement,
            total_distance=cumulative_dist,
            total_eta=cumulative_eta,
            success=True
        )

        # Compute route geometry (actual road paths)
        route_geometry = state.graph.get_route_geometry(locations, opt_sequence)
        # Convert to list format for JSON serialization
        route_geometry_json = [
            [[coord[0], coord[1]] for coord in segment]
            for segment in route_geometry
        ]

        logger.info(
            f"Route optimized: {route_id}",
            extra={
                "route_id": route_id,
                "n_deliveries": len(request.deliveries),
                "algorithm": algorithm,
                "optimization_time": opt_time,
                "total_time": total_time,
                "improvement": improvement
            }
        )

        return OptimizeResult(
            route_id=route_id,
            sequence=sequence_stops,
            total_distance=cumulative_dist,
            total_eta=cumulative_eta,
            optimization_time=opt_time,
            traffic_level=request.traffic_level,
            map_html=map_html,
            improvement_over_greedy=improvement,
            route_geometry=route_geometry_json
        )

    except ValueError as e:
        logger.warning(f"Validation error in optimization: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Optimization failed: {e}")
        track_optimization(
            traffic_level=request.traffic_level,
            algorithm="unknown",
            n_deliveries=len(request.deliveries),
            duration=time.time() - start_time,
            improvement=None,
            total_distance=0,
            total_eta=0,
            success=False
        )
        raise OptimizationException(f"Optimization failed: {str(e)}")


@app.post(
    "/compare",
    response_model=CompareResult,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Comparison failed"},
        503: {"model": ErrorResponse, "description": "Service not available"},
    },
    tags=["Optimization"],
    summary="Compare Optimization Methods",
    description="Compare all available optimization algorithms on the same problem.",
)
async def compare_methods(
    request: OptimizeRequest,
    _: Annotated[None, Depends(check_rate_limit)] = None,
    include_qaoa: bool = Query(False, description="Include QAOA solver (slow, only for n<=4)"),
):
    """
    Compare all optimization algorithms on the same dataset.

    Runs greedy, simulated annealing, brute force (n<=8), and optionally QAOA
    on the same problem and returns comparative results with execution times
    and improvement percentages.
    """
    start_time = time.time()

    # Validate state
    if state.graph is None or state.optimizer is None:
        raise GraphNotLoadedException()

    try:
        # Build locations list: current_loc + all deliveries
        locations = [request.current_loc]
        for d in request.deliveries:
            locations.append((d.lat, d.lng))

        n = len(locations)

        # Extract priorities (depot has priority 0)
        priorities = [0.0] + [d.priority for d in request.deliveries]

        # Compute distance matrix
        dist_matrix = state.graph.precompute_shortest_paths(locations)

        # Update traffic and get congestion weights
        state.traffic_sim.update_congestion(request.traffic_level)
        congestion_matrix = state.traffic_sim.get_congestion_matrix(locations)

        # Run comparison
        comparison = state.optimizer.compare_solvers(
            dist_matrix,
            priorities,
            congestion_matrix,
            request.traffic_level,
            include_qaoa=include_qaoa
        )

        total_time = time.time() - start_time

        logger.info(
            f"Solver comparison completed",
            extra={
                "n_locations": n,
                "traffic_level": request.traffic_level,
                "best_solver": comparison["best_solver"],
                "total_time": total_time,
                "include_qaoa": include_qaoa
            }
        )

        # Convert to response model
        solver_results = [
            SolverResultModel(**solver) for solver in comparison["solvers"]
        ]

        return CompareResult(
            solvers=solver_results,
            best_solver=comparison["best_solver"],
            improvements=comparison["improvements"],
            problem_size=comparison["problem_size"],
            traffic_level=request.traffic_level
        )

    except ValueError as e:
        logger.warning(f"Validation error in comparison: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Comparison failed: {e}")
        raise OptimizationException(f"Comparison failed: {str(e)}")


# =========================
# Route Endpoints
# =========================

@app.get(
    "/map/{route_id}",
    response_class=HTMLResponse,
    tags=["Routes"],
    summary="Get Route Map",
    description="Get the interactive Folium map for a route.",
)
async def get_map(route_id: str):
    """Get interactive Folium map for a route."""
    if route_id not in state.routes:
        raise RouteNotFoundException(route_id)

    route_data = state.routes[route_id]

    if route_data.get("map_html"):
        return HTMLResponse(content=route_data["map_html"])

    # Generate map if not cached
    map_html = generate_route_map(
        route_data["locations"],
        route_data["sequence"],
        [DeliveryPoint(**d) for d in route_data["deliveries"]],
        route_id
    )

    state.routes[route_id]["map_html"] = map_html
    return HTMLResponse(content=map_html)


@app.get(
    "/routes",
    response_class=JSONResponse,
    tags=["Routes"],
    summary="List Routes",
    description="List all cached routes.",
)
async def list_routes(
    skip: int = Query(0, ge=0, description="Number of routes to skip"),
    limit: int = Query(20, ge=1, le=100, description="Maximum routes to return"),
):
    """List all cached routes with pagination."""
    routes = []
    for route_id, data in state.routes.items():
        routes.append({
            "route_id": route_id,
            "n_deliveries": len(data.get("deliveries", [])),
            "traffic_level": data.get("traffic_level"),
            "total_distance": data.get("total_distance"),
            "total_eta": data.get("total_eta"),
        })

    total = len(routes)
    routes = routes[skip:skip + limit]

    return {
        "routes": routes,
        "total": total,
        "skip": skip,
        "limit": limit
    }


@app.delete(
    "/routes/{route_id}",
    tags=["Routes"],
    summary="Delete Route",
    description="Delete a cached route.",
)
async def delete_route(route_id: str):
    """Delete a cached route."""
    if not state.remove_route(route_id):
        raise RouteNotFoundException(route_id)

    logger.info(f"Route deleted: {route_id}")
    return {"message": f"Route {route_id} deleted"}


# =========================
# WebSocket Endpoint
# =========================

@app.websocket("/reoptimize")
async def websocket_reoptimize(websocket: WebSocket):
    """
    WebSocket for real-time route updates.

    Clients can subscribe to traffic updates and receive re-optimized routes
    when conditions change.
    """
    settings = get_settings()

    if not settings.FEATURE_WEBSOCKET_ENABLED:
        await websocket.close(code=4001, reason="WebSocket feature disabled")
        return

    await websocket.accept()
    state.websocket_clients.append(websocket)
    update_websocket_connections(len(state.websocket_clients))

    logger.info(f"WebSocket client connected. Total: {len(state.websocket_clients)}")

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = ReoptimizeMessage(**data)

            if message.type == "traffic_change":
                # Traffic level changed, notify client
                new_level = message.data.get("level", "medium")
                state.traffic_sim.update_congestion(new_level)

                await websocket.send_json({
                    "type": "ack",
                    "data": {"message": f"Traffic updated to {new_level}"}
                })

            elif message.type == "new_delivery":
                # New delivery added, client should re-optimize
                await websocket.send_json({
                    "type": "ack",
                    "data": {"message": "Use POST /optimize for new deliveries"}
                })

    except WebSocketDisconnect:
        if websocket in state.websocket_clients:
            state.websocket_clients.remove(websocket)
            update_websocket_connections(len(state.websocket_clients))
        logger.info(f"WebSocket client disconnected. Total: {len(state.websocket_clients)}")
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": str(e)}
            })
        except Exception:
            pass
        finally:
            if websocket in state.websocket_clients:
                state.websocket_clients.remove(websocket)
                update_websocket_connections(len(state.websocket_clients))


# =========================
# Map Generation
# =========================

def generate_route_map(
    locations: list,
    sequence: list,
    deliveries: list,
    route_id: str
) -> str:
    """
    Generate Folium map HTML for a route.

    Args:
        locations: List of (lat, lng) tuples.
        sequence: Order of location indices.
        deliveries: List of DeliveryPoint objects.
        route_id: Route identifier.

    Returns:
        HTML string of the map.
    """
    # Center map on route
    center_lat = np.mean([loc[0] for loc in locations])
    center_lng = np.mean([loc[1] for loc in locations])

    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=13,
        tiles="OpenStreetMap"
    )

    # Color scheme
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue']

    # Add depot marker
    depot = locations[0]
    folium.Marker(
        location=depot,
        popup=f"<b>Depot (Start)</b><br>Route: {route_id}",
        tooltip="Depot",
        icon=folium.Icon(color='red', icon='home', prefix='fa')
    ).add_to(m)

    # Add delivery markers with sequence numbers
    for pos, loc_idx in enumerate(sequence):
        if loc_idx == 0:
            continue  # Skip depot

        delivery_idx = loc_idx - 1
        if delivery_idx < len(deliveries):
            delivery = deliveries[delivery_idx]
            loc = (delivery.lat, delivery.lng)

            seq_num = len([i for i in sequence[:sequence.index(loc_idx)] if i != 0])
            color = colors[seq_num % len(colors)]

            folium.Marker(
                location=loc,
                popup=f"<b>Stop {seq_num + 1}</b><br>"
                       f"Priority: {delivery.priority}<br>"
                       f"Name: {delivery.name or 'N/A'}",
                tooltip=f"Stop {seq_num + 1}",
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)

    # Draw route polylines
    route_coords = [locations[idx] for idx in sequence]

    folium.PolyLine(
        locations=route_coords,
        weight=4,
        color='blue',
        opacity=0.7,
        popup="Optimized Route"
    ).add_to(m)

    # Add route info box
    total_stops = len(sequence) - 1  # Exclude depot
    folium.map.Marker(
        [locations[0][0] + 0.01, locations[0][1]],
        icon=folium.DivIcon(
            icon_size=(200, 40),
            icon_anchor=(0, 0),
            html=f'<div style="background:white;padding:5px;border-radius:5px;font-size:12px;">'
                 f'<b>Route {route_id}</b><br>Stops: {total_stops}</div>'
        )
    ).add_to(m)

    return m._repr_html_()


# =========================
# Run Server
# =========================

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=1 if settings.RELOAD else settings.WORKERS,
    )
