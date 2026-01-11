# System Architecture

## Overview

The Quantum Traffic Optimizer is a FastAPI-based service that provides delivery route optimization using quantum-inspired algorithms (QUBO/QAOA).

## Architecture Diagram

```
                                    ┌─────────────────┐
                                    │   Prometheus    │
                                    │   /metrics      │
                                    └────────┬────────┘
                                             │
┌─────────────┐    ┌─────────────────────────┼─────────────────────────────┐
│   Clients   │    │                         │                             │
│  (Web/Mobile)│───▶│  ┌─────────────────────▼─────────────────────┐       │
└─────────────┘    │  │           FastAPI Application              │       │
                   │  │  ┌───────────────────────────────────────┐ │       │
┌─────────────┐    │  │  │            Middleware Stack           │ │       │
│  Streamlit  │───▶│  │  │  ┌───────────────────────────────────┐│ │       │
│  Frontend   │    │  │  │  │  Security Headers                 ││ │       │
└─────────────┘    │  │  │  │  Rate Limiting                    ││ │       │
                   │  │  │  │  Request Logging                  ││ │       │
                   │  │  │  │  Metrics Collection               ││ │       │
                   │  │  │  │  CORS                             ││ │       │
                   │  │  │  └───────────────────────────────────┘│ │       │
                   │  │  └───────────────────────────────────────┘ │       │
                   │  │                                            │       │
                   │  │  ┌───────────────────────────────────────┐ │       │
                   │  │  │           API Endpoints               │ │       │
                   │  │  │  POST /optimize                       │ │       │
                   │  │  │  GET  /routes                         │ │       │
                   │  │  │  GET  /map/{route_id}                 │ │       │
                   │  │  │  GET  /health                         │ │       │
                   │  │  │  WS   /reoptimize                     │ │       │
                   │  │  └───────────────────────────────────────┘ │       │
                   │  └────────────────────────────────────────────┘       │
                   │                        │                              │
                   │           ┌────────────┼────────────┐                 │
                   │           ▼            ▼            ▼                 │
                   │  ┌─────────────┐ ┌──────────┐ ┌──────────────┐        │
                   │  │   OSMGraph  │ │  Traffic │ │    QUBO      │        │
                   │  │   Builder   │ │Simulator │ │  Optimizer   │        │
                   │  └──────┬──────┘ └────┬─────┘ └──────┬───────┘        │
                   │         │             │              │                │
                   │         ▼             ▼              ▼                │
                   │  ┌─────────────────────────────────────────────┐      │
                   │  │              Application State               │      │
                   │  │  - Graph cache                               │      │
                   │  │  - Route cache                               │      │
                   │  │  - WebSocket clients                         │      │
                   │  └─────────────────────────────────────────────┘      │
                   │                        │                              │
                   │         ┌──────────────┼──────────────┐               │
                   │         ▼              ▼              ▼               │
                   │  ┌───────────┐  ┌───────────┐  ┌───────────┐          │
                   │  │PostgreSQL │  │   Redis   │  │   File    │          │
                   │  │(Optional) │  │(Optional) │  │  Cache    │          │
                   │  └───────────┘  └───────────┘  └───────────┘          │
                   └───────────────────────────────────────────────────────┘
```

## Component Details

### 1. FastAPI Application (`src/main.py`)

The main application entry point that:
- Initializes all components on startup
- Configures middleware stack
- Registers API endpoints
- Manages application lifecycle

### 2. Configuration (`src/config.py`)

Centralized configuration using Pydantic Settings:
- Environment variable support
- Type validation
- Default values
- Computed properties

### 3. Security (`src/security.py`)

Authentication and authorization:
- JWT token authentication
- API key authentication
- Rate limiting (in-memory)
- Security headers middleware

### 4. Database (`src/database.py`, `src/database_models.py`)

PostgreSQL persistence layer:
- Async SQLAlchemy ORM
- Connection pooling
- Route storage
- Optimization metrics
- Audit logging

### 5. Cache (`src/cache.py`)

Redis caching layer:
- Route data caching
- Graph caching
- Rate limiting support
- Distributed locks
- In-memory fallback

### 6. Logging (`src/logging_config.py`)

Structured logging:
- JSON formatting for production
- Correlation ID tracking
- Request context propagation
- Audit logging

### 7. Metrics (`src/metrics.py`)

Prometheus monitoring:
- HTTP request metrics
- Optimization metrics
- System health metrics
- Custom business metrics

### 8. OSM Graph Builder (`src/graph_builder.py`)

Road network management:
- OpenStreetMap data loading
- Graph caching (GraphML)
- Shortest path computation
- Demo mode for testing

### 9. Traffic Simulator (`src/traffic_sim.py`)

Congestion modeling:
- Dynamic traffic levels
- Edge-based congestion
- Time-based multipliers

### 10. QUBO Optimizer (`src/qubo_optimizer.py`)

Route optimization:
- QUBO problem encoding
- QAOA solver (quantum-inspired)
- Simulated annealing fallback
- Greedy baseline

## Data Flow

### Optimization Request Flow

```
1. Client Request
   │
   ▼
2. Rate Limit Check
   │
   ▼
3. Request Validation (Pydantic)
   │
   ▼
4. Authentication (Optional)
   │
   ▼
5. Build Distance Matrix
   │  └── OSMGraph.precompute_shortest_paths()
   │
   ▼
6. Apply Traffic Simulation
   │  └── TrafficSimulator.get_congestion_matrix()
   │
   ▼
7. Run Optimization
   │  └── QUBOOptimizer.optimize()
   │      ├── QAOA (if enabled)
   │      └── Brute Force / SA
   │
   ▼
8. Generate Map (if requested)
   │  └── Folium visualization
   │
   ▼
9. Cache Route
   │  ├── In-memory cache
   │  ├── Redis (if enabled)
   │  └── PostgreSQL (if enabled)
   │
   ▼
10. Return Response
    │  └── OptimizeResult
    │
    ▼
11. Track Metrics
    └── Prometheus counters/histograms
```

## Scaling Considerations

### Horizontal Scaling

- Stateless API design (with external state storage)
- Redis for distributed caching
- PostgreSQL for persistence
- Load balancer support (X-Forwarded-For)

### Vertical Scaling

- Worker process configuration
- Connection pool sizing
- Memory management for graph data

### Bottlenecks

1. **Graph Loading**: Large OSM graphs require significant memory
2. **QAOA Optimization**: CPU-intensive, limited to small instances
3. **Map Generation**: Memory-intensive HTML generation

## Security Model

### Authentication

```
┌─────────────────────────────────────────────────────────────┐
│                    Request                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Authorization: Bearer <JWT>                              ││
│  │ X-API-Key: <API_KEY>                                     ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────┴────────────────────┐
         │                                         │
         ▼                                         ▼
┌─────────────────┐                     ┌─────────────────┐
│  JWT Validation │                     │ API Key Check   │
│  - Decode token │                     │ - Hash compare  │
│  - Check expiry │                     │ - Constant time │
│  - Verify sig   │                     │                 │
└────────┬────────┘                     └────────┬────────┘
         │                                       │
         └───────────────┬───────────────────────┘
                         ▼
                ┌─────────────────┐
                │  User Context   │
                │  - ID           │
                │  - Scopes       │
                └─────────────────┘
```

### Rate Limiting

- Per-IP rate limiting
- Configurable limits (per-minute, per-hour)
- In-memory tracking (Redis in production)
- Retry-After headers

### Security Headers

- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Content-Security-Policy (for HTML responses)
- Referrer-Policy: strict-origin-when-cross-origin

## Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Counters   │  │ Histograms  │  │   Gauges    │          │
│  │ - requests  │  │ - latency   │  │ - routes    │          │
│  │ - errors    │  │ - opt time  │  │ - websocket │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│                  ┌─────────────┐                            │
│                  │  /metrics   │                            │
│                  └──────┬──────┘                            │
└─────────────────────────┼───────────────────────────────────┘
                          │
                          ▼
                  ┌─────────────┐
                  │ Prometheus  │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │  Grafana    │
                  └─────────────┘
```

## File Structure

```
src/
├── __init__.py
├── main.py              # FastAPI application
├── config.py            # Configuration management
├── models.py            # Pydantic data models
├── security.py          # Authentication & authorization
├── database.py          # Database connection management
├── database_models.py   # SQLAlchemy ORM models
├── cache.py             # Redis caching layer
├── logging_config.py    # Structured logging
├── metrics.py           # Prometheus metrics
├── exceptions.py        # Custom exceptions
├── api_router.py        # API versioning
├── graph_builder.py     # OSM graph management
├── qubo_optimizer.py    # QAOA optimization
├── traffic_sim.py       # Traffic simulation
└── utils.py             # Utility functions

tests/
├── test_api.py
├── test_graph_builder.py
├── test_qubo_optimizer.py
├── test_traffic_sim.py
└── test_full_system.py

docs/
├── ARCHITECTURE.md      # This file
├── PRODUCTION.md        # Deployment guide
└── API.md               # API documentation

frontend/
└── streamlit_app.py     # Streamlit UI
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Web Framework | FastAPI |
| Server | Uvicorn |
| Database | PostgreSQL + SQLAlchemy |
| Cache | Redis |
| Authentication | JWT (python-jose) |
| Metrics | Prometheus |
| Logging | Python logging (JSON) |
| Graph | OSMnx + NetworkX |
| Optimization | Qiskit QAOA |
| Visualization | Folium |
| Frontend | Streamlit |
