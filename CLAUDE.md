# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantum-inspired logistics optimization application that solves TSP/VRP using QUBO/QAOA via Qiskit. Python/FastAPI backend + React/TypeScript frontend with real-time traffic simulation targeting Vijayawada, India road network.

## Common Commands

### Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn src.main:app --reload

# Run production server (4 workers)
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend
```bash
cd quantum-traffic-ui
npm install
npm run dev          # Development (http://localhost:5173)
npm run build        # Production build
```

### Testing
```bash
# Run all tests (uses demo mode, disables DB/Redis)
OSM_DEMO_MODE=true DATABASE_ENABLED=false REDIS_ENABLED=false pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Fast test run (stop on first failure)
pytest tests/ -v -x
```

### Linting & Formatting
```bash
flake8 src/ tests/ --max-line-length=120
black src/ tests/ --line-length=120
isort src/ tests/
mypy src/ --ignore-missing-imports
```

### Docker
```bash
docker compose up -d                           # Start backend + DB + Redis
docker compose --profile frontend up -d        # Include React frontend
docker compose down -v --remove-orphans        # Stop and clean
```

## Architecture

```
src/                          quantum-traffic-ui/src/
├── main.py      (FastAPI)    ├── stores/
│   ↓                         │   ├── authStore.ts  (JWT)
├── qubo_optimizer.py         │   └── routeStore.ts (state)
│   (QAOA/QUBO solver)        ├── services/
├── traffic_sim.py            │   ├── api.ts       (Axios)
│   (congestion model)        │   └── websocket.ts
├── graph_builder.py          ├── pages/
│   (OSMnx road network)      │   └── Dashboard.tsx
├── clustering.py             └── components/Dashboard/
│   (K-means for 200+ nodes)      ├── AlgorithmComparison.tsx
├── traffic_api.py                └── ComparisonChart.tsx
│   (TomTom/HERE integration)
└── models.py (Pydantic)
```

### Core Modules

- **qubo_optimizer.py**: QUBO encoding of TSP, QAOA solver via Qiskit (configurable layers), greedy fallback. Methods: `encode_qubo()`, `solve_qaoa()`, `solve_greedy()`
- **traffic_sim.py**: Dynamic congestion with Poisson distribution. Traffic levels: low (1.0x), medium (1.5x), high (2.5x). Methods: `update_congestion()`, `get_dynamic_weights()`
- **graph_builder.py**: OSMnx graph manager for Vijayawada bounding box (16.50-16.55, 80.62-80.68). Falls back to synthetic demo graph. Precomputes shortest paths with Dijkstra
- **clustering.py**: Hierarchical K-means clustering for 200+ delivery points. Auto-computes optimal K via silhouette score. Class: `HierarchicalOptimizer`
- **traffic_api.py**: Real-time traffic API integration (TomTom/HERE) with 5-min cache TTL. Falls back to simulation when unavailable. Class: `TrafficAPIService`
- **main.py**: FastAPI app with middleware stack (CORS, rate limiting, metrics). Key endpoints: `POST /optimize`, `POST /compare-algorithms`, `WS /reoptimize`, `GET /health`

### Frontend State (Zustand)

- **authStore**: JWT tokens in localStorage, login/logout, token refresh
- **routeStore**: Current location, deliveries list, optimization results, traffic level

## Key Configuration

Environment variables loaded via `src/config.py` (Pydantic BaseSettings):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OSM_DEMO_MODE` | false | Use synthetic graph instead of downloading OSM |
| `DATABASE_ENABLED` | true | Enable PostgreSQL |
| `REDIS_ENABLED` | true | Enable Redis caching |
| `QAOA_LAYERS` | 3 | QAOA circuit depth (p parameter) |
| `QAOA_TIMEOUT` | 5 | Optimization timeout in seconds |
| `USE_QAOA_IN_API` | true | Use QAOA vs greedy in API |

Frontend uses Vite env vars: `VITE_API_URL`, `VITE_WS_URL`

## API Request Format

```json
POST /optimize
{
  "current_loc": [16.52, 80.63],
  "deliveries": [
    {"lat": 16.54, "lng": 80.65, "priority": 2.0}
  ],
  "traffic_level": "medium",
  "include_map": true
}
```

## Database Migrations

```bash
alembic upgrade head                           # Apply migrations
alembic revision --autogenerate -m "desc"      # Create migration
alembic downgrade -1                           # Rollback
```
