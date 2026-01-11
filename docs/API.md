# API Documentation

## Overview

The Quantum Traffic Optimizer API provides endpoints for route optimization using quantum-inspired algorithms.

**Base URL:** `https://api.quantum-traffic.example.com` (production)

**API Version:** v1.0.0

## Authentication

The API supports two authentication methods:

### JWT Bearer Token

```http
Authorization: Bearer <token>
```

Obtain tokens via the authentication endpoint (if configured).

### API Key

```http
X-API-Key: <your-api-key>
```

Contact the administrator to obtain an API key.

### Development Mode

In development mode (`ENVIRONMENT=development`), authentication is optional.

## Rate Limiting

- **Per minute:** 60 requests/IP
- **Per hour:** 1000 requests/IP

Rate limit headers:
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1699999999
```

---

## Endpoints

### Health Checks

#### GET /health

Basic health check.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "graph_loaded": true
}
```

#### GET /health/live

Kubernetes liveness probe.

**Response:**
```json
{
  "status": "alive"
}
```

#### GET /health/ready

Kubernetes readiness probe.

**Response:**
```json
{
  "status": "ready",
  "checks": {
    "graph": {"status": "healthy", "healthy": true},
    "optimizer": {"status": "healthy", "healthy": true},
    "database": {"status": "healthy", "healthy": true},
    "cache": {"status": "healthy", "healthy": true}
  }
}
```

---

### Route Optimization

#### POST /optimize

Optimize delivery route using quantum-inspired algorithms.

**Request Body:**
```json
{
  "current_loc": [16.5063, 80.6480],
  "deliveries": [
    {
      "lat": 16.5175,
      "lng": 80.6198,
      "priority": 2,
      "name": "Customer A",
      "id": "delivery-001"
    },
    {
      "lat": 16.5412,
      "lng": 80.6352,
      "priority": 1,
      "name": "Customer B",
      "id": "delivery-002"
    }
  ],
  "traffic_level": "medium",
  "include_map": true
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `current_loc` | `[float, float]` | Yes | Starting location [lat, lng] |
| `deliveries` | `array` | Yes | List of delivery points (1-20) |
| `deliveries[].lat` | `float` | Yes | Latitude (16.50-16.55) |
| `deliveries[].lng` | `float` | Yes | Longitude (80.62-80.68) |
| `deliveries[].priority` | `int` | No | Priority 0-10 (default: 0) |
| `deliveries[].name` | `string` | No | Display name |
| `deliveries[].id` | `string` | No | Unique identifier |
| `traffic_level` | `string` | No | "low", "medium", "high" (default: "medium") |
| `include_map` | `bool` | No | Include Folium map HTML (default: false) |

**Response:**
```json
{
  "route_id": "abc123xy",
  "sequence": [
    {
      "position": 0,
      "delivery": {
        "lat": 16.5412,
        "lng": 80.6352,
        "priority": 1,
        "name": "Customer B",
        "id": "delivery-002"
      },
      "distance_from_prev": 1250.5,
      "eta_from_prev": 3.5,
      "cumulative_distance": 1250.5,
      "cumulative_eta": 3.5
    },
    {
      "position": 1,
      "delivery": {
        "lat": 16.5175,
        "lng": 80.6198,
        "priority": 2,
        "name": "Customer A",
        "id": "delivery-001"
      },
      "distance_from_prev": 890.3,
      "eta_from_prev": 2.8,
      "cumulative_distance": 2140.8,
      "cumulative_eta": 6.3
    }
  ],
  "total_distance": 2140.8,
  "total_eta": 6.3,
  "optimization_time": 0.245,
  "traffic_level": "medium",
  "map_html": "<html>...</html>",
  "improvement_over_greedy": 15.3
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `route_id` | `string` | Unique route identifier |
| `sequence` | `array` | Optimized delivery sequence |
| `sequence[].position` | `int` | Position in sequence (0-indexed) |
| `sequence[].delivery` | `object` | Delivery point details |
| `sequence[].distance_from_prev` | `float` | Distance from previous stop (meters) |
| `sequence[].eta_from_prev` | `float` | ETA from previous stop (minutes) |
| `sequence[].cumulative_distance` | `float` | Total distance so far (meters) |
| `sequence[].cumulative_eta` | `float` | Total ETA so far (minutes) |
| `total_distance` | `float` | Total route distance (meters) |
| `total_eta` | `float` | Total estimated time (minutes) |
| `optimization_time` | `float` | Algorithm runtime (seconds) |
| `traffic_level` | `string` | Applied traffic level |
| `map_html` | `string` | Folium map HTML (if requested) |
| `improvement_over_greedy` | `float` | % improvement over greedy baseline |

**Error Responses:**

| Status | Description |
|--------|-------------|
| 400 | Invalid request (validation error) |
| 429 | Rate limit exceeded |
| 500 | Optimization failed |
| 503 | Service not available (graph not loaded) |

---

### Route Management

#### GET /map/{route_id}

Get the interactive Folium map for a route.

**Response:** HTML page with interactive map

#### GET /routes

List all cached routes.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip` | `int` | 0 | Number of routes to skip |
| `limit` | `int` | 20 | Maximum routes to return (1-100) |

**Response:**
```json
{
  "routes": [
    {
      "route_id": "abc123xy",
      "n_deliveries": 5,
      "traffic_level": "medium",
      "total_distance": 5230.5,
      "total_eta": 18.7
    }
  ],
  "total": 42,
  "skip": 0,
  "limit": 20
}
```

#### DELETE /routes/{route_id}

Delete a cached route.

**Response:**
```json
{
  "message": "Route abc123xy deleted"
}
```

---

### WebSocket

#### WS /reoptimize

Real-time route updates via WebSocket.

**Connection:**
```javascript
const ws = new WebSocket('wss://api.quantum-traffic.example.com/reoptimize');
```

**Messages:**

**Traffic Change:**
```json
{
  "type": "traffic_change",
  "data": {
    "level": "high"
  }
}
```

**Acknowledgment:**
```json
{
  "type": "ack",
  "data": {
    "message": "Traffic updated to high"
  }
}
```

---

### Metrics

#### GET /metrics

Prometheus metrics endpoint.

**Response:** Prometheus text format

**Available Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests |
| `http_request_duration_seconds` | Histogram | Request latency |
| `optimization_requests_total` | Counter | Optimization requests |
| `optimization_duration_seconds` | Histogram | Optimization time |
| `routes_cached_total` | Gauge | Routes in cache |
| `osm_graph_loaded` | Gauge | Graph status |

---

## Error Handling

All errors return a consistent format:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error description",
  "details": {}
}
```

### Error Types

| Error | Status | Description |
|-------|--------|-------------|
| `ValidationError` | 400 | Invalid request parameters |
| `RateLimitExceeded` | 429 | Too many requests |
| `GraphNotLoaded` | 503 | Graph not available |
| `OptimizationFailed` | 500 | Algorithm failure |
| `RouteNotFound` | 404 | Route ID not found |
| `AuthenticationRequired` | 401 | Missing credentials |
| `Forbidden` | 403 | Invalid credentials |

---

## Code Examples

### Python

```python
import httpx

API_URL = "https://api.quantum-traffic.example.com"
API_KEY = "your-api-key"

async def optimize_route():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/optimize",
            headers={"X-API-Key": API_KEY},
            json={
                "current_loc": [16.5063, 80.6480],
                "deliveries": [
                    {"lat": 16.5175, "lng": 80.6198, "priority": 2},
                    {"lat": 16.5412, "lng": 80.6352, "priority": 1},
                ],
                "traffic_level": "medium",
                "include_map": False,
            },
        )
        return response.json()
```

### JavaScript

```javascript
const optimizeRoute = async () => {
  const response = await fetch('https://api.quantum-traffic.example.com/optimize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': 'your-api-key',
    },
    body: JSON.stringify({
      current_loc: [16.5063, 80.6480],
      deliveries: [
        { lat: 16.5175, lng: 80.6198, priority: 2 },
        { lat: 16.5412, lng: 80.6352, priority: 1 },
      ],
      traffic_level: 'medium',
      include_map: false,
    }),
  });
  return response.json();
};
```

### cURL

```bash
curl -X POST https://api.quantum-traffic.example.com/optimize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "current_loc": [16.5063, 80.6480],
    "deliveries": [
      {"lat": 16.5175, "lng": 80.6198, "priority": 2},
      {"lat": 16.5412, "lng": 80.6352, "priority": 1}
    ],
    "traffic_level": "medium",
    "include_map": false
  }'
```

---

## SDK/Client Libraries

Official client libraries are available for:

- **Python:** `pip install quantum-traffic-client`
- **JavaScript:** `npm install @quantum-traffic/client`
- **Go:** `go get github.com/quantum-traffic/go-client`

---

## Changelog

### v1.0.0 (2024-01-01)

- Initial release
- QUBO/QAOA optimization
- Traffic-aware routing
- Folium map generation
- WebSocket support
- PostgreSQL persistence
- Redis caching
- Prometheus metrics
