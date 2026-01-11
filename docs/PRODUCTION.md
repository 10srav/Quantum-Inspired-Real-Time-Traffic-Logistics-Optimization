# Production Deployment Guide

This guide covers deploying the Quantum Traffic Optimizer to production.

## Prerequisites

- Python 3.11+
- PostgreSQL 15+ (optional, for persistence)
- Redis 7+ (optional, for caching)
- Docker (optional, for containerized deployment)

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Quantum-Inspired-Real-Time-Traffic-Logistics-Optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# IMPORTANT: Change JWT_SECRET_KEY for production!
```

### 3. Run the Application

```bash
# Development mode
python -m src.main

# Production mode with Uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Configuration

### Environment Variables

All configuration is done via environment variables. See `.env.example` for all options.

#### Critical Production Settings

| Variable | Description | Required |
|----------|-------------|----------|
| `ENVIRONMENT` | Set to `production` | Yes |
| `DEBUG` | Set to `False` | Yes |
| `JWT_SECRET_KEY` | Unique secret key (32+ chars) | Yes |
| `CORS_ORIGINS` | Allowed origins (no wildcards) | Yes |
| `DATABASE_URL` | PostgreSQL connection string | If using DB |
| `REDIS_URL` | Redis connection string | If using cache |

#### Generate Secure Keys

```bash
# Generate JWT secret
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate API key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Security Checklist

- [ ] Changed `JWT_SECRET_KEY` from default
- [ ] Set `ENVIRONMENT=production`
- [ ] Set `DEBUG=False`
- [ ] Configured specific `CORS_ORIGINS` (no wildcards)
- [ ] Enabled `RATE_LIMIT_ENABLED=True`
- [ ] Configured `API_KEYS_ENABLED=True` if using API keys
- [ ] Using HTTPS (via reverse proxy)
- [ ] Database credentials secured
- [ ] Redis password set (if using Redis)

## Database Setup

### PostgreSQL

```bash
# Create database
createdb quantum_traffic

# Run migrations (if using Alembic)
alembic upgrade head
```

### Enable in Configuration

```env
DATABASE_ENABLED=True
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/quantum_traffic
```

## Redis Setup

### Install and Start Redis

```bash
# Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis

# Docker
docker run -d -p 6379:6379 redis:7-alpine
```

### Enable in Configuration

```env
REDIS_ENABLED=True
REDIS_URL=redis://localhost:6379/0
```

## Monitoring

### Prometheus Metrics

Metrics are available at `/metrics` when `METRICS_ENABLED=True`.

Key metrics:
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency
- `optimization_requests_total` - Optimization requests
- `optimization_duration_seconds` - Optimization time
- `routes_cached_total` - Cached routes count

### Health Checks

| Endpoint | Description |
|----------|-------------|
| `/health` | Basic health check |
| `/health/live` | Kubernetes liveness probe |
| `/health/ready` | Kubernetes readiness probe |

### Logging

Configure logging format:

```env
LOG_LEVEL=INFO
LOG_FORMAT=json  # Use 'json' for production log aggregation
```

JSON logs include:
- Timestamp
- Correlation ID
- Request context
- Error details

## Deployment Options

### Docker

```bash
# Build image
docker build -t quantum-traffic:latest .

# Run container
docker run -d \
  --name quantum-traffic \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e JWT_SECRET_KEY=your-secret-key \
  quantum-traffic:latest
```

### Docker Compose

```bash
docker-compose up -d
```

### Kubernetes

See `k8s/` directory for Kubernetes manifests (if available).

## Performance Tuning

### Worker Configuration

```env
WORKERS=4  # Number of CPU cores
```

### Connection Pools

```env
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10
```

### Caching

Enable Redis caching for better performance:

```env
REDIS_ENABLED=True
CACHE_TTL=3600  # 1 hour
```

### QAOA Settings

For faster API responses, disable QAOA:

```env
USE_QAOA_IN_API=False  # Uses brute force/SA instead
```

## Troubleshooting

### Common Issues

**1. Graph not loading**
- Check `OSM_CACHE_DIR` exists and is writable
- Verify network access to OpenStreetMap
- Try `OSM_DEMO_MODE=True` for testing

**2. Database connection errors**
- Verify PostgreSQL is running
- Check `DATABASE_URL` format
- Ensure database exists

**3. Redis connection errors**
- Verify Redis is running
- Check `REDIS_URL` format
- Test with `redis-cli ping`

**4. Rate limiting issues**
- Adjust `RATE_LIMIT_PER_MINUTE`
- Check client IP detection (X-Forwarded-For)

### Debug Mode

Enable debug mode for troubleshooting (never in production):

```env
DEBUG=True
LOG_LEVEL=DEBUG
```

## API Documentation

- Swagger UI: `/docs` (development only)
- ReDoc: `/redoc` (development only)
- OpenAPI spec: `/openapi.json` (development only)

To enable in production:

```env
DEBUG=True  # Or configure separately
```

## Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: [repository-url]/docs
