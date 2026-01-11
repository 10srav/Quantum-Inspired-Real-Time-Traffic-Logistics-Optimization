# Operational Runbook

## Overview

This runbook provides operational procedures for the Quantum Traffic Optimizer in production environments.

## Table of Contents

1. [Deployment](#deployment)
2. [Health Monitoring](#health-monitoring)
3. [Common Operations](#common-operations)
4. [Troubleshooting](#troubleshooting)
5. [Incident Response](#incident-response)
6. [Backup & Recovery](#backup--recovery)

---

## Deployment

### Prerequisites

- Kubernetes cluster (1.25+)
- kubectl configured
- Helm 3.x (optional)
- AWS CLI configured (for EKS)

### Initial Deployment

```bash
# 1. Create namespace
kubectl apply -f k8s/namespace.yaml

# 2. Create secrets (edit values first!)
kubectl apply -f k8s/secrets.yaml

# 3. Apply ConfigMap
kubectl apply -f k8s/configmap.yaml

# 4. Deploy database and cache
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml

# 5. Wait for dependencies
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=postgres -n quantum-traffic --timeout=120s
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=redis -n quantum-traffic --timeout=60s

# 6. Run database migrations
kubectl exec -it deploy/quantum-traffic-api -n quantum-traffic -- alembic upgrade head

# 7. Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/monitoring.yaml

# 8. Verify deployment
kubectl get pods -n quantum-traffic
kubectl get svc -n quantum-traffic
```

### Rolling Update

```bash
# Update image tag
kubectl set image deployment/quantum-traffic-api \
  quantum-traffic=ghcr.io/your-org/quantum-traffic:v1.2.0 \
  -n quantum-traffic

# Watch rollout
kubectl rollout status deployment/quantum-traffic-api -n quantum-traffic

# Rollback if needed
kubectl rollout undo deployment/quantum-traffic-api -n quantum-traffic
```

### Docker Compose (Development/Testing)

```bash
# Start all services
make docker-up

# Start with dev tools (pgadmin, redis-commander)
make docker-up-full

# View logs
make docker-logs

# Stop services
make docker-down
```

---

## Health Monitoring

### Health Check Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `/health` | Basic health | `{"status": "healthy"}` |
| `/health/live` | Kubernetes liveness | `{"status": "alive"}` |
| `/health/ready` | Kubernetes readiness | `{"status": "ready"}` |
| `/metrics` | Prometheus metrics | Prometheus format |

### Prometheus Queries

**Request Rate:**
```promql
sum(rate(http_requests_total{job="quantum-traffic-api"}[5m]))
```

**Error Rate:**
```promql
sum(rate(http_requests_total{job="quantum-traffic-api",status_code=~"5.."}[5m]))
/
sum(rate(http_requests_total{job="quantum-traffic-api"}[5m]))
```

**P95 Latency:**
```promql
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="quantum-traffic-api"}[5m])) by (le))
```

**Optimization Duration:**
```promql
histogram_quantile(0.95, sum(rate(optimization_duration_seconds_bucket[5m])) by (le))
```

### Alerting

Key alerts to configure:

1. **ServiceDown**: API unreachable for >1 minute
2. **HighErrorRate**: >5% errors for >5 minutes
3. **HighLatency**: P95 >2 seconds for >5 minutes
4. **HighMemory**: >90% memory usage for >5 minutes
5. **DatabaseConnectionFailed**: DB health check failing

---

## Common Operations

### Scaling

```bash
# Manual scale
kubectl scale deployment/quantum-traffic-api --replicas=5 -n quantum-traffic

# Check HPA status
kubectl get hpa -n quantum-traffic

# Update HPA limits
kubectl patch hpa quantum-traffic-hpa -n quantum-traffic \
  --type='json' \
  -p='[{"op": "replace", "path": "/spec/maxReplicas", "value": 20}]'
```

### Database Operations

```bash
# Connect to database
kubectl exec -it statefulset/postgres -n quantum-traffic -- psql -U quantum -d quantum_traffic

# Run migrations
kubectl exec -it deploy/quantum-traffic-api -n quantum-traffic -- alembic upgrade head

# Check migration status
kubectl exec -it deploy/quantum-traffic-api -n quantum-traffic -- alembic current

# Create backup
kubectl exec -it statefulset/postgres -n quantum-traffic -- \
  pg_dump -U quantum quantum_traffic > backup_$(date +%Y%m%d).sql
```

### Cache Operations

```bash
# Connect to Redis
kubectl exec -it statefulset/redis -n quantum-traffic -- redis-cli -a $REDIS_PASSWORD

# Check cache keys
kubectl exec -it statefulset/redis -n quantum-traffic -- \
  redis-cli -a $REDIS_PASSWORD KEYS "qto:*"

# Flush cache
kubectl exec -it statefulset/redis -n quantum-traffic -- \
  redis-cli -a $REDIS_PASSWORD FLUSHDB
```

### Log Access

```bash
# API logs
kubectl logs -f -l app.kubernetes.io/name=quantum-traffic -n quantum-traffic

# Filter by level
kubectl logs -f deploy/quantum-traffic-api -n quantum-traffic | grep ERROR

# Export logs
kubectl logs deploy/quantum-traffic-api -n quantum-traffic --since=1h > api_logs.txt
```

---

## Troubleshooting

### High CPU Usage

**Symptoms:**
- Slow response times
- HPA scaling up rapidly

**Investigation:**
```bash
# Check pod resource usage
kubectl top pods -n quantum-traffic

# Check optimization timeout settings
kubectl exec deploy/quantum-traffic-api -n quantum-traffic -- env | grep QAOA

# Check for runaway optimizations
kubectl logs deploy/quantum-traffic-api -n quantum-traffic | grep "optimization_time"
```

**Resolution:**
1. Reduce QAOA_LAYERS (e.g., 3 → 2)
2. Increase QAOA_TIMEOUT
3. Scale horizontally
4. Disable QAOA for large requests

### High Memory Usage

**Symptoms:**
- OOMKilled pods
- Slow garbage collection

**Investigation:**
```bash
# Check memory usage
kubectl top pods -n quantum-traffic

# Check for memory leaks
kubectl logs deploy/quantum-traffic-api -n quantum-traffic | grep "memory"
```

**Resolution:**
1. Increase memory limits
2. Reduce MAX_CACHED_ROUTES
3. Check for graph caching issues
4. Restart pods if needed

### Database Connection Issues

**Symptoms:**
- `/health/ready` failing
- 500 errors on API

**Investigation:**
```bash
# Check PostgreSQL status
kubectl get pods -l app.kubernetes.io/name=postgres -n quantum-traffic

# Check connection pool
kubectl logs deploy/quantum-traffic-api -n quantum-traffic | grep "database"

# Test connection
kubectl exec deploy/quantum-traffic-api -n quantum-traffic -- \
  python -c "from src.database import db_manager; import asyncio; print(asyncio.run(db_manager.health_check()))"
```

**Resolution:**
1. Check PostgreSQL pod status
2. Verify secrets are correct
3. Increase connection pool size
4. Restart API pods

### Redis Connection Issues

**Symptoms:**
- Cache misses increasing
- Rate limiting not working

**Investigation:**
```bash
# Check Redis status
kubectl get pods -l app.kubernetes.io/name=redis -n quantum-traffic

# Test connection
kubectl exec -it statefulset/redis -n quantum-traffic -- redis-cli ping
```

**Resolution:**
1. Check Redis pod status
2. Verify Redis password in secrets
3. Check network policies
4. API will fall back to in-memory cache

### Graph Loading Failures

**Symptoms:**
- Startup takes too long
- OSM download errors

**Investigation:**
```bash
# Check startup logs
kubectl logs deploy/quantum-traffic-api -n quantum-traffic | grep "graph"

# Check graph cache volume
kubectl exec deploy/quantum-traffic-api -n quantum-traffic -- ls -la /app/data/
```

**Resolution:**
1. Enable OSM_DEMO_MODE for testing
2. Check network connectivity to OSM servers
3. Clear and rebuild graph cache
4. Use pre-cached graph files

---

## Incident Response

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| SEV1 | Complete outage | 15 minutes | All pods down, DB unreachable |
| SEV2 | Major degradation | 30 minutes | >50% errors, high latency |
| SEV3 | Minor impact | 2 hours | Single pod issues, slow queries |
| SEV4 | No immediate impact | 24 hours | Performance optimization needed |

### SEV1 Response

1. **Acknowledge** the incident
2. **Communicate** to stakeholders
3. **Investigate** using health endpoints and logs
4. **Mitigate** by scaling, rollback, or failover
5. **Resolve** root cause
6. **Post-mortem** within 24 hours

### Quick Recovery Commands

```bash
# Restart all API pods
kubectl rollout restart deployment/quantum-traffic-api -n quantum-traffic

# Force delete stuck pods
kubectl delete pod <pod-name> -n quantum-traffic --grace-period=0 --force

# Emergency rollback
kubectl rollout undo deployment/quantum-traffic-api -n quantum-traffic

# Scale to zero and back
kubectl scale deployment/quantum-traffic-api --replicas=0 -n quantum-traffic
kubectl scale deployment/quantum-traffic-api --replicas=3 -n quantum-traffic
```

---

## Backup & Recovery

### Database Backup

**Automated Backup (CronJob):**
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15-alpine
            command:
            - /bin/sh
            - -c
            - pg_dump -h postgres-service -U quantum quantum_traffic | gzip > /backup/db_$(date +%Y%m%d).sql.gz
```

**Manual Backup:**
```bash
# Create backup
kubectl exec statefulset/postgres -n quantum-traffic -- \
  pg_dump -U quantum quantum_traffic > backup.sql

# Compress
gzip backup.sql

# Copy to local
kubectl cp quantum-traffic/postgres-0:backup.sql.gz ./backup.sql.gz
```

### Database Restore

```bash
# Stop API pods
kubectl scale deployment/quantum-traffic-api --replicas=0 -n quantum-traffic

# Restore from backup
kubectl exec -i statefulset/postgres -n quantum-traffic -- \
  psql -U quantum quantum_traffic < backup.sql

# Restart API pods
kubectl scale deployment/quantum-traffic-api --replicas=3 -n quantum-traffic
```

### Redis Recovery

Redis data is non-critical (cache only). Recovery is automatic on restart.

```bash
# If needed, flush and warm cache
kubectl exec statefulset/redis -n quantum-traffic -- redis-cli -a $REDIS_PASSWORD FLUSHALL

# Warm cache by sending test requests
curl -X POST http://api-endpoint/optimize -d '...'
```

---

## Contact Information

| Role | Contact | Escalation |
|------|---------|------------|
| On-Call Engineer | PagerDuty | Immediate |
| Platform Team | #platform-support | 15 min |
| Database Admin | #dba-team | 30 min |
| Security Team | security@company.com | As needed |
