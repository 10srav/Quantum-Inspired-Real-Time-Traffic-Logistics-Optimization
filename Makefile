# =============================================================================
# Quantum Traffic Optimizer - Makefile
# =============================================================================
# Common operations for development, testing, and deployment
# =============================================================================

.PHONY: help install dev test lint format security docker docker-build docker-up docker-down \
        migrate migrate-create db-shell redis-cli clean docs run benchmark

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
DOCKER_COMPOSE := docker compose
PYTEST := pytest
APP_NAME := quantum-traffic-optimizer

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "$(BLUE)Quantum Traffic Optimizer - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make install      # Install dependencies"
	@echo "  make dev          # Start development server"
	@echo "  make docker-up    # Start all services with Docker"
	@echo "  make test         # Run tests"

# =============================================================================
# Installation
# =============================================================================

install: ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install black isort flake8 mypy bandit safety pytest pytest-cov pytest-asyncio httpx locust pre-commit
	pre-commit install

install-all: install install-dev ## Install all dependencies

# =============================================================================
# Development
# =============================================================================

dev: ## Start development server with auto-reload
	$(PYTHON) -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

run: ## Start production server
	$(PYTHON) -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

shell: ## Start Python shell with app context
	$(PYTHON) -c "from src.main import *; import IPython; IPython.embed()"

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	OSM_DEMO_MODE=true DATABASE_ENABLED=false REDIS_ENABLED=false \
	$(PYTEST) tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	OSM_DEMO_MODE=true DATABASE_ENABLED=false REDIS_ENABLED=false \
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage (faster)
	OSM_DEMO_MODE=true DATABASE_ENABLED=false REDIS_ENABLED=false \
	$(PYTEST) tests/ -v --tb=short -x

test-api: ## Run API tests only
	OSM_DEMO_MODE=true DATABASE_ENABLED=false REDIS_ENABLED=false \
	$(PYTEST) tests/test_api.py -v

test-performance: ## Run performance benchmarks
	$(PYTHON) -m tests.performance.benchmark

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run all linters
	@echo "$(BLUE)Running flake8...$(NC)"
	flake8 src/ tests/ --max-line-length=120
	@echo "$(BLUE)Running mypy...$(NC)"
	mypy src/ --ignore-missing-imports || true
	@echo "$(GREEN)Linting complete!$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Running isort...$(NC)"
	isort src/ tests/
	@echo "$(BLUE)Running black...$(NC)"
	black src/ tests/ --line-length=120
	@echo "$(GREEN)Formatting complete!$(NC)"

format-check: ## Check code formatting without making changes
	black --check --diff src/ tests/ --line-length=120
	isort --check-only --diff src/ tests/

security: ## Run security scans
	@echo "$(BLUE)Running bandit security scan...$(NC)"
	bandit -r src/ -ll
	@echo "$(BLUE)Checking dependencies with safety...$(NC)"
	safety check -r requirements.txt || true
	@echo "$(GREEN)Security scan complete!$(NC)"

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build Docker images
	$(DOCKER_COMPOSE) build

docker-up: ## Start all services
	$(DOCKER_COMPOSE) up -d

docker-up-full: ## Start all services including dev tools
	$(DOCKER_COMPOSE) --profile dev --profile frontend up -d

docker-down: ## Stop all services
	$(DOCKER_COMPOSE) down

docker-logs: ## View container logs
	$(DOCKER_COMPOSE) logs -f

docker-logs-api: ## View API logs
	$(DOCKER_COMPOSE) logs -f api

docker-restart: ## Restart all services
	$(DOCKER_COMPOSE) restart

docker-clean: ## Remove all containers and volumes
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -f

docker-shell: ## Open shell in API container
	$(DOCKER_COMPOSE) exec api /bin/sh

# =============================================================================
# Database
# =============================================================================

migrate: ## Run database migrations
	alembic upgrade head

migrate-create: ## Create a new migration (usage: make migrate-create MSG="description")
	alembic revision --autogenerate -m "$(MSG)"

migrate-down: ## Rollback last migration
	alembic downgrade -1

migrate-history: ## Show migration history
	alembic history --verbose

db-shell: ## Connect to PostgreSQL shell
	$(DOCKER_COMPOSE) exec postgres psql -U quantum -d quantum_traffic

db-reset: ## Reset database (WARNING: destroys all data)
	$(DOCKER_COMPOSE) exec postgres psql -U quantum -d postgres -c "DROP DATABASE IF EXISTS quantum_traffic;"
	$(DOCKER_COMPOSE) exec postgres psql -U quantum -d postgres -c "CREATE DATABASE quantum_traffic;"
	$(MAKE) migrate

# =============================================================================
# Redis
# =============================================================================

redis-cli: ## Connect to Redis CLI
	$(DOCKER_COMPOSE) exec redis redis-cli -a redis123

redis-flush: ## Flush all Redis data
	$(DOCKER_COMPOSE) exec redis redis-cli -a redis123 FLUSHALL

# =============================================================================
# Load Testing
# =============================================================================

loadtest: ## Run load tests with Locust (web UI)
	locust -f tests/performance/locustfile.py --host=http://localhost:8000

loadtest-headless: ## Run headless load test (100 users, 2 min)
	locust -f tests/performance/locustfile.py \
		--host=http://localhost:8000 \
		--users 100 --spawn-rate 10 --run-time 2m \
		--headless --html=load_test_report.html

benchmark: ## Run performance benchmarks
	$(PYTHON) -m tests.performance.benchmark

# =============================================================================
# Documentation
# =============================================================================

docs: ## Generate documentation
	@echo "$(BLUE)Generating API documentation...$(NC)"
	@echo "Visit http://localhost:8000/docs for Swagger UI"
	@echo "Visit http://localhost:8000/redoc for ReDoc"

docs-serve: ## Serve documentation locally
	mkdocs serve

# =============================================================================
# Kubernetes
# =============================================================================

k8s-apply: ## Apply Kubernetes manifests
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/secrets.yaml
	kubectl apply -f k8s/postgres.yaml
	kubectl apply -f k8s/redis.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl apply -f k8s/ingress.yaml
	kubectl apply -f k8s/hpa.yaml

k8s-delete: ## Delete Kubernetes resources
	kubectl delete -f k8s/ --ignore-not-found

k8s-logs: ## View Kubernetes pod logs
	kubectl logs -f -l app.kubernetes.io/name=quantum-traffic -n quantum-traffic

k8s-status: ## Show Kubernetes deployment status
	kubectl get all -n quantum-traffic

# =============================================================================
# Terraform
# =============================================================================

tf-init: ## Initialize Terraform
	cd terraform && terraform init

tf-plan: ## Plan Terraform changes
	cd terraform && terraform plan

tf-apply: ## Apply Terraform changes
	cd terraform && terraform apply

tf-destroy: ## Destroy Terraform resources
	cd terraform && terraform destroy

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "coverage.xml" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(NC)"

clean-all: clean docker-clean ## Clean everything including Docker
	@echo "$(GREEN)Full cleanup complete!$(NC)"

# =============================================================================
# Release
# =============================================================================

version: ## Show current version
	@grep -m1 'version' pyproject.toml | cut -d'"' -f2

tag: ## Create a git tag (usage: make tag VERSION=1.0.0)
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	git push origin v$(VERSION)
