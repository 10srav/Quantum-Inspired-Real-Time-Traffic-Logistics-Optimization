-- =============================================================================
-- Database Initialization Script for Quantum Traffic Optimizer
-- =============================================================================
-- This script runs when PostgreSQL container is first created
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schema (tables will be created by Alembic migrations)
-- This is just for initial setup

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE quantum_traffic TO quantum;

-- Create indexes for common queries (if not created by Alembic)
-- These are just examples, actual indexes are in Alembic migrations
