"""Initial schema for Quantum Traffic Optimizer.

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema."""
    # Create routes table
    op.create_table(
        "routes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=True),
        sa.Column("api_key_hash", sa.String(64), nullable=True),
        sa.Column("current_location", postgresql.JSONB(), nullable=False),
        sa.Column("deliveries", postgresql.JSONB(), nullable=False),
        sa.Column("sequence", postgresql.JSONB(), nullable=False),
        sa.Column("total_distance", sa.Float(), nullable=False),
        sa.Column("total_eta", sa.Float(), nullable=False),
        sa.Column("optimization_time", sa.Float(), nullable=False),
        sa.Column("traffic_level", sa.String(16), nullable=False),
        sa.Column("improvement_over_greedy", sa.Float(), nullable=True),
        sa.Column("map_html", sa.Text(), nullable=True),
        sa.Column("n_deliveries", sa.Integer(), nullable=False),
        sa.Column("algorithm_used", sa.String(32), default="brute_force", nullable=False),
        sa.Column("request_metadata", postgresql.JSONB(), nullable=True),
    )

    # Create indexes for routes
    op.create_index("idx_routes_created_at", "routes", ["created_at"])
    op.create_index("idx_routes_user_created", "routes", ["user_id", "created_at"])
    op.create_index("idx_routes_traffic_level", "routes", ["traffic_level"])
    op.create_index("idx_routes_user_id", "routes", ["user_id"])
    op.create_index("idx_routes_api_key_hash", "routes", ["api_key_hash"])

    # Create optimization_metrics table
    op.create_table(
        "optimization_metrics",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("route_id", sa.String(36), sa.ForeignKey("routes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("n_deliveries", sa.Integer(), nullable=False),
        sa.Column("algorithm_used", sa.String(32), nullable=False),
        sa.Column("traffic_level", sa.String(16), nullable=False),
        sa.Column("execution_time_ms", sa.Float(), nullable=False),
        sa.Column("memory_usage_mb", sa.Float(), nullable=True),
        sa.Column("qaoa_layers", sa.Integer(), nullable=True),
        sa.Column("qaoa_iterations", sa.Integer(), nullable=True),
        sa.Column("qaoa_converged", sa.Boolean(), nullable=True),
        sa.Column("cost_value", sa.Float(), nullable=False),
        sa.Column("greedy_cost", sa.Float(), nullable=True),
        sa.Column("improvement_percentage", sa.Float(), nullable=True),
    )

    # Create indexes for optimization_metrics
    op.create_index("idx_metrics_route_id", "optimization_metrics", ["route_id"])
    op.create_index("idx_metrics_timestamp", "optimization_metrics", ["timestamp"])
    op.create_index("idx_metrics_algorithm", "optimization_metrics", ["algorithm_used"])

    # Create api_keys table
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("key_hash", sa.String(64), unique=True, nullable=False),
        sa.Column("key_prefix", sa.String(8), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("owner_email", sa.String(256), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True, nullable=False),
        sa.Column("request_count", sa.Integer(), default=0, nullable=False),
        sa.Column("scopes", postgresql.JSONB(), default=list, nullable=False),
        sa.Column("rate_limit_override", sa.Integer(), nullable=True),
    )

    # Create indexes for api_keys
    op.create_index("idx_api_keys_key_hash", "api_keys", ["key_hash"])
    op.create_index("idx_api_keys_active", "api_keys", ["is_active"])

    # Create audit_logs table
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("resource_type", sa.String(32), nullable=False),
        sa.Column("resource_id", sa.String(64), nullable=True),
        sa.Column("user_id", sa.String(64), nullable=True),
        sa.Column("api_key_prefix", sa.String(8), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.String(512), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column("status", sa.String(16), default="success", nullable=False),
    )

    # Create indexes for audit_logs
    op.create_index("idx_audit_timestamp", "audit_logs", ["timestamp"])
    op.create_index("idx_audit_action", "audit_logs", ["action"])
    op.create_index("idx_audit_user_id", "audit_logs", ["user_id"])
    op.create_index("idx_audit_user_action", "audit_logs", ["user_id", "action"])
    op.create_index("idx_audit_timestamp_action", "audit_logs", ["timestamp", "action"])


def downgrade() -> None:
    """Downgrade database schema."""
    # Drop audit_logs
    op.drop_index("idx_audit_timestamp_action", table_name="audit_logs")
    op.drop_index("idx_audit_user_action", table_name="audit_logs")
    op.drop_index("idx_audit_user_id", table_name="audit_logs")
    op.drop_index("idx_audit_action", table_name="audit_logs")
    op.drop_index("idx_audit_timestamp", table_name="audit_logs")
    op.drop_table("audit_logs")

    # Drop api_keys
    op.drop_index("idx_api_keys_active", table_name="api_keys")
    op.drop_index("idx_api_keys_key_hash", table_name="api_keys")
    op.drop_table("api_keys")

    # Drop optimization_metrics
    op.drop_index("idx_metrics_algorithm", table_name="optimization_metrics")
    op.drop_index("idx_metrics_timestamp", table_name="optimization_metrics")
    op.drop_index("idx_metrics_route_id", table_name="optimization_metrics")
    op.drop_table("optimization_metrics")

    # Drop routes
    op.drop_index("idx_routes_api_key_hash", table_name="routes")
    op.drop_index("idx_routes_user_id", table_name="routes")
    op.drop_index("idx_routes_traffic_level", table_name="routes")
    op.drop_index("idx_routes_user_created", table_name="routes")
    op.drop_index("idx_routes_created_at", table_name="routes")
    op.drop_table("routes")
