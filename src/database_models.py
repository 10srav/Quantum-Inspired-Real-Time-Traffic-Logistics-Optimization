"""
Database Models for Quantum Traffic Optimizer.

This module defines SQLAlchemy ORM models for persistent storage
of routes, optimization metrics, and API keys.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid4())


class Route(Base):
    """
    Optimized route storage.

    Stores route optimization results for retrieval and analytics.
    """

    __tablename__ = "routes"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        server_default=func.now(),
        nullable=False
    )

    # User association (for multi-tenancy)
    user_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        index=True,
        nullable=True
    )
    api_key_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        index=True,
        nullable=True
    )

    # Route data
    current_location: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Starting location {lat, lng}"
    )
    deliveries: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        comment="List of delivery points"
    )
    sequence: Mapped[List[int]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Optimized sequence indices"
    )

    # Route metrics
    total_distance: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Total distance in meters"
    )
    total_eta: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Total ETA in minutes"
    )
    optimization_time: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Optimization compute time in seconds"
    )
    traffic_level: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="Traffic level: low/medium/high"
    )
    improvement_over_greedy: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Percentage improvement over greedy baseline"
    )

    # Map storage (consider moving to S3 for large maps)
    map_html: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Folium map HTML (can be large)"
    )

    # Metadata
    n_deliveries: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Number of deliveries"
    )
    algorithm_used: Mapped[str] = mapped_column(
        String(32),
        default="brute_force",
        nullable=False,
        comment="Optimization algorithm used"
    )
    request_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional request metadata"
    )

    # Relationships
    metrics: Mapped[List["OptimizationMetric"]] = relationship(
        "OptimizationMetric",
        back_populates="route",
        cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_routes_created_at", "created_at"),
        Index("idx_routes_user_created", "user_id", "created_at"),
        Index("idx_routes_traffic_level", "traffic_level"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "route_id": self.id,
            "created_at": self.created_at.isoformat(),
            "current_location": self.current_location,
            "deliveries": self.deliveries,
            "sequence": self.sequence,
            "total_distance": self.total_distance,
            "total_eta": self.total_eta,
            "optimization_time": self.optimization_time,
            "traffic_level": self.traffic_level,
            "improvement_over_greedy": self.improvement_over_greedy,
            "n_deliveries": self.n_deliveries,
            "algorithm_used": self.algorithm_used,
        }


class OptimizationMetric(Base):
    """
    Optimization performance metrics for analytics.

    Stores detailed metrics for each optimization run.
    """

    __tablename__ = "optimization_metrics"

    # Primary key
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    # Foreign key to route
    route_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("routes.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=func.now(),
        nullable=False
    )

    # Optimization parameters
    n_deliveries: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )
    algorithm_used: Mapped[str] = mapped_column(
        String(32),
        nullable=False
    )
    traffic_level: Mapped[str] = mapped_column(
        String(16),
        nullable=False
    )

    # Performance metrics
    execution_time_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Execution time in milliseconds"
    )
    memory_usage_mb: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Peak memory usage in MB"
    )

    # QAOA-specific metrics
    qaoa_layers: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    qaoa_iterations: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    qaoa_converged: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True
    )

    # Result quality metrics
    cost_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Optimization cost value"
    )
    greedy_cost: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Greedy baseline cost"
    )
    improvement_percentage: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True
    )

    # Relationships
    route: Mapped["Route"] = relationship(
        "Route",
        back_populates="metrics"
    )

    # Indexes
    __table_args__ = (
        Index("idx_metrics_timestamp", "timestamp"),
        Index("idx_metrics_algorithm", "algorithm_used"),
    )


class APIKey(Base):
    """
    API key storage for authentication.

    Stores hashed API keys with metadata for access control.
    """

    __tablename__ = "api_keys"

    # Primary key
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    # Key data (store hash, not plain text)
    key_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True
    )
    key_prefix: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        comment="First 8 chars for identification"
    )

    # Metadata
    name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Human-readable key name"
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    owner_email: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=func.now(),
        nullable=False
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )

    # Usage tracking
    request_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )

    # Permissions
    scopes: Mapped[List[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
        comment="List of allowed scopes"
    )
    rate_limit_override: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Custom rate limit per minute"
    )

    # Indexes
    __table_args__ = (
        Index("idx_api_keys_active", "is_active"),
    )

    def is_valid(self) -> bool:
        """Check if the key is valid (active and not expired)."""
        if not self.is_active:
            return False
        if self.expires_at and self.expires_at < utc_now():
            return False
        return True


class AuditLog(Base):
    """
    Audit log for security and compliance.

    Records important actions for security monitoring.
    """

    __tablename__ = "audit_logs"

    # Primary key
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=func.now(),
        nullable=False,
        index=True
    )

    # Action details
    action: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Action type: create_route, delete_route, auth_success, auth_failure, etc."
    )
    resource_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Resource type: route, api_key, etc."
    )
    resource_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True
    )

    # Actor information
    user_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True
    )
    api_key_prefix: Mapped[Optional[str]] = mapped_column(
        String(8),
        nullable=True
    )
    ip_address: Mapped[Optional[str]] = mapped_column(
        String(45),  # IPv6 max length
        nullable=True
    )
    user_agent: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True
    )

    # Additional context
    details: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional action details"
    )
    status: Mapped[str] = mapped_column(
        String(16),
        default="success",
        nullable=False,
        comment="Action status: success, failure, error"
    )

    # Indexes
    __table_args__ = (
        Index("idx_audit_user_action", "user_id", "action"),
        Index("idx_audit_timestamp_action", "timestamp", "action"),
    )
