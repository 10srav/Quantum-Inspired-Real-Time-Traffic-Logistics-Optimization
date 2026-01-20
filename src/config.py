"""
Configuration Management for Quantum Traffic Optimizer.

This module provides centralized configuration using pydantic-settings
with environment variable support for production deployments.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden via environment variables.
    For example, APP_NAME can be set via the APP_NAME environment variable.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # =========================
    # Application Settings
    # =========================
    APP_NAME: str = Field(default="Quantum Traffic Optimizer", description="Application name")
    APP_VERSION: str = Field(default="1.0.0", description="Application version")
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    ENVIRONMENT: str = Field(default="development", description="Environment (development/staging/production)")

    # =========================
    # Server Settings
    # =========================
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    WORKERS: int = Field(default=4, description="Number of worker processes")
    RELOAD: bool = Field(default=False, description="Enable auto-reload")

    # =========================
    # API Settings
    # =========================
    API_PREFIX: str = Field(default="/api/v1", description="API prefix for versioning")
    MAX_DELIVERIES: int = Field(default=20, description="Maximum deliveries per request")
    REQUEST_TIMEOUT: int = Field(default=30, description="Request timeout in seconds")
    MAX_CACHED_ROUTES: int = Field(default=100, description="Maximum routes in memory cache")

    # =========================
    # Security Settings
    # =========================
    # JWT Configuration
    JWT_SECRET_KEY: str = Field(
        default="CHANGE-THIS-IN-PRODUCTION-USE-SECURE-KEY",
        description="JWT secret key (MUST change in production)"
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiration")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token expiration")

    # API Keys
    API_KEYS_ENABLED: bool = Field(default=False, description="Enable API key authentication")
    VALID_API_KEYS: str = Field(default="", description="Comma-separated list of valid API keys")

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="Requests per minute per IP")
    RATE_LIMIT_PER_HOUR: int = Field(default=1000, description="Requests per hour per IP")

    # CORS
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8501,http://localhost:5173,http://localhost:5174,http://localhost:5175,http://localhost:5176",
        description="Comma-separated allowed origins"
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True, description="Allow credentials in CORS")

    # =========================
    # Database Settings
    # =========================
    DATABASE_ENABLED: bool = Field(default=False, description="Enable database persistence")
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/quantum_traffic",
        description="Database connection URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=20, description="Connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(default=10, description="Max overflow connections")
    DATABASE_POOL_RECYCLE: int = Field(default=3600, description="Connection recycle time (seconds)")

    # =========================
    # Redis Settings
    # =========================
    REDIS_ENABLED: bool = Field(default=False, description="Enable Redis caching")
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    REDIS_PREFIX: str = Field(default="qto:", description="Redis key prefix")
    CACHE_TTL: int = Field(default=3600, description="Default cache TTL in seconds")

    # =========================
    # OSM/Graph Settings
    # =========================
    BBOX_SOUTH: float = Field(default=16.50, description="Bounding box south latitude")
    BBOX_NORTH: float = Field(default=16.55, description="Bounding box north latitude")
    BBOX_WEST: float = Field(default=80.62, description="Bounding box west longitude")
    BBOX_EAST: float = Field(default=80.68, description="Bounding box east longitude")
    OSM_CACHE_DIR: str = Field(default="./data", description="OSM graph cache directory")
    OSM_USE_CACHE: bool = Field(default=True, description="Use cached OSM graph")
    OSM_DEMO_MODE: bool = Field(default=False, description="Use synthetic demo graph")

    # =========================
    # QAOA/Optimization Settings
    # =========================
    QAOA_LAYERS: int = Field(default=3, description="Number of QAOA layers")
    QAOA_TIMEOUT: float = Field(default=5.0, description="QAOA optimization timeout")
    QAOA_SEED: int = Field(default=42, description="Random seed for reproducibility")
    USE_QAOA_IN_API: bool = Field(default=False, description="Use QAOA in API (slower)")

    # =========================
    # Traffic Settings
    # =========================
    TRAFFIC_LOW_MULTIPLIER: float = Field(default=1.0, description="Low traffic speed multiplier")
    TRAFFIC_MEDIUM_MULTIPLIER: float = Field(default=1.5, description="Medium traffic speed multiplier")
    TRAFFIC_HIGH_MULTIPLIER: float = Field(default=2.5, description="High traffic speed multiplier")
    DEFAULT_SPEED_KMH: float = Field(default=30.0, description="Default speed in km/h")

    # =========================
    # Monitoring Settings
    # =========================
    SENTRY_DSN: str = Field(default="", description="Sentry DSN for error tracking")
    SENTRY_TRACES_SAMPLE_RATE: float = Field(default=0.1, description="Sentry traces sample rate")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Log format (json/text)")
    METRICS_ENABLED: bool = Field(default=True, description="Enable Prometheus metrics")
    METRICS_PATH: str = Field(default="/metrics", description="Metrics endpoint path")

    # =========================
    # Feature Flags
    # =========================
    FEATURE_WEBSOCKET_ENABLED: bool = Field(default=True, description="Enable WebSocket endpoints")
    FEATURE_MAP_GENERATION: bool = Field(default=True, description="Enable map HTML generation")
    FEATURE_ASYNC_OPTIMIZATION: bool = Field(default=False, description="Enable async optimization")

    # =========================
    # Validators
    # =========================
    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        allowed = {"development", "staging", "production", "testing"}
        if v.lower() not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of: {allowed}")
        return v.lower()

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of: {allowed}")
        return v.upper()

    @field_validator("LOG_FORMAT")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format."""
        allowed = {"json", "text"}
        if v.lower() not in allowed:
            raise ValueError(f"LOG_FORMAT must be one of: {allowed}")
        return v.lower()

    # =========================
    # Computed Properties
    # =========================
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins into a list."""
        if not self.CORS_ORIGINS:
            return []
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def valid_api_keys_list(self) -> List[str]:
        """Parse API keys into a list."""
        if not self.VALID_API_KEYS:
            return []
        return [key.strip() for key in self.VALID_API_KEYS.split(",") if key.strip()]

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.ENVIRONMENT == "development"

    def get_bbox(self) -> tuple:
        """Get bounding box as tuple."""
        return (self.BBOX_SOUTH, self.BBOX_NORTH, self.BBOX_WEST, self.BBOX_EAST)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses LRU cache to ensure settings are loaded only once
    and reused across the application.

    Returns:
        Settings instance with environment overrides applied.
    """
    return Settings()


# Convenience function for dependency injection
def get_settings_dependency() -> Settings:
    """FastAPI dependency for settings injection."""
    return get_settings()
