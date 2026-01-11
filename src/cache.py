"""
Redis Caching Layer for Quantum Traffic Optimizer.

This module provides caching functionality using Redis for:
- Route data caching
- Graph caching
- Rate limiting support
- Session storage
"""

import json
import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional, TypeVar, Union

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

# Type variable for generic cache operations
T = TypeVar("T")

# Try to import redis, but make it optional
try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None


class CacheManager:
    """
    Redis cache manager with async support.

    Provides high-level caching operations with automatic
    serialization and TTL management.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize cache manager.

        Args:
            settings: Application settings. Uses default if not provided.
        """
        self.settings = settings or get_settings()
        self._client: Optional[Redis] = None
        self._initialized = False

    @property
    def is_available(self) -> bool:
        """Check if Redis is available and enabled."""
        return REDIS_AVAILABLE and self.settings.REDIS_ENABLED

    async def connect(self) -> None:
        """
        Connect to Redis server.

        Raises:
            RuntimeError: If Redis is not available or connection fails.
        """
        if not self.is_available:
            logger.info("Redis is disabled or not available")
            return

        if self._client is not None:
            return

        try:
            self._client = redis.from_url(
                self.settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self._client.ping()
            self._initialized = True
            logger.info("Connected to Redis successfully")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._client = None
            raise

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._initialized = False
            logger.info("Disconnected from Redis")

    async def health_check(self) -> bool:
        """
        Check Redis connectivity.

        Returns:
            True if Redis is healthy, False otherwise.
        """
        if not self.is_available:
            return True  # No Redis, so it's "healthy"

        if self._client is None:
            return False

        try:
            await self._client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    def _make_key(self, key: str) -> str:
        """Create a prefixed cache key."""
        return f"{self.settings.REDIS_PREFIX}{key}"

    # =========================
    # Basic Operations
    # =========================

    async def get(self, key: str) -> Optional[str]:
        """
        Get a value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.
        """
        if not self._initialized:
            return None

        try:
            return await self._client.get(self._make_key(key))
        except Exception as e:
            logger.warning(f"Cache get failed for {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set a value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds. Uses default if not provided.

        Returns:
            True if successful, False otherwise.
        """
        if not self._initialized:
            return False

        try:
            ttl = ttl or self.settings.CACHE_TTL
            await self._client.setex(
                self._make_key(key),
                ttl,
                value
            )
            return True
        except Exception as e:
            logger.warning(f"Cache set failed for {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete a value from cache.

        Args:
            key: Cache key.

        Returns:
            True if key was deleted, False otherwise.
        """
        if not self._initialized:
            return False

        try:
            result = await self._client.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.warning(f"Cache delete failed for {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        if not self._initialized:
            return False

        try:
            return await self._client.exists(self._make_key(key)) > 0
        except Exception as e:
            logger.warning(f"Cache exists check failed for {key}: {e}")
            return False

    # =========================
    # JSON Operations
    # =========================

    async def get_json(self, key: str) -> Optional[Any]:
        """
        Get a JSON value from cache.

        Args:
            key: Cache key.

        Returns:
            Deserialized JSON value or None.
        """
        value = await self.get(key)
        if value is None:
            return None

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in cache for key {key}")
            return None

    async def set_json(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set a JSON value in cache.

        Args:
            key: Cache key.
            value: Value to serialize and cache.
            ttl: Time-to-live in seconds.

        Returns:
            True if successful, False otherwise.
        """
        try:
            json_value = json.dumps(value, default=str)
            return await self.set(key, json_value, ttl)
        except (TypeError, ValueError) as e:
            logger.warning(f"JSON serialization failed for {key}: {e}")
            return False

    # =========================
    # Route Caching
    # =========================

    async def cache_route(
        self,
        route_id: str,
        route_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache route optimization result.

        Args:
            route_id: Route identifier.
            route_data: Route data dictionary.
            ttl: Cache TTL in seconds.

        Returns:
            True if cached successfully.
        """
        key = f"route:{route_id}"
        return await self.set_json(key, route_data, ttl)

    async def get_cached_route(self, route_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached route.

        Args:
            route_id: Route identifier.

        Returns:
            Route data dictionary or None.
        """
        key = f"route:{route_id}"
        return await self.get_json(key)

    async def delete_cached_route(self, route_id: str) -> bool:
        """
        Delete cached route.

        Args:
            route_id: Route identifier.

        Returns:
            True if deleted.
        """
        key = f"route:{route_id}"
        return await self.delete(key)

    async def list_cached_routes(self, pattern: str = "route:*") -> List[str]:
        """
        List all cached route IDs.

        Args:
            pattern: Key pattern to match.

        Returns:
            List of route IDs.
        """
        if not self._initialized:
            return []

        try:
            full_pattern = f"{self.settings.REDIS_PREFIX}{pattern}"
            keys = await self._client.keys(full_pattern)
            # Extract route IDs from keys
            prefix_len = len(f"{self.settings.REDIS_PREFIX}route:")
            return [k[prefix_len:] for k in keys]
        except Exception as e:
            logger.warning(f"Failed to list cached routes: {e}")
            return []

    # =========================
    # Graph Caching
    # =========================

    async def cache_graph(
        self,
        bbox: tuple,
        graph_data: bytes,
        ttl: int = 86400  # 24 hours
    ) -> bool:
        """
        Cache OSM graph data.

        Args:
            bbox: Bounding box tuple.
            graph_data: Serialized graph bytes.
            ttl: Cache TTL in seconds.

        Returns:
            True if cached successfully.
        """
        if not self._initialized:
            return False

        key = f"graph:{bbox}"
        try:
            # Use raw bytes storage for graph data
            await self._client.setex(
                self._make_key(key),
                ttl,
                graph_data.hex()  # Convert bytes to hex string
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to cache graph: {e}")
            return False

    async def get_cached_graph(self, bbox: tuple) -> Optional[bytes]:
        """
        Retrieve cached graph.

        Args:
            bbox: Bounding box tuple.

        Returns:
            Graph bytes or None.
        """
        key = f"graph:{bbox}"
        value = await self.get(key)
        if value is None:
            return None

        try:
            return bytes.fromhex(value)
        except ValueError:
            return None

    # =========================
    # Rate Limiting Support
    # =========================

    async def increment_rate_limit(
        self,
        key: str,
        window_seconds: int
    ) -> int:
        """
        Increment rate limit counter.

        Args:
            key: Rate limit key (e.g., IP address).
            window_seconds: Time window in seconds.

        Returns:
            Current count within the window.
        """
        if not self._initialized:
            return 0

        full_key = self._make_key(f"rate:{key}")
        try:
            pipe = self._client.pipeline()
            pipe.incr(full_key)
            pipe.expire(full_key, window_seconds)
            results = await pipe.execute()
            return results[0]
        except Exception as e:
            logger.warning(f"Rate limit increment failed: {e}")
            return 0

    async def get_rate_limit_count(self, key: str) -> int:
        """
        Get current rate limit count.

        Args:
            key: Rate limit key.

        Returns:
            Current count.
        """
        full_key = self._make_key(f"rate:{key}")
        value = await self.get(f"rate:{key}")
        if value is None:
            return 0
        try:
            return int(value)
        except ValueError:
            return 0

    # =========================
    # Distributed Lock
    # =========================

    async def acquire_lock(
        self,
        name: str,
        timeout: int = 10,
        blocking: bool = True,
        blocking_timeout: int = 5
    ) -> Optional[str]:
        """
        Acquire a distributed lock.

        Args:
            name: Lock name.
            timeout: Lock auto-release timeout in seconds.
            blocking: Whether to wait for the lock.
            blocking_timeout: Maximum time to wait for lock.

        Returns:
            Lock token if acquired, None otherwise.
        """
        if not self._initialized:
            return "fake-lock-token"  # Allow operation without Redis

        import secrets
        lock_key = self._make_key(f"lock:{name}")
        token = secrets.token_urlsafe(16)

        try:
            if blocking:
                # Simple blocking implementation
                import asyncio
                start = asyncio.get_event_loop().time()
                while asyncio.get_event_loop().time() - start < blocking_timeout:
                    acquired = await self._client.set(
                        lock_key,
                        token,
                        nx=True,
                        ex=timeout
                    )
                    if acquired:
                        return token
                    await asyncio.sleep(0.1)
                return None
            else:
                acquired = await self._client.set(
                    lock_key,
                    token,
                    nx=True,
                    ex=timeout
                )
                return token if acquired else None

        except Exception as e:
            logger.warning(f"Failed to acquire lock {name}: {e}")
            return None

    async def release_lock(self, name: str, token: str) -> bool:
        """
        Release a distributed lock.

        Args:
            name: Lock name.
            token: Lock token from acquire_lock.

        Returns:
            True if lock was released.
        """
        if not self._initialized:
            return True

        lock_key = self._make_key(f"lock:{name}")

        # Lua script for atomic check-and-delete
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        try:
            result = await self._client.eval(script, 1, lock_key, token)
            return result == 1
        except Exception as e:
            logger.warning(f"Failed to release lock {name}: {e}")
            return False


# Global cache manager instance
cache_manager = CacheManager()


async def get_cache() -> CacheManager:
    """
    FastAPI dependency for cache manager.

    Returns:
        CacheManager instance.
    """
    return cache_manager


async def init_cache() -> None:
    """Initialize cache on application startup."""
    if cache_manager.is_available:
        await cache_manager.connect()


async def close_cache() -> None:
    """Close cache on application shutdown."""
    await cache_manager.disconnect()


# =========================
# In-Memory Fallback Cache
# =========================

class InMemoryCache:
    """
    Simple in-memory cache fallback when Redis is not available.

    WARNING: This cache is not distributed and will be lost on restart.
    Use only for development or single-instance deployments.
    """

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry)
        self._max_size = max_size

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        import time
        now = time.time()
        expired = [k for k, (_, exp) in self._cache.items() if exp and exp < now]
        for k in expired:
            del self._cache[k]

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        if len(self._cache) >= self._max_size:
            # Remove oldest 20%
            to_remove = list(self._cache.keys())[:self._max_size // 5]
            for k in to_remove:
                del self._cache[k]

    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        import time
        self._cleanup_expired()
        if key not in self._cache:
            return None
        value, expiry = self._cache[key]
        if expiry and expiry < time.time():
            del self._cache[key]
            return None
        return value

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        import time
        self._cleanup_expired()
        self._evict_if_needed()
        expiry = time.time() + ttl if ttl else None
        self._cache[key] = (value, expiry)

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()


# Global in-memory cache fallback
memory_cache = InMemoryCache()
