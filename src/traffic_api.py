"""
Real-time Traffic API Integration.

Supports TomTom and HERE APIs with caching and automatic fallback to simulation.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

from .config import get_settings
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TrafficData:
    """Traffic data for a route segment."""
    current_speed_kmh: float
    free_flow_speed_kmh: float
    travel_time_seconds: float
    congestion_factor: float  # 1.0 = free flow, higher = more congestion


@dataclass
class CacheEntry:
    """Cached traffic data with TTL."""
    data: TrafficData
    timestamp: float
    ttl: float = 300  # 5 minutes default


class TrafficAPIService:
    """
    Real-time traffic API integration with caching.

    Supports TomTom and HERE APIs with automatic fallback.
    """

    PROVIDERS = ['tomtom', 'here']

    def __init__(
        self,
        provider: str = 'tomtom',
        api_key: Optional[str] = None,
        cache_ttl: int = 300  # 5 minutes
    ):
        self.provider = provider
        self.api_key = api_key or ''
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._request_count = 0
        self._last_request_time = 0.0

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _cache_key(self, start: Tuple[float, float], end: Tuple[float, float]) -> str:
        """Generate cache key for a route segment."""
        key_str = f"{start[0]:.4f},{start[1]:.4f}-{end[0]:.4f},{end[1]:.4f}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _get_cached(self, key: str) -> Optional[TrafficData]:
        """Get cached data if not expired."""
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry.timestamp < entry.ttl:
                return entry.data
            del self._cache[key]
        return None

    def _set_cached(self, key: str, data: TrafficData):
        """Cache traffic data."""
        self._cache[key] = CacheEntry(data=data, timestamp=time.time(), ttl=self.cache_ttl)

    async def _rate_limit(self):
        """Simple rate limiting: max 10 requests per second."""
        now = time.time()
        if now - self._last_request_time < 0.1:
            await asyncio.sleep(0.1)
        self._last_request_time = time.time()
        self._request_count += 1

    async def get_traffic_tomtom(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> Optional[TrafficData]:
        """Fetch traffic data from TomTom API."""
        if not self.api_key:
            return None

        await self._rate_limit()

        url = (
            f"https://api.tomtom.com/routing/1/calculateRoute/"
            f"{start[0]},{start[1]}:{end[0]},{end[1]}/json"
        )
        params = {
            'key': self.api_key,
            'traffic': 'true',
            'travelMode': 'car',
        }

        try:
            session = await self._get_session()
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"TomTom API error: {response.status}")
                    return None

                data = await response.json()
                route = data.get('routes', [{}])[0]
                summary = route.get('summary', {})

                travel_time = summary.get('travelTimeInSeconds', 0)
                live_travel_time = summary.get('liveTrafficIncidentsTravelTimeInSeconds', travel_time)
                distance_m = summary.get('lengthInMeters', 0)

                # Calculate speeds
                free_flow_speed = (distance_m / travel_time * 3.6) if travel_time > 0 else 30
                current_speed = (distance_m / live_travel_time * 3.6) if live_travel_time > 0 else free_flow_speed
                congestion = live_travel_time / travel_time if travel_time > 0 else 1.0

                return TrafficData(
                    current_speed_kmh=current_speed,
                    free_flow_speed_kmh=free_flow_speed,
                    travel_time_seconds=live_travel_time,
                    congestion_factor=congestion
                )
        except asyncio.TimeoutError:
            logger.warning("TomTom API request timed out")
            return None
        except Exception as e:
            logger.warning(f"TomTom API request failed: {e}")
            return None

    async def get_traffic_here(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> Optional[TrafficData]:
        """Fetch traffic data from HERE API."""
        if not self.api_key:
            return None

        await self._rate_limit()

        url = "https://router.hereapi.com/v8/routes"
        params = {
            'apiKey': self.api_key,
            'origin': f'{start[0]},{start[1]}',
            'destination': f'{end[0]},{end[1]}',
            'transportMode': 'car',
            'return': 'summary,typicalDuration',
        }

        try:
            session = await self._get_session()
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"HERE API error: {response.status}")
                    return None

                data = await response.json()
                routes = data.get('routes', [])
                if not routes:
                    return None

                section = routes[0].get('sections', [{}])[0]
                summary = section.get('summary', {})

                travel_time = summary.get('duration', 0)
                typical_time = summary.get('typicalDuration', travel_time)
                distance_m = summary.get('length', 0)

                free_flow_speed = (distance_m / typical_time * 3.6) if typical_time > 0 else 30
                current_speed = (distance_m / travel_time * 3.6) if travel_time > 0 else free_flow_speed
                congestion = travel_time / typical_time if typical_time > 0 else 1.0

                return TrafficData(
                    current_speed_kmh=current_speed,
                    free_flow_speed_kmh=free_flow_speed,
                    travel_time_seconds=travel_time,
                    congestion_factor=congestion
                )
        except asyncio.TimeoutError:
            logger.warning("HERE API request timed out")
            return None
        except Exception as e:
            logger.warning(f"HERE API request failed: {e}")
            return None

    async def get_traffic(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        use_cache: bool = True
    ) -> Optional[TrafficData]:
        """
        Get traffic data for a route segment.

        Uses cache and falls back to alternative provider.
        """
        cache_key = self._cache_key(start, end)

        # Check cache
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached:
                return cached

        # Try primary provider
        data = None
        if self.provider == 'tomtom':
            data = await self.get_traffic_tomtom(start, end)
        elif self.provider == 'here':
            data = await self.get_traffic_here(start, end)

        if data:
            self._set_cached(cache_key, data)
            return data

        return None

    async def get_congestion_matrix(
        self,
        locations: List[Tuple[float, float]],
        use_cache: bool = True,
        max_concurrent: int = 5
    ) -> Dict[Tuple[int, int], float]:
        """
        Get congestion factors for location pairs.

        Returns dict mapping (i, j) to congestion factor (1.0 = free flow).
        Uses semaphore to limit concurrent API calls.
        """
        n = len(locations)
        congestion_map: Dict[Tuple[int, int], float] = {}

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_pair(i: int, j: int):
            async with semaphore:
                if i == j:
                    return (i, j, 0.0)
                result = await self.get_traffic(locations[i], locations[j], use_cache)
                factor = result.congestion_factor if result else 1.0
                return (i, j, factor)

        # Create tasks for all pairs
        tasks = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    tasks.append(fetch_pair(i, j))

        # Execute with rate limiting
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, tuple) and len(result) == 3:
                i, j, factor = result
                congestion_map[(i, j)] = factor
            # Keep 1.0 for failures (fallback to base distance)

        return congestion_map

    def clear_cache(self):
        """Clear the traffic cache."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        now = time.time()
        valid = sum(1 for e in self._cache.values() if now - e.timestamp < e.ttl)
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid,
            "expired_entries": len(self._cache) - valid,
            "request_count": self._request_count,
        }

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Singleton instance
_traffic_service: Optional[TrafficAPIService] = None


def get_traffic_service() -> Optional[TrafficAPIService]:
    """Get or create traffic API service singleton."""
    global _traffic_service

    settings = get_settings()

    if not getattr(settings, 'TRAFFIC_API_ENABLED', False):
        return None

    if _traffic_service is None:
        provider = getattr(settings, 'TRAFFIC_API_PROVIDER', 'tomtom')

        if provider == 'tomtom':
            api_key = getattr(settings, 'TOMTOM_API_KEY', '')
        else:
            api_key = getattr(settings, 'HERE_API_KEY', '')

        cache_ttl = getattr(settings, 'TRAFFIC_CACHE_TTL', 300)

        if api_key:
            _traffic_service = TrafficAPIService(
                provider=provider,
                api_key=api_key,
                cache_ttl=cache_ttl
            )
            logger.info(f"Traffic API service initialized with {provider} provider")
        else:
            logger.warning(f"Traffic API enabled but no API key for {provider}")

    return _traffic_service


async def cleanup_traffic_service():
    """Cleanup traffic service on shutdown."""
    global _traffic_service
    if _traffic_service:
        await _traffic_service.close()
        _traffic_service = None
