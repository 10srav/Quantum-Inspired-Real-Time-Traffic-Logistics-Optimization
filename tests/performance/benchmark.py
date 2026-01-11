"""
Performance Benchmarking for Quantum Traffic Optimizer.

This module provides benchmarks for optimization algorithms
and API performance testing.

Usage:
    python -m tests.performance.benchmark
"""

import asyncio
import statistics
import time
from typing import List, Tuple

import httpx


# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = "http://localhost:8000"

# Vijayawada bounds
BBOX = {
    "south": 16.50,
    "north": 16.55,
    "west": 80.62,
    "east": 80.68,
}


# =============================================================================
# Benchmark Utilities
# =============================================================================


def random_location() -> Tuple[float, float]:
    """Generate a random location within bounds."""
    import random

    lat = random.uniform(BBOX["south"], BBOX["north"])
    lng = random.uniform(BBOX["west"], BBOX["east"])
    return (lat, lng)


def generate_deliveries(count: int) -> List[dict]:
    """Generate random delivery points."""
    import random

    return [
        {
            "lat": random.uniform(BBOX["south"], BBOX["north"]),
            "lng": random.uniform(BBOX["west"], BBOX["east"]),
            "priority": random.randint(0, 10),
        }
        for _ in range(count)
    ]


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str, times: List[float]):
        self.name = name
        self.times = times
        self.mean = statistics.mean(times)
        self.median = statistics.median(times)
        self.stdev = statistics.stdev(times) if len(times) > 1 else 0
        self.min = min(times)
        self.max = max(times)
        self.p95 = sorted(times)[int(len(times) * 0.95)] if times else 0
        self.p99 = sorted(times)[int(len(times) * 0.99)] if times else 0

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Count: {len(self.times)}\n"
            f"  Mean: {self.mean:.4f}s\n"
            f"  Median: {self.median:.4f}s\n"
            f"  Std Dev: {self.stdev:.4f}s\n"
            f"  Min: {self.min:.4f}s\n"
            f"  Max: {self.max:.4f}s\n"
            f"  P95: {self.p95:.4f}s\n"
            f"  P99: {self.p99:.4f}s"
        )


# =============================================================================
# Benchmarks
# =============================================================================


async def benchmark_optimization(
    client: httpx.AsyncClient,
    n_deliveries: int,
    iterations: int = 10,
    include_map: bool = False,
) -> BenchmarkResult:
    """
    Benchmark optimization endpoint.

    Args:
        client: HTTP client
        n_deliveries: Number of deliveries
        iterations: Number of iterations
        include_map: Include map generation

    Returns:
        BenchmarkResult with timing data
    """
    times = []

    for i in range(iterations):
        current_loc = random_location()
        deliveries = generate_deliveries(n_deliveries)

        payload = {
            "current_loc": list(current_loc),
            "deliveries": deliveries,
            "traffic_level": "medium",
            "include_map": include_map,
        }

        start = time.perf_counter()
        response = await client.post("/optimize", json=payload)
        elapsed = time.perf_counter() - start

        if response.status_code == 200:
            times.append(elapsed)
        else:
            print(f"  Warning: Request {i+1} failed with status {response.status_code}")

    name = f"Optimization ({n_deliveries} deliveries{'+ map' if include_map else ''})"
    return BenchmarkResult(name, times)


async def benchmark_health_check(
    client: httpx.AsyncClient,
    iterations: int = 100,
) -> BenchmarkResult:
    """Benchmark health check endpoint."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        response = await client.get("/health")
        elapsed = time.perf_counter() - start

        if response.status_code == 200:
            times.append(elapsed)

    return BenchmarkResult("Health Check", times)


async def benchmark_concurrent_requests(
    client: httpx.AsyncClient,
    concurrency: int = 10,
    requests_per_client: int = 5,
) -> BenchmarkResult:
    """
    Benchmark concurrent optimization requests.

    Args:
        client: HTTP client
        concurrency: Number of concurrent requests
        requests_per_client: Requests per concurrent client

    Returns:
        BenchmarkResult with timing data
    """

    async def single_request():
        current_loc = random_location()
        deliveries = generate_deliveries(5)

        payload = {
            "current_loc": list(current_loc),
            "deliveries": deliveries,
            "traffic_level": "medium",
            "include_map": False,
        }

        start = time.perf_counter()
        response = await client.post("/optimize", json=payload)
        elapsed = time.perf_counter() - start

        return elapsed if response.status_code == 200 else None

    # Run concurrent requests
    all_times = []
    for _ in range(requests_per_client):
        tasks = [single_request() for _ in range(concurrency)]
        results = await asyncio.gather(*tasks)
        all_times.extend([t for t in results if t is not None])

    return BenchmarkResult(f"Concurrent ({concurrency} clients)", all_times)


async def run_benchmarks():
    """Run all benchmarks."""
    print("=" * 60)
    print("Quantum Traffic Optimizer - Performance Benchmarks")
    print("=" * 60)
    print(f"API URL: {API_BASE_URL}")
    print()

    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        # Check if API is available
        try:
            response = await client.get("/health")
            if response.status_code != 200:
                print("Error: API is not healthy")
                return
        except Exception as e:
            print(f"Error: Could not connect to API: {e}")
            return

        print("Running benchmarks...\n")

        # Health check benchmark
        result = await benchmark_health_check(client, iterations=50)
        print(result)
        print()

        # Optimization benchmarks with different delivery counts
        for n_deliveries in [3, 5, 10, 15, 20]:
            result = await benchmark_optimization(client, n_deliveries, iterations=10)
            print(result)
            print()

        # Optimization with map generation
        result = await benchmark_optimization(client, 5, iterations=5, include_map=True)
        print(result)
        print()

        # Concurrent request benchmark
        for concurrency in [5, 10, 20]:
            result = await benchmark_concurrent_requests(client, concurrency=concurrency)
            print(result)
            print()

    print("=" * 60)
    print("Benchmarks Complete")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
