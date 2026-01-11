"""
Load Testing for Quantum Traffic Optimizer.

This module provides Locust load testing scenarios for the API.

Usage:
    # Start Locust web UI
    locust -f tests/performance/locustfile.py --host=http://localhost:8000

    # Run headless with 100 users, spawn rate 10/s, for 5 minutes
    locust -f tests/performance/locustfile.py --host=http://localhost:8000 \
           --users 100 --spawn-rate 10 --run-time 5m --headless

    # Run with HTML report
    locust -f tests/performance/locustfile.py --host=http://localhost:8000 \
           --users 50 --spawn-rate 5 --run-time 2m --headless \
           --html=load_test_report.html
"""

import json
import random
from typing import List, Tuple

from locust import HttpUser, between, events, task
from locust.runners import MasterRunner


# =============================================================================
# Test Data
# =============================================================================

# Vijayawada bounding box
BBOX = {
    "south": 16.50,
    "north": 16.55,
    "west": 80.62,
    "east": 80.68,
}


def random_location() -> Tuple[float, float]:
    """Generate a random location within Vijayawada bounds."""
    lat = random.uniform(BBOX["south"], BBOX["north"])
    lng = random.uniform(BBOX["west"], BBOX["east"])
    return (lat, lng)


def generate_deliveries(count: int) -> List[dict]:
    """Generate random delivery points."""
    return [
        {
            "lat": random.uniform(BBOX["south"], BBOX["north"]),
            "lng": random.uniform(BBOX["west"], BBOX["east"]),
            "priority": random.randint(0, 10),
            "name": f"Delivery {i+1}",
        }
        for i in range(count)
    ]


# =============================================================================
# Locust User Classes
# =============================================================================


class QuickHealthCheckUser(HttpUser):
    """
    User that only performs health checks.

    Used to verify basic connectivity and measure baseline latency.
    """

    weight = 1  # Lower weight = fewer of these users
    wait_time = between(1, 3)

    @task(3)
    def health_check(self):
        """Basic health check."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"Unhealthy status: {data}")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def liveness_probe(self):
        """Kubernetes liveness probe."""
        self.client.get("/health/live")

    @task(1)
    def readiness_probe(self):
        """Kubernetes readiness probe."""
        self.client.get("/health/ready")


class OptimizationUser(HttpUser):
    """
    User that performs route optimization requests.

    This is the main load testing scenario for the optimization API.
    """

    weight = 5  # Higher weight = more of these users
    wait_time = between(2, 5)

    def on_start(self):
        """Setup user state."""
        self.route_ids = []

    @task(10)
    def optimize_small_route(self):
        """Optimize route with 2-5 deliveries (most common case)."""
        current_loc = random_location()
        deliveries = generate_deliveries(random.randint(2, 5))

        payload = {
            "current_loc": list(current_loc),
            "deliveries": deliveries,
            "traffic_level": random.choice(["low", "medium", "high"]),
            "include_map": False,  # Skip map generation for load testing
        }

        with self.client.post(
            "/optimize",
            json=payload,
            catch_response=True,
            name="/optimize [small]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "route_id" in data:
                    self.route_ids.append(data["route_id"])
                    response.success()
                else:
                    response.failure("Missing route_id in response")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Status: {response.status_code}")

    @task(3)
    def optimize_medium_route(self):
        """Optimize route with 6-10 deliveries."""
        current_loc = random_location()
        deliveries = generate_deliveries(random.randint(6, 10))

        payload = {
            "current_loc": list(current_loc),
            "deliveries": deliveries,
            "traffic_level": random.choice(["low", "medium", "high"]),
            "include_map": False,
        }

        with self.client.post(
            "/optimize",
            json=payload,
            catch_response=True,
            name="/optimize [medium]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "route_id" in data:
                    self.route_ids.append(data["route_id"])
                    response.success()
                else:
                    response.failure("Missing route_id")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Status: {response.status_code}")

    @task(1)
    def optimize_large_route(self):
        """Optimize route with 11-20 deliveries (stress test)."""
        current_loc = random_location()
        deliveries = generate_deliveries(random.randint(11, 20))

        payload = {
            "current_loc": list(current_loc),
            "deliveries": deliveries,
            "traffic_level": random.choice(["low", "medium", "high"]),
            "include_map": False,
        }

        with self.client.post(
            "/optimize",
            json=payload,
            catch_response=True,
            name="/optimize [large]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "route_id" in data:
                    self.route_ids.append(data["route_id"])
                    response.success()
                else:
                    response.failure("Missing route_id")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Status: {response.status_code}")

    @task(5)
    def list_routes(self):
        """List cached routes."""
        with self.client.get(
            "/routes",
            params={"skip": 0, "limit": 10},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(2)
    def get_route_map(self):
        """Get route map for a previously created route."""
        if not self.route_ids:
            return

        route_id = random.choice(self.route_ids)
        with self.client.get(
            f"/map/{route_id}",
            catch_response=True,
            name="/map/{route_id}",
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # Route may have been evicted from cache
                self.route_ids.remove(route_id)
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


class MapGenerationUser(HttpUser):
    """
    User that tests map generation (heavy operation).

    Map generation is resource-intensive, so this user type
    is used to stress test that specific functionality.
    """

    weight = 2
    wait_time = between(5, 10)

    @task
    def optimize_with_map(self):
        """Optimize route with map generation enabled."""
        current_loc = random_location()
        deliveries = generate_deliveries(random.randint(3, 7))

        payload = {
            "current_loc": list(current_loc),
            "deliveries": deliveries,
            "traffic_level": random.choice(["low", "medium", "high"]),
            "include_map": True,  # Enable map generation
        }

        with self.client.post(
            "/optimize",
            json=payload,
            catch_response=True,
            name="/optimize [with map]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("map_html"):
                    response.success()
                else:
                    response.failure("Map HTML not generated")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Status: {response.status_code}")


class APIKeyUser(HttpUser):
    """
    User that tests API key authentication.

    Requires API_KEYS_ENABLED=true and valid API keys configured.
    """

    weight = 1
    wait_time = between(3, 6)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set API key header (configure in environment)
        self.api_key = "test-api-key"  # Replace with valid key

    @task
    def optimize_with_api_key(self):
        """Optimize route with API key authentication."""
        current_loc = random_location()
        deliveries = generate_deliveries(random.randint(2, 5))

        payload = {
            "current_loc": list(current_loc),
            "deliveries": deliveries,
            "traffic_level": "medium",
            "include_map": False,
        }

        headers = {"X-API-Key": self.api_key}

        with self.client.post(
            "/optimize",
            json=payload,
            headers=headers,
            catch_response=True,
            name="/optimize [API key]",
        ) as response:
            if response.status_code in [200, 401, 403]:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


# =============================================================================
# Event Hooks
# =============================================================================


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print("=" * 60)
    print("Quantum Traffic Optimizer - Load Test Started")
    print("=" * 60)
    print(f"Target host: {environment.host}")
    print(f"Users: {environment.parsed_options.num_users if hasattr(environment, 'parsed_options') else 'N/A'}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    print("=" * 60)
    print("Load Test Completed")
    print("=" * 60)

    # Print summary statistics
    stats = environment.runner.stats

    print(f"\nTotal Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"95th Percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"99th Percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    print(f"Requests/sec: {stats.total.total_rps:.2f}")

    if stats.total.num_failures > 0:
        failure_rate = (stats.total.num_failures / stats.total.num_requests) * 100
        print(f"Failure Rate: {failure_rate:.2f}%")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Called for each request."""
    # Custom logging or metrics can be added here
    pass


# =============================================================================
# Custom Test Scenarios
# =============================================================================


class SpikeTestUser(HttpUser):
    """
    User for spike testing - sends requests in bursts.

    Used to test how the system handles sudden load increases.
    """

    weight = 0  # Disabled by default, enable by setting weight > 0
    wait_time = between(0.1, 0.5)  # Very short wait time

    @task
    def spike_optimize(self):
        """Rapid-fire optimization requests."""
        current_loc = random_location()
        deliveries = generate_deliveries(3)

        payload = {
            "current_loc": list(current_loc),
            "deliveries": deliveries,
            "traffic_level": "medium",
            "include_map": False,
        }

        self.client.post("/optimize", json=payload, name="/optimize [spike]")


class StressTestUser(HttpUser):
    """
    User for stress testing - maximum payload sizes.

    Used to find breaking points in the system.
    """

    weight = 0  # Disabled by default
    wait_time = between(1, 2)

    @task
    def stress_optimize(self):
        """Stress test with maximum deliveries."""
        current_loc = random_location()
        deliveries = generate_deliveries(20)  # Maximum allowed

        payload = {
            "current_loc": list(current_loc),
            "deliveries": deliveries,
            "traffic_level": "high",
            "include_map": True,
        }

        self.client.post("/optimize", json=payload, name="/optimize [stress]")
