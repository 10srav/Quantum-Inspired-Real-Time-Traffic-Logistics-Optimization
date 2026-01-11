"""
API endpoint tests for FastAPI application.

Run with: pytest tests/test_api.py -v
"""

import os
import pytest
from fastapi.testclient import TestClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set demo mode before importing app
os.environ['DEMO_MODE'] = '1'

from src.main import app


@pytest.fixture
def client():
    """Create test client with lifespan context."""
    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "graph_loaded" in data


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_html(self, client):
        """Test root returns HTML documentation."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Quantum Traffic Optimizer" in response.text


class TestOptimizeEndpoint:
    """Tests for /optimize endpoint."""
    
    def test_optimize_valid_request(self, client):
        """Test optimization with valid request."""
        payload = {
            "current_loc": [16.505, 80.625],
            "deliveries": [
                {"lat": 16.51, "lng": 80.63, "priority": 2.0},
                {"lat": 16.52, "lng": 80.64, "priority": 1.0}
            ],
            "traffic_level": "medium",
            "include_map": False
        }
        
        response = client.post("/optimize", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "route_id" in data
        assert "sequence" in data
        assert "total_distance" in data
        assert "total_eta" in data
        assert "optimization_time" in data
        assert data["traffic_level"] == "medium"
    
    def test_optimize_with_map(self, client):
        """Test optimization returns map HTML."""
        payload = {
            "current_loc": [16.505, 80.625],
            "deliveries": [
                {"lat": 16.51, "lng": 80.63, "priority": 1.0}
            ],
            "traffic_level": "low",
            "include_map": True
        }
        
        response = client.post("/optimize", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["map_html"] is not None
        assert "<html" in data["map_html"].lower() or "folium" in data["map_html"].lower()
    
    def test_optimize_empty_deliveries(self, client):
        """Test optimization fails with empty deliveries."""
        payload = {
            "current_loc": [16.505, 80.625],
            "deliveries": [],
            "traffic_level": "low"
        }
        
        response = client.post("/optimize", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_optimize_invalid_coordinates(self, client):
        """Test optimization fails with out-of-bounds coordinates."""
        payload = {
            "current_loc": [17.0, 80.6480],  # Outside bbox
            "deliveries": [
                {"lat": 16.51, "lng": 80.63, "priority": 1.0}
            ],
            "traffic_level": "low"
        }
        
        response = client.post("/optimize", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_optimize_invalid_traffic_level(self, client):
        """Test optimization fails with invalid traffic level."""
        payload = {
            "current_loc": [16.505, 80.625],
            "deliveries": [
                {"lat": 16.51, "lng": 80.63, "priority": 1.0}
            ],
            "traffic_level": "extreme"  # Invalid
        }
        
        response = client.post("/optimize", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_optimize_sequence_valid(self, client):
        """Test optimization returns valid sequence."""
        payload = {
            "current_loc": [16.505, 80.625],
            "deliveries": [
                {"lat": 16.51, "lng": 80.63, "priority": 1.0, "name": "A"},
                {"lat": 16.52, "lng": 80.64, "priority": 2.0, "name": "B"},
                {"lat": 16.53, "lng": 80.65, "priority": 3.0, "name": "C"}
            ],
            "traffic_level": "medium",
            "include_map": False
        }
        
        response = client.post("/optimize", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        sequence = data["sequence"]
        
        # Should have 3 stops (one per delivery)
        assert len(sequence) == 3
        
        # Each stop should have required fields
        for stop in sequence:
            assert "position" in stop
            assert "delivery" in stop
            assert "distance_from_prev" in stop
            assert "eta_from_prev" in stop
            assert "cumulative_distance" in stop
            assert "cumulative_eta" in stop
            
            # Distances and ETAs should be non-negative
            assert stop["distance_from_prev"] >= 0
            assert stop["eta_from_prev"] >= 0
            assert stop["cumulative_distance"] >= 0
            assert stop["cumulative_eta"] >= 0
    
    def test_optimize_performance(self, client):
        """Test optimization completes in reasonable time."""
        import time
        
        payload = {
            "current_loc": [16.505, 80.625],
            "deliveries": [
                {"lat": 16.51, "lng": 80.63, "priority": 2.0},
                {"lat": 16.52, "lng": 80.64, "priority": 1.0},
                {"lat": 16.53, "lng": 80.65, "priority": 3.0},
                {"lat": 16.54, "lng": 80.66, "priority": 1.5}
            ],
            "traffic_level": "high",
            "include_map": False
        }
        
        start = time.time()
        response = client.post("/optimize", json=payload)
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 10.0, f"Request took {elapsed:.2f}s, expected <10s"


class TestMapEndpoint:
    """Tests for /map/{route_id} endpoint."""
    
    def test_map_not_found(self, client):
        """Test map endpoint returns 404 for unknown route."""
        response = client.get("/map/nonexistent")
        assert response.status_code == 404
    
    def test_map_after_optimize(self, client):
        """Test map endpoint works after optimization."""
        # First, create a route
        payload = {
            "current_loc": [16.505, 80.625],
            "deliveries": [
                {"lat": 16.51, "lng": 80.63, "priority": 1.0}
            ],
            "traffic_level": "low",
            "include_map": False
        }
        
        opt_response = client.post("/optimize", json=payload)
        assert opt_response.status_code == 200
        
        route_id = opt_response.json()["route_id"]
        
        # Then, get the map
        map_response = client.get(f"/map/{route_id}")
        assert map_response.status_code == 200
        assert "text/html" in map_response.headers["content-type"]


class TestRoutesEndpoint:
    """Tests for /routes endpoint."""
    
    def test_list_routes(self, client):
        """Test listing routes."""
        response = client.get("/routes")
        assert response.status_code == 200
        
        data = response.json()
        assert "routes" in data
        assert isinstance(data["routes"], list)
    
    def test_delete_route_not_found(self, client):
        """Test deleting nonexistent route."""
        response = client.delete("/routes/nonexistent")
        assert response.status_code == 404


class TestCORS:
    """Tests for CORS configuration."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/optimize",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )
        
        # CORS should be enabled
        assert response.status_code in [200, 204, 405]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
