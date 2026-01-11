"""
Unit tests for OSMGraph (graph_builder.py).

Run with: pytest tests/test_graph_builder.py -v
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph_builder import OSMGraph, SAMPLE_LOCATIONS


class TestOSMGraph:
    """Test cases for OSMGraph class."""
    
    @pytest.fixture
    def graph(self):
        """Create OSMGraph instance for testing."""
        return OSMGraph(use_cache=True)
    
    def test_initialization(self, graph):
        """Test that graph initializes correctly."""
        assert graph is not None
        assert graph.bbox == OSMGraph.DEFAULT_BBOX
        assert graph.graph is not None
    
    def test_bbox_validation(self, graph):
        """Test bounding box validation."""
        # Inside bbox (smaller bbox: 16.50-16.55, 80.62-80.68)
        assert graph.validate_location(16.52, 80.65) is True
        assert graph.validate_location(16.50, 80.62) is True
        assert graph.validate_location(16.55, 80.68) is True

        # Outside bbox
        assert graph.validate_location(17.0, 80.65) is False
        assert graph.validate_location(16.52, 81.0) is False
        assert graph.validate_location(16.4, 80.5) is False
    
    def test_nearest_node(self, graph):
        """Test nearest node finding."""
        lat, lng = SAMPLE_LOCATIONS[0]
        node = graph.get_nearest_node(lat, lng)
        
        assert node is not None
        assert isinstance(node, (int, np.integer))
    
    def test_nearest_node_caching(self, graph):
        """Test that nearest node results are cached."""
        lat, lng = 16.55, 80.65
        
        # First call
        node1 = graph.get_nearest_node(lat, lng)
        
        # Second call should use cache
        node2 = graph.get_nearest_node(lat, lng)
        
        assert node1 == node2
        assert (round(lat, 6), round(lng, 6)) in graph.node_cache
    
    def test_precompute_shortest_paths_shape(self, graph):
        """Test distance matrix has correct shape."""
        locations = SAMPLE_LOCATIONS[:5]
        dist_matrix = graph.precompute_shortest_paths(locations)
        
        n = len(locations)
        assert dist_matrix.shape == (n, n)
    
    def test_precompute_shortest_paths_diagonal(self, graph):
        """Test distance matrix has zeros on diagonal."""
        locations = SAMPLE_LOCATIONS[:5]
        dist_matrix = graph.precompute_shortest_paths(locations)
        
        for i in range(len(locations)):
            assert dist_matrix[i, i] == 0.0
    
    def test_precompute_shortest_paths_symmetric(self, graph):
        """Test distance matrix is approximately symmetric."""
        locations = SAMPLE_LOCATIONS[:5]
        dist_matrix = graph.precompute_shortest_paths(locations)
        
        # Allow some tolerance for directed graphs
        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                # Distances should be similar but not necessarily identical
                ratio = dist_matrix[i, j] / dist_matrix[j, i] if dist_matrix[j, i] > 0 else 1
                assert 0.5 < ratio < 2.0, f"Asymmetry too large at ({i}, {j})"
    
    def test_precompute_shortest_paths_positive(self, graph):
        """Test distance matrix has positive off-diagonal values."""
        locations = SAMPLE_LOCATIONS[:5]
        dist_matrix = graph.precompute_shortest_paths(locations)
        
        for i in range(len(locations)):
            for j in range(len(locations)):
                if i != j:
                    assert dist_matrix[i, j] > 0, f"Non-positive distance at ({i}, {j})"
    
    def test_precompute_shortest_paths_finite(self, graph):
        """Test distance matrix has finite values."""
        locations = SAMPLE_LOCATIONS[:5]
        dist_matrix = graph.precompute_shortest_paths(locations)
        
        assert np.all(np.isfinite(dist_matrix))
    
    def test_get_route_weight(self, graph):
        """Test route weight calculation with congestion."""
        locations = SAMPLE_LOCATIONS[:3]
        dist_matrix = graph.precompute_shortest_paths(locations)
        
        base_weight = graph.get_route_weight(0, 1, dist_matrix, 'low')
        medium_weight = graph.get_route_weight(0, 1, dist_matrix, 'medium')
        high_weight = graph.get_route_weight(0, 1, dist_matrix, 'high')
        
        # Weights should increase with congestion
        assert base_weight > 0
        assert medium_weight > base_weight
        assert high_weight > medium_weight
    
    def test_congestion_multipliers(self, graph):
        """Test congestion multiplier values."""
        assert graph.CONGESTION_MULTIPLIERS['low'] == 1.0
        assert graph.CONGESTION_MULTIPLIERS['medium'] == 1.5
        assert graph.CONGESTION_MULTIPLIERS['high'] == 2.5
    
    def test_haversine_distance(self, graph):
        """Test haversine distance calculation."""
        # Two points approximately 1 km apart
        lat1, lng1 = 16.5063, 80.6480
        lat2, lng2 = 16.5063, 80.6570  # ~1 km east
        
        dist = graph._haversine(lat1, lng1, lat2, lng2)
        
        # Should be approximately 1 km (within 0.5 km tolerance)
        assert 0.5 < dist < 1.5
    
    def test_haversine_same_point(self, graph):
        """Test haversine distance for same point is zero."""
        lat, lng = 16.5063, 80.6480
        dist = graph._haversine(lat, lng, lat, lng)
        assert dist == 0.0


class TestSampleLocations:
    """Test the sample locations data."""
    
    def test_sample_locations_count(self):
        """Test correct number of sample locations."""
        assert len(SAMPLE_LOCATIONS) == 5
    
    def test_sample_locations_in_bbox(self):
        """Test all sample locations are within bbox."""
        graph = OSMGraph(use_cache=True)
        
        for lat, lng in SAMPLE_LOCATIONS:
            assert graph.validate_location(lat, lng), f"Location ({lat}, {lng}) outside bbox"
    
    def test_sample_locations_format(self):
        """Test sample locations are tuples of floats."""
        for loc in SAMPLE_LOCATIONS:
            assert isinstance(loc, tuple)
            assert len(loc) == 2
            assert isinstance(loc[0], float)
            assert isinstance(loc[1], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
