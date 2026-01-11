"""
Unit tests for TrafficSimulator (traffic_sim.py).

Run with: pytest tests/test_traffic_sim.py -v
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph_builder import OSMGraph, SAMPLE_LOCATIONS
from src.traffic_sim import TrafficSimulator


class TestTrafficSimulator:
    """Test cases for TrafficSimulator class."""
    
    @pytest.fixture
    def graph(self):
        """Create OSMGraph instance."""
        return OSMGraph(use_cache=True)
    
    @pytest.fixture
    def simulator(self, graph):
        """Create TrafficSimulator instance."""
        return TrafficSimulator(graph, seed=42)
    
    def test_initialization(self, simulator):
        """Test simulator initializes correctly."""
        assert simulator is not None
        assert simulator.seed == 42
        assert simulator.congestion_level in ['low', 'medium', 'high']
    
    def test_congestion_multipliers(self, simulator):
        """Test congestion multiplier values."""
        assert simulator.CONGESTION_MULTIPLIERS['low'] == 1.0
        assert simulator.CONGESTION_MULTIPLIERS['medium'] == 1.5
        assert simulator.CONGESTION_MULTIPLIERS['high'] == 2.5
    
    def test_congestion_rates(self, simulator):
        """Test congestion rate values."""
        assert simulator.CONGESTION_RATES['low'] == 0.10
        assert simulator.CONGESTION_RATES['medium'] == 0.20
        assert simulator.CONGESTION_RATES['high'] == 0.30
    
    def test_update_congestion_low(self, simulator):
        """Test updating to low congestion."""
        simulator.update_congestion('low')
        assert simulator.congestion_level == 'low'
        assert simulator.get_level_multiplier() == 1.0
    
    def test_update_congestion_medium(self, simulator):
        """Test updating to medium congestion."""
        simulator.update_congestion('medium')
        assert simulator.congestion_level == 'medium'
        assert simulator.get_level_multiplier() == 1.5
    
    def test_update_congestion_high(self, simulator):
        """Test updating to high congestion."""
        simulator.update_congestion('high')
        assert simulator.congestion_level == 'high'
        assert simulator.get_level_multiplier() == 2.5
    
    def test_update_congestion_invalid(self, simulator):
        """Test invalid congestion level raises error."""
        with pytest.raises(ValueError):
            simulator.update_congestion('extreme')
    
    def test_get_edge_congestion_default(self, simulator):
        """Test default edge congestion is 1.0."""
        # For edges not explicitly congested
        congestion = simulator.get_edge_congestion(999999, 999998, 0)
        assert congestion == 1.0
    
    def test_get_dynamic_weights_shape(self, simulator, graph):
        """Test dynamic weights matrix has correct shape."""
        locations = SAMPLE_LOCATIONS[:5]
        dist_matrix = graph.precompute_shortest_paths(locations)
        
        weighted = simulator.get_dynamic_weights(locations, dist_matrix)
        
        assert weighted.shape == dist_matrix.shape
    
    def test_get_dynamic_weights_diagonal(self, simulator, graph):
        """Test dynamic weights has zeros on diagonal."""
        locations = SAMPLE_LOCATIONS[:5]
        dist_matrix = graph.precompute_shortest_paths(locations)
        
        weighted = simulator.get_dynamic_weights(locations, dist_matrix)
        
        for i in range(len(locations)):
            assert weighted[i, i] == 0.0
    
    def test_get_dynamic_weights_increases_with_level(self, graph):
        """Test weighted distances increase with traffic level."""
        locations = SAMPLE_LOCATIONS[:3]
        dist_matrix = graph.precompute_shortest_paths(locations)
        
        sim_low = TrafficSimulator(graph, seed=42, initial_level='low')
        sim_medium = TrafficSimulator(graph, seed=42, initial_level='medium')
        sim_high = TrafficSimulator(graph, seed=42, initial_level='high')
        
        weights_low = sim_low.get_dynamic_weights(locations, dist_matrix)
        weights_medium = sim_medium.get_dynamic_weights(locations, dist_matrix)
        weights_high = sim_high.get_dynamic_weights(locations, dist_matrix)
        
        # Average weights should increase
        avg_low = np.mean(weights_low[weights_low > 0])
        avg_medium = np.mean(weights_medium[weights_medium > 0])
        avg_high = np.mean(weights_high[weights_high > 0])
        
        assert avg_medium > avg_low
        assert avg_high > avg_medium
    
    def test_get_congestion_matrix_shape(self, simulator, graph):
        """Test congestion matrix has correct shape."""
        locations = SAMPLE_LOCATIONS[:5]
        
        congestion = simulator.get_congestion_matrix(locations)
        
        assert congestion.shape == (len(locations), len(locations))
    
    def test_get_congestion_matrix_positive(self, simulator, graph):
        """Test congestion factors are positive."""
        locations = SAMPLE_LOCATIONS[:5]
        
        congestion = simulator.get_congestion_matrix(locations)
        
        assert np.all(congestion > 0)
    
    def test_estimate_travel_time_basic(self, simulator):
        """Test travel time estimation."""
        # 1000m at base speed (8.33 m/s) = ~120 seconds
        time = simulator.estimate_travel_time(1000.0, congestion_factor=1.0)
        
        assert 100 < time < 150  # Approximately 120 seconds
    
    def test_estimate_travel_time_with_congestion(self, simulator):
        """Test travel time increases with congestion."""
        time_low = simulator.estimate_travel_time(1000.0, congestion_factor=1.0)
        time_high = simulator.estimate_travel_time(1000.0, congestion_factor=2.5)
        
        # High congestion should take 2.5x longer
        assert time_high > time_low
        assert abs(time_high / time_low - 2.5) < 0.1
    
    def test_estimate_travel_time_zero_distance(self, simulator):
        """Test zero distance gives zero time."""
        time = simulator.estimate_travel_time(0.0)
        assert time == 0.0
    
    def test_reset(self, simulator):
        """Test reset functionality."""
        # Change state
        simulator.update_congestion('high')
        
        # Reset
        simulator.reset()
        
        # Should be back to initial state with regenerated congestion
        assert simulator.seed == 42
    
    def test_reset_with_new_seed(self, simulator):
        """Test reset with new seed."""
        original_seed = simulator.seed
        
        simulator.reset(seed=123)
        
        assert simulator.seed == 123
        assert simulator.seed != original_seed
    
    def test_deterministic_with_seed(self, graph):
        """Test same seed produces same results."""
        sim1 = TrafficSimulator(graph, seed=42)
        sim1.update_congestion('high')
        weights1 = sim1.get_dynamic_weights(
            SAMPLE_LOCATIONS[:3],
            graph.precompute_shortest_paths(SAMPLE_LOCATIONS[:3])
        )
        
        sim2 = TrafficSimulator(graph, seed=42)
        sim2.update_congestion('high')
        weights2 = sim2.get_dynamic_weights(
            SAMPLE_LOCATIONS[:3],
            graph.precompute_shortest_paths(SAMPLE_LOCATIONS[:3])
        )
        
        np.testing.assert_array_almost_equal(weights1, weights2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
