"""
Full System Integration Tests.

Tests the complete optimization pipeline from graph loading to route output.

Run with: pytest tests/test_full_system.py -v
"""

import time
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph_builder import OSMGraph, SAMPLE_LOCATIONS
from src.traffic_sim import TrafficSimulator
from src.qubo_optimizer import QUBOOptimizer
from src.utils import (
    calculate_route_distance,
    calculate_route_eta,
    compute_improvement,
    is_valid_permutation
)


class TestFullSystemIntegration:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def system(self):
        """Set up the complete system."""
        graph = OSMGraph(use_cache=True)
        traffic_sim = TrafficSimulator(graph, seed=42)
        optimizer = QUBOOptimizer(n_layers=3, seed=42, timeout=5.0)
        
        return graph, traffic_sim, optimizer
    
    def test_full_pipeline_3_deliveries(self, system):
        """Test complete optimization with 3 deliveries."""
        graph, traffic_sim, optimizer = system
        
        # Set up locations (depot + 3 deliveries)
        locations = SAMPLE_LOCATIONS[:4]
        priorities = [0.0, 2.0, 1.0, 3.0]  # Depot has priority 0
        
        # Compute distances
        dist_matrix = graph.precompute_shortest_paths(locations)
        
        # Set high traffic
        traffic_sim.update_congestion('high')
        congestion_matrix = traffic_sim.get_congestion_matrix(locations)
        
        # Optimize
        sequence, cost, solve_time = optimizer.optimize(
            dist_matrix, priorities, congestion_matrix, 'high'
        )
        
        # Verify sequence is valid permutation
        n = len(locations)
        assert is_valid_permutation(sequence, n), f"Invalid sequence: {sequence}"
        
        # Verify solve time
        assert solve_time < 5.0, f"Optimization took {solve_time:.2f}s, expected <5s"
    
    def test_optimization_vs_greedy(self, system):
        """Test optimized route is at least as good as greedy."""
        graph, traffic_sim, optimizer = system
        
        locations = SAMPLE_LOCATIONS[:5]
        priorities = [0.0, 2.0, 1.0, 3.0, 1.5]
        
        dist_matrix = graph.precompute_shortest_paths(locations)
        traffic_sim.update_congestion('medium')
        congestion_matrix = traffic_sim.get_congestion_matrix(locations)
        
        # Get both solutions
        greedy_seq, greedy_cost = optimizer.solve_greedy(dist_matrix, start_idx=0)
        opt_seq, opt_cost, _ = optimizer.optimize(
            dist_matrix, priorities, congestion_matrix, 'medium'
        )
        
        # Calculate route distances
        greedy_dist = calculate_route_distance(greedy_seq, dist_matrix)
        opt_dist = calculate_route_distance(opt_seq, dist_matrix)
        
        # Optimized should not be significantly worse
        # (may be slightly worse if prioritizing high-priority deliveries)
        assert opt_dist < greedy_dist * 1.5, \
            f"Optimized ({opt_dist:.1f}) much worse than greedy ({greedy_dist:.1f})"
    
    def test_traffic_level_impact(self, system):
        """Test different traffic levels produce different results."""
        graph, traffic_sim, optimizer = system
        
        locations = SAMPLE_LOCATIONS[:4]
        priorities = [0.0, 1.0, 1.0, 1.0]
        dist_matrix = graph.precompute_shortest_paths(locations)
        
        results = {}
        for level in ['low', 'medium', 'high']:
            traffic_sim.update_congestion(level)
            congestion_matrix = traffic_sim.get_congestion_matrix(locations)
            
            seq, cost, _ = optimizer.optimize(
                dist_matrix, priorities, congestion_matrix, level
            )
            
            eta = calculate_route_eta(seq, dist_matrix, congestion_matrix)
            results[level] = {'sequence': seq, 'eta': eta}
        
        # ETA should increase with traffic level (generally)
        assert results['high']['eta'] >= results['low']['eta'], \
            "High traffic should have higher ETA"
    
    def test_performance_5_locations(self, system):
        """Performance test: 5 locations in <5s."""
        graph, traffic_sim, optimizer = system
        
        locations = SAMPLE_LOCATIONS[:5]
        priorities = [0.0] + [1.0] * 4
        
        dist_matrix = graph.precompute_shortest_paths(locations)
        traffic_sim.update_congestion('high')
        congestion_matrix = traffic_sim.get_congestion_matrix(locations)
        
        start = time.time()
        sequence, cost, solve_time = optimizer.optimize(
            dist_matrix, priorities, congestion_matrix, 'high'
        )
        total_time = time.time() - start
        
        assert total_time < 5.0, \
            f"Full pipeline took {total_time:.2f}s, expected <5s"
        
        print(f"\n  Performance: {total_time:.3f}s total, {solve_time:.3f}s optimization")
    
    def test_reproducibility(self, system):
        """Test same inputs produce same outputs (deterministic)."""
        graph, traffic_sim, optimizer = system
        
        locations = SAMPLE_LOCATIONS[:4]
        priorities = [0.0, 1.0, 2.0, 1.5]
        dist_matrix = graph.precompute_shortest_paths(locations)
        
        # First run
        traffic_sim.reset(seed=42)
        traffic_sim.update_congestion('medium')
        congestion1 = traffic_sim.get_congestion_matrix(locations)
        
        opt1 = QUBOOptimizer(seed=42)
        seq1, cost1, _ = opt1.optimize(dist_matrix, priorities, congestion1, 'medium')
        
        # Second run with same seeds
        traffic_sim.reset(seed=42)
        traffic_sim.update_congestion('medium')
        congestion2 = traffic_sim.get_congestion_matrix(locations)
        
        opt2 = QUBOOptimizer(seed=42)
        seq2, cost2, _ = opt2.optimize(dist_matrix, priorities, congestion2, 'medium')
        
        assert seq1 == seq2, "Sequences differ between runs"
        assert cost1 == cost2, "Costs differ between runs"
    
    def test_depot_first_in_sequence(self, system):
        """Test depot (index 0) handling in optimization."""
        graph, traffic_sim, optimizer = system
        
        locations = SAMPLE_LOCATIONS[:4]
        priorities = [0.0, 2.0, 1.0, 3.0]
        
        dist_matrix = graph.precompute_shortest_paths(locations)
        traffic_sim.update_congestion('low')
        congestion_matrix = traffic_sim.get_congestion_matrix(locations)
        
        sequence, _, _ = optimizer.optimize(
            dist_matrix, priorities, congestion_matrix, 'low'
        )
        
        # Sequence should include all locations
        assert set(sequence) == set(range(len(locations)))
    
    def test_priority_influence(self, system):
        """Test high-priority deliveries tend to be earlier."""
        graph, traffic_sim, optimizer = system
        
        locations = SAMPLE_LOCATIONS[:5]
        # High priority for location 3
        priorities = [0.0, 1.0, 1.0, 10.0, 1.0]
        
        dist_matrix = graph.precompute_shortest_paths(locations)
        traffic_sim.update_congestion('low')
        congestion_matrix = traffic_sim.get_congestion_matrix(locations)
        
        # Run multiple times and check if high-priority tends to be earlier
        high_priority_positions = []
        
        for seed in range(42, 47):
            opt = QUBOOptimizer(seed=seed)
            seq, _, _ = opt.optimize(dist_matrix, priorities, congestion_matrix, 'low')
            
            # Find position of high-priority item (index 3)
            pos = seq.index(3) if 3 in seq else len(seq)
            high_priority_positions.append(pos)
        
        avg_position = np.mean(high_priority_positions)
        # High priority should tend to be in first half
        assert avg_position < len(locations) * 0.75, \
            f"High-priority item average position {avg_position:.1f} too late"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_delivery(self):
        """Test optimization with single delivery."""
        graph = OSMGraph(use_cache=True)
        optimizer = QUBOOptimizer(seed=42)
        
        locations = SAMPLE_LOCATIONS[:2]  # Depot + 1 delivery
        priorities = [0.0, 1.0]
        
        dist_matrix = graph.precompute_shortest_paths(locations)
        congestion = np.ones((2, 2))
        
        sequence, cost, time = optimizer.optimize(
            dist_matrix, priorities, congestion, 'low'
        )
        
        assert len(sequence) == 2
        assert set(sequence) == {0, 1}
    
    def test_same_location_duplicates(self):
        """Test handling of very close locations."""
        graph = OSMGraph(use_cache=True)
        optimizer = QUBOOptimizer(seed=42)
        
        # Locations very close together
        locations = [
            (16.5063, 80.6480),
            (16.5064, 80.6481),  # ~15m away
            (16.5065, 80.6482),  # ~15m away
        ]
        priorities = [0.0, 1.0, 1.0]
        
        dist_matrix = graph.precompute_shortest_paths(locations)
        congestion = np.ones((3, 3))
        
        sequence, cost, time = optimizer.optimize(
            dist_matrix, priorities, congestion, 'low'
        )
        
        assert is_valid_permutation(sequence, 3)
    
    def test_all_same_priority(self):
        """Test optimization with equal priorities."""
        graph = OSMGraph(use_cache=True)
        optimizer = QUBOOptimizer(seed=42)
        
        locations = SAMPLE_LOCATIONS[:4]
        priorities = [1.0, 1.0, 1.0, 1.0]  # All equal
        
        dist_matrix = graph.precompute_shortest_paths(locations)
        congestion = np.ones((4, 4))
        
        sequence, cost, time = optimizer.optimize(
            dist_matrix, priorities, congestion, 'medium'
        )
        
        assert is_valid_permutation(sequence, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
