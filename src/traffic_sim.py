"""
Traffic Simulator for Quantum Traffic Optimization.

This module provides dynamic traffic congestion simulation for the road network,
modifying edge weights based on simulated real-time traffic conditions.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .graph_builder import OSMGraph


class TrafficSimulator:
    """
    Simulates dynamic traffic congestion on road network.
    
    Models real-world traffic patterns by applying congestion multipliers
    to road segments, supporting different traffic levels and random
    congestion distribution.
    
    Attributes:
        graph: OSMGraph instance for the road network.
        congestion_level: Current traffic level ('low', 'medium', 'high').
        edge_congestion: Dict mapping edge IDs to congestion factors.
        seed: Random seed for reproducibility.
    """
    
    # Traffic congestion multipliers
    CONGESTION_MULTIPLIERS = {
        'low': 1.0,
        'medium': 1.5,
        'high': 2.5
    }
    
    # Percentage of edges affected at each level
    CONGESTION_RATES = {
        'low': 0.10,    # 10% of edges have some congestion
        'medium': 0.20,  # 20% of edges congested
        'high': 0.30,    # 30% of edges heavily congested
    }
    
    # Base speed in m/s (approximately 30 km/h)
    BASE_SPEED = 8.33
    
    def __init__(
        self,
        osm_graph: OSMGraph,
        seed: int = 42,
        initial_level: str = 'low'
    ):
        """
        Initialize the TrafficSimulator.
        
        Args:
            osm_graph: OSMGraph instance for the road network.
            seed: Random seed for reproducibility (default: 42).
            initial_level: Initial traffic level ('low', 'medium', 'high').
        """
        self.osm_graph = osm_graph
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        
        self.congestion_level = initial_level
        self.edge_congestion: Dict[Tuple, float] = {}
        
        # Initialize congestion state
        self.update_congestion(initial_level)
    
    def update_congestion(self, level: str) -> None:
        """
        Update edge weights based on congestion level.
        
        Simulates traffic by randomly selecting edges and applying
        Poisson-distributed congestion multipliers.
        
        Args:
            level: Traffic level ('low', 'medium', 'high').
        """
        if level not in self.CONGESTION_MULTIPLIERS:
            raise ValueError(f"Invalid congestion level: {level}. Must be 'low', 'medium', or 'high'.")
        
        self.congestion_level = level
        self.edge_congestion.clear()
        
        graph = self.osm_graph.graph
        if graph is None:
            return
        
        # Get all edges
        edges = list(graph.edges(keys=True))
        n_edges = len(edges)
        
        if n_edges == 0:
            return
        
        # Determine number of congested edges
        congestion_rate = self.CONGESTION_RATES.get(level, 0.1)
        n_congested = int(n_edges * congestion_rate)
        
        # Randomly select edges to congest
        congested_indices = self.rng.choice(n_edges, size=n_congested, replace=False)
        
        # Base multiplier for this level
        base_multiplier = self.CONGESTION_MULTIPLIERS[level]
        
        # Apply Poisson-distributed congestion to selected edges
        for idx in congested_indices:
            edge = edges[idx]
            
            # Poisson distribution with mean = level multiplier
            # This adds variability to congestion
            if level == 'high':
                mu = 2.0
            elif level == 'medium':
                mu = 1.5
            else:
                mu = 1.0
            
            # Generate Poisson factor and scale
            poisson_factor = self.rng.poisson(mu)
            congestion_factor = 1.0 + (poisson_factor * 0.3)  # Scale factor
            
            # Cap maximum congestion at 4x
            congestion_factor = min(congestion_factor, 4.0)
            
            self.edge_congestion[(edge[0], edge[1], edge[2])] = congestion_factor
    
    def get_edge_congestion(self, u: int, v: int, key: int = 0) -> float:
        """
        Get congestion factor for a specific edge.
        
        Args:
            u: Source node.
            v: Destination node.
            key: Edge key for multigraph.
            
        Returns:
            Congestion multiplier (1.0 if no congestion).
        """
        return self.edge_congestion.get((u, v, key), 1.0)
    
    def get_dynamic_weights(
        self,
        locations: List[Tuple[float, float]],
        dist_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Get current time-weighted travel estimates between locations.
        
        Applies the current congestion state to the base distance matrix
        to compute dynamic travel times.
        
        Args:
            locations: List of (lat, lng) tuples.
            dist_matrix: Base distance matrix from OSMGraph.
            
        Returns:
            Weighted distance matrix reflecting current traffic.
        """
        n = len(locations)
        weighted_matrix = np.zeros((n, n))
        
        # Get base multiplier for current level
        level_multiplier = self.CONGESTION_MULTIPLIERS[self.congestion_level]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                # Apply level-based multiplier plus some random variation
                # This simulates that different routes have different congestion
                route_variation = 1.0 + self.rng.uniform(-0.1, 0.2)
                weighted_matrix[i, j] = dist_matrix[i, j] * level_multiplier * route_variation
        
        return weighted_matrix
    
    def get_congestion_matrix(
        self,
        locations: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Get current congestion factors for location pairs.
        
        Args:
            locations: List of (lat, lng) tuples.
            
        Returns:
            Matrix of congestion multipliers between locations.
        """
        n = len(locations)
        congestion_matrix = np.ones((n, n))
        
        level_multiplier = self.CONGESTION_MULTIPLIERS[self.congestion_level]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Add some variation based on location indices
                    variation = 1.0 + 0.1 * np.sin((i + j) * self.seed % 10)
                    congestion_matrix[i, j] = level_multiplier * variation
        
        return congestion_matrix
    
    def estimate_travel_time(
        self,
        distance_m: float,
        congestion_factor: float = 1.0
    ) -> float:
        """
        Estimate travel time in seconds.
        
        Args:
            distance_m: Distance in meters.
            congestion_factor: Congestion multiplier.
            
        Returns:
            Estimated travel time in seconds.
        """
        effective_speed = self.BASE_SPEED / congestion_factor
        return distance_m / effective_speed
    
    def get_current_level(self) -> str:
        """Get the current congestion level."""
        return self.congestion_level
    
    def get_level_multiplier(self, level: Optional[str] = None) -> float:
        """
        Get the base multiplier for a traffic level.
        
        Args:
            level: Traffic level (uses current level if None).
            
        Returns:
            Congestion multiplier value.
        """
        if level is None:
            level = self.congestion_level
        return self.CONGESTION_MULTIPLIERS.get(level, 1.0)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the simulator to initial state.
        
        Args:
            seed: New random seed (uses original if None).
        """
        if seed is not None:
            self.seed = seed
        
        self.rng = np.random.default_rng(self.seed)
        random.seed(self.seed)
        self.edge_congestion.clear()
        self.update_congestion(self.congestion_level)


def demo():
    """Demonstrate the TrafficSimulator functionality."""
    print("=" * 60)
    print("TrafficSimulator Demo")
    print("=" * 60)
    
    # Initialize graph and simulator
    osm_graph = OSMGraph()
    simulator = TrafficSimulator(osm_graph, seed=42)
    
    # Sample locations
    from .graph_builder import SAMPLE_LOCATIONS
    locations = SAMPLE_LOCATIONS[:5]
    
    # Compute base distances
    dist_matrix = osm_graph.precompute_shortest_paths(locations)
    
    print(f"\nBase distance matrix (meters):")
    print(np.round(dist_matrix, 1))
    
    # Test different congestion levels
    for level in ['low', 'medium', 'high']:
        simulator.update_congestion(level)
        weighted = simulator.get_dynamic_weights(locations, dist_matrix)
        
        print(f"\n{level.upper()} traffic - weighted distances:")
        print(np.round(weighted, 1))
        
        # Show congestion matrix
        congestion = simulator.get_congestion_matrix(locations)
        print(f"\nCongestion factors ({level}):")
        print(np.round(congestion, 2))
    
    # Test travel time estimation
    print("\n" + "-" * 40)
    print("Travel time estimates (location 0 to 1):")
    base_dist = dist_matrix[0, 1]
    for level in ['low', 'medium', 'high']:
        factor = simulator.get_level_multiplier(level)
        time_s = simulator.estimate_travel_time(base_dist, factor)
        print(f"  {level}: {time_s:.1f}s ({time_s/60:.1f} min)")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    demo()
