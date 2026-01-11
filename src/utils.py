"""
Utility Functions for Quantum Traffic Optimization.

This module provides helper functions for ETA calculations, cost computation,
distance calculations, and coordinate validation.
"""

import math
from typing import List, Optional, Tuple

import numpy as np


# Constants
EARTH_RADIUS_KM = 6371.0
DEFAULT_SPEED_KMH = 30.0  # Average urban speed
METERS_PER_KM = 1000.0
SECONDS_PER_HOUR = 3600.0
MINUTES_PER_HOUR = 60.0

# Vijayawada bounding box
VIJAYAWADA_BBOX = (16.5, 16.7, 80.6, 80.7)  # (south, north, west, east)


def haversine_distance(
    lat1: float,
    lng1: float,
    lat2: float,
    lng2: float
) -> float:
    """
    Calculate great-circle distance between two points.
    
    Uses the Haversine formula to compute the shortest distance
    over the Earth's surface.
    
    Args:
        lat1, lng1: First point coordinates (degrees).
        lat2, lng2: Second point coordinates (degrees).
        
    Returns:
        Distance in kilometers.
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    
    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return EARTH_RADIUS_KM * c


def calculate_eta(
    distance_m: float,
    speed_kmh: float = DEFAULT_SPEED_KMH,
    congestion_factor: float = 1.0
) -> float:
    """
    Calculate estimated time of arrival in minutes.
    
    Args:
        distance_m: Distance in meters.
        speed_kmh: Base speed in km/h.
        congestion_factor: Traffic multiplier (1.0 = no congestion).
        
    Returns:
        Estimated travel time in minutes.
    """
    if distance_m <= 0:
        return 0.0
    
    # Adjust speed for congestion
    effective_speed = speed_kmh / congestion_factor
    
    # Convert distance to km and calculate time
    distance_km = distance_m / METERS_PER_KM
    time_hours = distance_km / effective_speed
    
    return time_hours * MINUTES_PER_HOUR


def calculate_total_cost(
    sequence: List[int],
    dist_matrix: np.ndarray,
    congestion_weights: Optional[np.ndarray] = None,
    priorities: Optional[List[float]] = None,
    lambda_weight: float = 0.5
) -> float:
    """
    Calculate total weighted cost for a route sequence.
    
    Cost = Σ distance + λ * Σ (congestion * distance) - Σ priority_bonus
    
    Args:
        sequence: Order of location indices to visit.
        dist_matrix: Distance matrix (n x n).
        congestion_weights: Optional congestion multipliers (n x n).
        priorities: Optional priority weights for each location.
        lambda_weight: Weight for congestion penalty.
        
    Returns:
        Total route cost.
    """
    if len(sequence) < 2:
        return 0.0
    
    n = len(sequence)
    total_cost = 0.0
    
    # Sum distances along the route
    for i in range(n - 1):
        src, dst = sequence[i], sequence[i + 1]
        base_dist = dist_matrix[src, dst]
        
        # Add base distance
        total_cost += base_dist
        
        # Add congestion penalty if provided
        if congestion_weights is not None:
            congestion = congestion_weights[src, dst]
            total_cost += lambda_weight * congestion * base_dist
    
    # Subtract priority bonuses (earlier = better for high priority)
    if priorities is not None:
        max_dist = np.max(dist_matrix) if dist_matrix.size > 0 else 1000
        for pos, loc_idx in enumerate(sequence):
            if loc_idx < len(priorities):
                # Earlier positions get larger bonus for high priority
                position_factor = (n - pos) / n
                total_cost -= priorities[loc_idx] * position_factor * 0.1 * max_dist
    
    return total_cost


def calculate_route_distance(
    sequence: List[int],
    dist_matrix: np.ndarray
) -> float:
    """
    Calculate total distance for a route sequence.
    
    Args:
        sequence: Order of location indices.
        dist_matrix: Distance matrix.
        
    Returns:
        Total distance in same units as dist_matrix.
    """
    if len(sequence) < 2:
        return 0.0
    
    total = 0.0
    for i in range(len(sequence) - 1):
        total += dist_matrix[sequence[i], sequence[i + 1]]
    
    return total


def calculate_route_eta(
    sequence: List[int],
    dist_matrix: np.ndarray,
    congestion_weights: Optional[np.ndarray] = None,
    speed_kmh: float = DEFAULT_SPEED_KMH
) -> float:
    """
    Calculate total ETA for a route sequence.
    
    Args:
        sequence: Order of location indices.
        dist_matrix: Distance matrix (meters).
        congestion_weights: Optional congestion multipliers.
        speed_kmh: Base speed.
        
    Returns:
        Total time in minutes.
    """
    if len(sequence) < 2:
        return 0.0
    
    total_time = 0.0
    for i in range(len(sequence) - 1):
        src, dst = sequence[i], sequence[i + 1]
        distance = dist_matrix[src, dst]
        
        congestion = 1.0
        if congestion_weights is not None:
            congestion = congestion_weights[src, dst]
        
        total_time += calculate_eta(distance, speed_kmh, congestion)
    
    return total_time


def validate_coordinates(
    lat: float,
    lng: float,
    bbox: Tuple[float, float, float, float] = VIJAYAWADA_BBOX
) -> bool:
    """
    Validate coordinates are within bounding box.
    
    Args:
        lat: Latitude to check.
        lng: Longitude to check.
        bbox: Bounding box (south, north, west, east).
        
    Returns:
        True if coordinates are within bounds.
    """
    south, north, west, east = bbox
    return south <= lat <= north and west <= lng <= east


def validate_locations(
    locations: List[Tuple[float, float]],
    bbox: Tuple[float, float, float, float] = VIJAYAWADA_BBOX
) -> List[bool]:
    """
    Validate multiple locations against bounding box.
    
    Args:
        locations: List of (lat, lng) tuples.
        bbox: Bounding box.
        
    Returns:
        List of booleans indicating validity.
    """
    return [validate_coordinates(lat, lng, bbox) for lat, lng in locations]


def is_valid_permutation(sequence: List[int], n: int) -> bool:
    """
    Check if sequence is a valid permutation of 0..n-1.
    
    Args:
        sequence: Sequence to validate.
        n: Expected number of elements.
        
    Returns:
        True if valid permutation.
    """
    if len(sequence) != n:
        return False
    return set(sequence) == set(range(n))


def format_eta(minutes: float) -> str:
    """
    Format ETA as human-readable string.
    
    Args:
        minutes: Time in minutes.
        
    Returns:
        Formatted string like "1h 30m" or "45m".
    """
    if minutes < 1:
        return f"{int(minutes * 60)}s"
    elif minutes < 60:
        return f"{int(minutes)}m"
    else:
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        if mins > 0:
            return f"{hours}h {mins}m"
        return f"{hours}h"


def format_distance(meters: float) -> str:
    """
    Format distance as human-readable string.
    
    Args:
        meters: Distance in meters.
        
    Returns:
        Formatted string like "1.5 km" or "500 m".
    """
    if meters < 1000:
        return f"{int(meters)} m"
    else:
        return f"{meters / 1000:.1f} km"


def compute_improvement(
    optimized_cost: float,
    baseline_cost: float
) -> float:
    """
    Compute percentage improvement over baseline.
    
    Args:
        optimized_cost: Cost of optimized solution.
        baseline_cost: Cost of baseline solution.
        
    Returns:
        Percentage improvement (negative = worse).
    """
    if baseline_cost <= 0:
        return 0.0
    return (baseline_cost - optimized_cost) / baseline_cost * 100


def generate_route_id() -> str:
    """
    Generate a unique route identifier.
    
    Returns:
        8-character alphanumeric ID.
    """
    import uuid
    return str(uuid.uuid4())[:8]


# Demo function
def demo():
    """Demonstrate utility functions."""
    print("=" * 60)
    print("Utility Functions Demo")
    print("=" * 60)
    
    # Test haversine distance
    lat1, lng1 = 16.5063, 80.6480
    lat2, lng2 = 16.5175, 80.6198
    dist = haversine_distance(lat1, lng1, lat2, lng2)
    print(f"\nHaversine distance: {dist:.2f} km ({dist * 1000:.0f} m)")
    
    # Test ETA calculation
    eta_low = calculate_eta(dist * 1000, congestion_factor=1.0)
    eta_high = calculate_eta(dist * 1000, congestion_factor=2.5)
    print(f"ETA (low traffic): {format_eta(eta_low)}")
    print(f"ETA (high traffic): {format_eta(eta_high)}")
    
    # Test coordinate validation
    valid = validate_coordinates(16.55, 80.65)
    invalid = validate_coordinates(17.0, 80.65)
    print(f"\n(16.55, 80.65) valid: {valid}")
    print(f"(17.0, 80.65) valid: {invalid}")
    
    # Test cost calculation
    dist_matrix = np.array([
        [0, 1000, 2000],
        [1000, 0, 1500],
        [2000, 1500, 0]
    ])
    sequence = [0, 1, 2]
    cost = calculate_route_distance(sequence, dist_matrix)
    print(f"\nRoute distance: {format_distance(cost)}")
    
    # Test improvement calculation
    improvement = compute_improvement(1500, 2000)
    print(f"Improvement: {improvement:.1f}%")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    demo()
