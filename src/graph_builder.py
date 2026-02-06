"""
OSMnx Graph Builder for Quantum Traffic Optimization.

This module provides functionality to load and process OpenStreetMap road networks
for Vijayawada, India, and compute shortest paths between delivery locations.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# Try to import osmnx, provide fallback for environments without it
try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False


class OSMGraph:
    """
    OSMnx graph manager for Vijayawada road network.

    Handles graph loading, caching, and shortest path computations
    for delivery route optimization.

    Attributes:
        bbox: Bounding box (south, north, west, east) for the area.
        graph: NetworkX MultiDiGraph representing the road network.
        cache_path: Path to cached graph file.
        node_cache: Cached nearest nodes for location lookups.

    Environment Variables:
        DEMO_MODE: Set to "1" to use synthetic graph (faster startup).
    """

    # Default bounding box for Vijayawada, Andhra Pradesh, India
    # Using smaller area for faster OSM download
    DEFAULT_BBOX = (16.50, 16.55, 80.62, 80.68)  # Smaller area ~5x5 km
    
    # Congestion multipliers for travel time estimation
    CONGESTION_MULTIPLIERS = {
        'low': 1.0,
        'medium': 1.5,
        'high': 2.5
    }
    
    def __init__(
        self,
        bbox: Tuple[float, float, float, float] = None,
        cache_dir: str = "data",
        use_cache: bool = True,
        demo_mode: bool = None
    ):
        """
        Initialize the OSMGraph with a bounding box.

        Args:
            bbox: Tuple of (south, north, west, east) coordinates.
            cache_dir: Directory for caching graph data.
            use_cache: Whether to use cached graph if available.
            demo_mode: Force synthetic graph (None = check env var DEMO_MODE).
        """
        # Check for demo mode (environment variable or parameter)
        if demo_mode is None:
            demo_mode = os.environ.get('DEMO_MODE', '0') == '1'
        self.demo_mode = demo_mode

        self.bbox = bbox if bbox is not None else self.DEFAULT_BBOX
        self.cache_dir = Path(cache_dir)
        self.cache_path = self.cache_dir / "vijayawada_graph.graphml"
        self.dist_cache_path = self.cache_dir / "distance_cache.pkl"
        self.use_cache = use_cache

        self.graph: Optional[nx.MultiDiGraph] = None
        self.node_cache: Dict[Tuple[float, float], int] = {}
        self._dist_matrix_cache: Dict[str, np.ndarray] = {}
        self._locations_cache: Optional[List[Tuple[float, float]]] = None

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load or download graph
        self._load_or_download_graph()
    
    def _load_or_download_graph(self) -> None:
        """Load graph from cache or download from OSM."""
        # Use synthetic graph in demo mode
        if self.demo_mode:
            logger.info("Demo mode enabled. Using synthetic graph for fast startup.")
            self._create_synthetic_graph()
            return

        if self.use_cache and self.cache_path.exists() and OSMNX_AVAILABLE:
            logger.info(f"Loading cached graph from {self.cache_path}")
            self.graph = ox.load_graphml(self.cache_path)
            logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        elif OSMNX_AVAILABLE:
            logger.info(f"Downloading graph for bbox {self.bbox}...")
            # Configure osmnx for larger query areas to avoid splitting
            ox.settings.max_query_area_size = 50 * 1000 * 50 * 1000  # 50km x 50km
            ox.settings.timeout = 300  # 5 minutes timeout

            # OSMnx 2.x uses bbox tuple (north, south, east, west) or (left, bottom, right, top)
            # Try new API first, fall back to old API for compatibility
            try:
                # New OSMnx 2.x API: bbox=(north, south, east, west)
                self.graph = ox.graph_from_bbox(
                    bbox=(self.bbox[1], self.bbox[0], self.bbox[3], self.bbox[2]),
                    network_type='drive'
                )
            except TypeError:
                # Old OSMnx 1.x API with named parameters
                self.graph = ox.graph_from_bbox(
                    north=self.bbox[1],
                    south=self.bbox[0],
                    east=self.bbox[3],
                    west=self.bbox[2],
                    network_type='drive'
                )
            # Add travel time as edge weight (assuming 30 km/h average speed)
            self.graph = ox.add_edge_speeds(self.graph)
            self.graph = ox.add_edge_travel_times(self.graph)
            
            # Save to cache
            logger.info(f"Saving graph to {self.cache_path}")
            ox.save_graphml(self.graph, self.cache_path)
            logger.info(f"Downloaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        else:
            # Create a synthetic graph for testing when OSMnx is not available
            logger.warning("OSMnx not available. Creating synthetic graph for testing.")
            self._create_synthetic_graph()
    
    def _create_synthetic_graph(self) -> None:
        """Create a synthetic graph for testing when OSMnx is unavailable."""
        self.graph = nx.MultiDiGraph()

        # Add graph-level attributes expected by OSMnx
        self.graph.graph['crs'] = 'EPSG:4326'  # WGS84 coordinate system
        self.graph.graph['simplified'] = True

        # Create a grid of nodes within the bounding box
        n_points = 10
        lats = np.linspace(self.bbox[0], self.bbox[1], n_points)
        lngs = np.linspace(self.bbox[2], self.bbox[3], n_points)
        
        node_id = 0
        node_map = {}
        
        for i, lat in enumerate(lats):
            for j, lng in enumerate(lngs):
                self.graph.add_node(node_id, y=lat, x=lng)
                node_map[(i, j)] = node_id
                node_id += 1
        
        # Connect adjacent nodes with edges
        for i in range(n_points):
            for j in range(n_points):
                current = node_map[(i, j)]
                
                # Connect to right neighbor
                if j < n_points - 1:
                    neighbor = node_map[(i, j + 1)]
                    dist = self._haversine(lats[i], lngs[j], lats[i], lngs[j + 1])
                    self.graph.add_edge(current, neighbor, length=dist * 1000, travel_time=dist * 1000 / 8.33)
                    self.graph.add_edge(neighbor, current, length=dist * 1000, travel_time=dist * 1000 / 8.33)
                
                # Connect to bottom neighbor
                if i < n_points - 1:
                    neighbor = node_map[(i + 1, j)]
                    dist = self._haversine(lats[i], lngs[j], lats[i + 1], lngs[j])
                    self.graph.add_edge(current, neighbor, length=dist * 1000, travel_time=dist * 1000 / 8.33)
                    self.graph.add_edge(neighbor, current, length=dist * 1000, travel_time=dist * 1000 / 8.33)
    
    def _haversine(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """
        Calculate great-circle distance between two points in kilometers.
        
        Args:
            lat1, lng1: First point coordinates.
            lat2, lng2: Second point coordinates.
            
        Returns:
            Distance in kilometers.
        """
        R = 6371.0  # Earth's radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlng = np.radians(lng2 - lng1)
        
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlng / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    def get_nearest_node(self, lat: float, lng: float) -> int:
        """
        Find the nearest graph node to given coordinates.
        
        Args:
            lat: Latitude.
            lng: Longitude.
            
        Returns:
            Node ID of nearest node.
        """
        cache_key = (round(lat, 6), round(lng, 6))
        
        if cache_key in self.node_cache:
            return self.node_cache[cache_key]
        
        if OSMNX_AVAILABLE and hasattr(ox, 'nearest_nodes'):
            node = ox.nearest_nodes(self.graph, lng, lat)
        else:
            # Manual nearest node search
            min_dist = float('inf')
            nearest = None
            for node, data in self.graph.nodes(data=True):
                dist = self._haversine(lat, lng, data['y'], data['x'])
                if dist < min_dist:
                    min_dist = dist
                    nearest = node
            node = nearest
        
        self.node_cache[cache_key] = node
        return node
    
    def precompute_shortest_paths(
        self,
        locations: List[Tuple[float, float]],
        weight: str = 'length'
    ) -> np.ndarray:
        """
        Compute all-pairs shortest path distance matrix.
        
        Args:
            locations: List of (lat, lng) tuples.
            weight: Edge weight to use ('length' or 'travel_time').
            
        Returns:
            Symmetric distance matrix of shape (n, n) with zeros on diagonal.
        """
        n = len(locations)
        dist_matrix = np.zeros((n, n))
        
        # Get nearest nodes for all locations
        nodes = [self.get_nearest_node(lat, lng) for lat, lng in locations]
        
        # Compute shortest paths
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    # Try to find shortest path
                    path_length = nx.shortest_path_length(
                        self.graph, nodes[i], nodes[j], weight=weight
                    )
                    dist_matrix[i, j] = path_length
                    dist_matrix[j, i] = path_length
                except nx.NetworkXNoPath:
                    # If no path exists, use haversine distance * factor
                    fallback_dist = self._haversine(
                        locations[i][0], locations[i][1],
                        locations[j][0], locations[j][1]
                    ) * 1000 * 1.4  # meters with road factor
                    dist_matrix[i, j] = fallback_dist
                    dist_matrix[j, i] = fallback_dist
        
        # Cache the locations for later use
        self._locations_cache = locations
        
        return dist_matrix
    
    def get_route_weight(
        self,
        src_idx: int,
        dst_idx: int,
        dist_matrix: np.ndarray,
        congestion_level: str = 'low'
    ) -> float:
        """
        Get weighted distance between two indexed locations.
        
        Args:
            src_idx: Source location index.
            dst_idx: Destination location index.
            dist_matrix: Precomputed distance matrix.
            congestion_level: Traffic level ('low', 'medium', 'high').
            
        Returns:
            Weighted distance considering congestion.
        """
        base_dist = dist_matrix[src_idx, dst_idx]
        multiplier = self.CONGESTION_MULTIPLIERS.get(congestion_level, 1.0)
        return base_dist * multiplier
    
    def get_route_geometry(
        self,
        locations: List[Tuple[float, float]],
        sequence: List[int]
    ) -> List[List[Tuple[float, float]]]:
        """
        Get route polyline coordinates for visualization.

        Args:
            locations: List of (lat, lng) tuples.
            sequence: Order of location indices to visit.

        Returns:
            List of polyline segments, each as list of (lat, lng) tuples.
        """
        route_segments = []
        nodes = [self.get_nearest_node(lat, lng) for lat, lng in locations]

        for i in range(len(sequence) - 1):
            src_node = nodes[sequence[i]]
            dst_node = nodes[sequence[i + 1]]

            segment = self._find_road_path(src_node, dst_node, locations[sequence[i]], locations[sequence[i + 1]])
            route_segments.append(segment)

        return route_segments

    def _find_road_path(
        self,
        src_node: int,
        dst_node: int,
        src_loc: Tuple[float, float],
        dst_loc: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        """
        Find a road-following path between two nodes.

        Tries multiple strategies to avoid direct lines through water/obstacles:
        1. Direct shortest path (directed graph - respects one-way streets)
        2. Undirected shortest path (ignores one-way restrictions)
        3. Path via intermediate waypoints along roads
        4. Curved road-following approximation

        Args:
            src_node: Source graph node.
            dst_node: Destination graph node.
            src_loc: Source (lat, lng) coordinates.
            dst_loc: Destination (lat, lng) coordinates.

        Returns:
            List of (lat, lng) tuples forming a road-following path.
        """
        # Strategy 1: Try direct shortest path on directed graph first
        try:
            path = nx.shortest_path(self.graph, src_node, dst_node, weight='length')
            return [
                (self.graph.nodes[node]['y'], self.graph.nodes[node]['x'])
                for node in path
            ]
        except nx.NetworkXNoPath:
            pass

        # Strategy 2: Try undirected graph (ignores one-way restrictions for visualization)
        try:
            undirected = self.graph.to_undirected()
            path = nx.shortest_path(undirected, src_node, dst_node, weight='length')
            return [
                (self.graph.nodes[node]['y'], self.graph.nodes[node]['x'])
                for node in path
            ]
        except nx.NetworkXNoPath:
            pass

        # Strategy 3: Try finding nearby nodes that ARE connected
        try:
            # Get neighbors within 500m of source and destination
            src_neighbors = self._find_connected_nearby_nodes(src_node, src_loc, radius_km=0.5)
            dst_neighbors = self._find_connected_nearby_nodes(dst_node, dst_loc, radius_km=0.5)

            # Try to find a path through any combination of nearby nodes
            undirected = self.graph.to_undirected()
            best_path = None
            best_length = float('inf')

            for s_node in src_neighbors[:5]:  # Limit to top 5 for performance
                for d_node in dst_neighbors[:5]:
                    try:
                        path = nx.shortest_path(undirected, s_node, d_node, weight='length')
                        path_length = sum(
                            self.graph.edges.get((path[i], path[i+1], 0), {}).get('length', 100)
                            for i in range(len(path) - 1)
                        )
                        if path_length < best_length:
                            best_length = path_length
                            best_path = path
                    except nx.NetworkXNoPath:
                        continue

            if best_path:
                coords = [
                    (self.graph.nodes[node]['y'], self.graph.nodes[node]['x'])
                    for node in best_path
                ]
                # Prepend source location and append destination location
                return [src_loc] + coords + [dst_loc]
        except Exception:
            pass

        # Strategy 4: Create road-following curved path using nearby road nodes
        return self._create_road_approximation_path(src_node, dst_node, src_loc, dst_loc)

    def _find_connected_nearby_nodes(
        self,
        center_node: int,
        center_loc: Tuple[float, float],
        radius_km: float = 0.5
    ) -> List[int]:
        """
        Find nodes near a location that are well-connected in the graph.

        Args:
            center_node: The center node.
            center_loc: The center (lat, lng) coordinates.
            radius_km: Search radius in kilometers.

        Returns:
            List of node IDs sorted by distance from center.
        """
        nearby_nodes = []
        for node in self.graph.nodes():
            node_lat = self.graph.nodes[node]['y']
            node_lng = self.graph.nodes[node]['x']
            dist = self._haversine(center_loc[0], center_loc[1], node_lat, node_lng)
            if dist <= radius_km:
                # Prefer nodes with more connections
                degree = self.graph.degree(node)
                nearby_nodes.append((node, dist, degree))

        # Sort by distance, but prefer well-connected nodes
        nearby_nodes.sort(key=lambda x: (x[1] - 0.1 * min(x[2], 5)))
        return [n[0] for n in nearby_nodes]

    def _create_road_approximation_path(
        self,
        src_node: int,
        dst_node: int,
        src_loc: Tuple[float, float],
        dst_loc: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        """
        Create an approximated path that follows nearby roads.

        When no direct path exists, this finds nearby road nodes and
        creates a path that visually follows the road network instead
        of cutting through water or obstacles.
        """
        path_points = [src_loc]

        # Use more waypoints for smoother path
        num_waypoints = 10
        last_added_node = None

        for i in range(1, num_waypoints):
            t = i / num_waypoints
            # Interpolate position
            interp_lat = src_loc[0] + t * (dst_loc[0] - src_loc[0])
            interp_lng = src_loc[1] + t * (dst_loc[1] - src_loc[1])

            # Find nearest road node to this interpolated position
            nearest = self.get_nearest_node(interp_lat, interp_lng)

            # Skip if same as last added node
            if nearest == last_added_node:
                continue

            node_lat = self.graph.nodes[nearest]['y']
            node_lng = self.graph.nodes[nearest]['x']

            # Add the road node position (with larger threshold)
            dist_to_line = self._haversine(interp_lat, interp_lng, node_lat, node_lng)
            if dist_to_line < 1.0:  # 1 km threshold - more lenient
                path_points.append((node_lat, node_lng))
                last_added_node = nearest

        path_points.append(dst_loc)

        # Remove duplicates and very close points
        unique_points = [path_points[0]]
        for p in path_points[1:]:
            last = unique_points[-1]
            # Skip if very close to last point (less than 20m)
            dist = self._haversine(p[0], p[1], last[0], last[1])
            if dist > 0.02:  # 20 meters
                unique_points.append(p)

        # Ensure destination is included
        if unique_points[-1] != dst_loc:
            unique_points.append(dst_loc)

        return unique_points

    def get_node_coordinates(self, node_id: int) -> Tuple[float, float]:
        """
        Get coordinates of a graph node.
        
        Args:
            node_id: Node ID in the graph.
            
        Returns:
            Tuple of (lat, lng).
        """
        return (self.graph.nodes[node_id]['y'], self.graph.nodes[node_id]['x'])
    
    def validate_location(self, lat: float, lng: float) -> bool:
        """
        Check if a location is within the bounding box.
        
        Args:
            lat: Latitude.
            lng: Longitude.
            
        Returns:
            True if location is within bounds.
        """
        return (
            self.bbox[0] <= lat <= self.bbox[1] and
            self.bbox[2] <= lng <= self.bbox[3]
        )


# Sample locations for testing (within smaller bbox)
SAMPLE_LOCATIONS = [
    (16.505, 80.625),   # Depot
    (16.51, 80.63),     # Delivery A
    (16.52, 80.64),     # Delivery B
    (16.53, 80.65),     # Delivery C
    (16.54, 80.66),     # Delivery D
]


def demo():
    """Demonstrate the OSMGraph functionality."""
    print("=" * 60)
    print("OSMGraph Demo - Vijayawada Road Network")
    print("=" * 60)
    
    # Initialize graph
    graph = OSMGraph()
    
    # Test with sample locations
    print(f"\nSample locations: {len(SAMPLE_LOCATIONS)}")
    for i, (lat, lng) in enumerate(SAMPLE_LOCATIONS):
        print(f"  {i}: ({lat:.4f}, {lng:.4f})")
    
    # Compute distance matrix
    print("\nComputing shortest path distances...")
    dist_matrix = graph.precompute_shortest_paths(SAMPLE_LOCATIONS)
    
    print("\nDistance matrix (meters):")
    print(np.round(dist_matrix, 1))
    
    # Test weighted distances
    print("\nWeighted distances from location 0 to 1:")
    for level in ['low', 'medium', 'high']:
        weighted = graph.get_route_weight(0, 1, dist_matrix, level)
        print(f"  {level}: {weighted:.1f}m")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    demo()
