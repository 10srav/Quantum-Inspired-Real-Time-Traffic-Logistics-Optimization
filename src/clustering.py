"""
Hierarchical Clustering for Large-Scale TSP.

Implements K-means clustering with cluster-level TSP optimization
for handling 200+ delivery points efficiently.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ClusterConfig:
    """Configuration for clustering."""
    cluster_threshold: int = 40  # Use clustering when n > this
    max_cluster_size: int = 40   # Maximum nodes per cluster
    min_cluster_size: int = 3    # Minimum nodes per cluster
    auto_k: bool = True          # Automatically determine K


@dataclass
class ClusterResult:
    """Result of cluster-level optimization."""
    cluster_id: int
    node_indices: List[int]  # Global indices of nodes in this cluster
    local_sequence: List[int]  # Optimized sequence within cluster (local indices)
    centroid: Tuple[float, float]
    cost: float
    solve_time: float


class HierarchicalOptimizer:
    """
    Two-level hierarchical optimizer for large-scale TSP.

    Strategy:
    1. Cluster locations using K-means
    2. Solve TSP within each cluster (using existing optimizer)
    3. Optimize inter-cluster visitation order
    4. Merge into final global sequence
    """

    def __init__(
        self,
        config: Optional[ClusterConfig] = None,
        qubo_optimizer: Optional[Any] = None
    ):
        self.config = config or ClusterConfig()
        self.optimizer = qubo_optimizer

        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, clustering disabled")

    def should_use_clustering(self, n: int) -> bool:
        """Determine if clustering should be used."""
        return SKLEARN_AVAILABLE and n > self.config.cluster_threshold

    def compute_optimal_k(
        self,
        coords: np.ndarray
    ) -> int:
        """
        Compute optimal number of clusters using silhouette score.

        Args:
            coords: Array of (lat, lng) coordinates.

        Returns:
            Optimal number of clusters.
        """
        n = len(coords)

        # Constraints: min/max cluster size
        min_k = max(2, n // self.config.max_cluster_size)
        max_k = min(n // self.config.min_cluster_size, n // 3, 20)  # Cap at 20 clusters

        if min_k >= max_k:
            return max(2, min_k)

        best_k = min_k
        best_score = -1

        for k in range(min_k, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(coords)

                # Check if we have at least 2 unique labels
                if len(np.unique(labels)) < 2:
                    continue

                score = silhouette_score(coords, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue

        logger.info(f"Optimal K={best_k} (silhouette={best_score:.3f}) for {n} nodes")
        return best_k

    def cluster_locations(
        self,
        coords: np.ndarray,
        k: Optional[int] = None
    ) -> Tuple[List[List[int]], np.ndarray]:
        """
        Cluster locations using K-means.

        Args:
            coords: Array of (lat, lng) coordinates.
            k: Number of clusters (auto-computed if None).

        Returns:
            Tuple of (cluster_indices, centroids).
            cluster_indices[i] = list of location indices in cluster i.
        """
        n = len(coords)

        if k is None and self.config.auto_k:
            k = self.compute_optimal_k(coords)
        elif k is None:
            k = max(2, n // self.config.max_cluster_size)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)

        # Group indices by cluster
        clusters: List[List[int]] = [[] for _ in range(k)]
        for i, label in enumerate(labels):
            clusters[label].append(i)

        # Remove empty clusters
        clusters = [c for c in clusters if len(c) > 0]

        return clusters, kmeans.cluster_centers_

    def solve_cluster_tsp(
        self,
        cluster_indices: List[int],
        dist_matrix: np.ndarray,
        priorities: List[float],
        congestion_weights: np.ndarray,
        traffic_level: str,
        coords: np.ndarray
    ) -> ClusterResult:
        """
        Solve TSP for a single cluster.

        Args:
            cluster_indices: Global indices of nodes in this cluster.
            dist_matrix: Full distance matrix (global).
            priorities: Priority weights (global).
            congestion_weights: Congestion multipliers (global).
            traffic_level: Traffic level.
            coords: Coordinates array (global).

        Returns:
            ClusterResult with optimized sequence.
        """
        n_cluster = len(cluster_indices)
        start_time = time.time()

        # Extract sub-matrices for this cluster
        sub_dist = np.zeros((n_cluster, n_cluster))
        sub_cong = np.zeros((n_cluster, n_cluster))
        sub_prio = [priorities[i] for i in cluster_indices]

        for i, gi in enumerate(cluster_indices):
            for j, gj in enumerate(cluster_indices):
                sub_dist[i, j] = dist_matrix[gi, gj]
                sub_cong[i, j] = congestion_weights[gi, gj]

        # Solve using the optimizer
        if self.optimizer is not None:
            local_seq, cost, _ = self.optimizer.optimize(
                sub_dist, sub_prio, sub_cong, traffic_level
            )
        else:
            # Fallback: greedy nearest neighbor
            local_seq, cost = self._greedy_tsp(sub_dist)

        solve_time = time.time() - start_time

        # Compute centroid
        cluster_coords = coords[cluster_indices]
        centroid = (float(np.mean(cluster_coords[:, 0])), float(np.mean(cluster_coords[:, 1])))

        return ClusterResult(
            cluster_id=0,  # Set by caller
            node_indices=cluster_indices,
            local_sequence=local_seq,
            centroid=centroid,
            cost=cost,
            solve_time=solve_time
        )

    def _greedy_tsp(self, dist_matrix: np.ndarray) -> Tuple[List[int], float]:
        """Simple greedy TSP solver."""
        n = len(dist_matrix)
        visited = [False] * n
        sequence = [0]
        visited[0] = True
        total_cost = 0.0
        current = 0

        for _ in range(n - 1):
            best_next = -1
            best_dist = float('inf')

            for j in range(n):
                if not visited[j] and dist_matrix[current, j] < best_dist:
                    best_dist = dist_matrix[current, j]
                    best_next = j

            if best_next >= 0:
                sequence.append(best_next)
                visited[best_next] = True
                total_cost += best_dist
                current = best_next

        return sequence, total_cost

    def optimize_cluster_order(
        self,
        cluster_results: List[ClusterResult],
        depot_coord: Tuple[float, float]
    ) -> List[int]:
        """
        Optimize the order of visiting clusters.

        Uses centroids to build inter-cluster TSP.

        Args:
            cluster_results: Results from cluster-level optimization.
            depot_coord: Depot coordinates.

        Returns:
            Ordered list of cluster indices.
        """
        n_clusters = len(cluster_results)

        if n_clusters <= 2:
            return list(range(n_clusters))

        # Build centroid distance matrix (including depot as index 0)
        centroids = [depot_coord] + [r.centroid for r in cluster_results]
        n = len(centroids)

        centroid_dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    lat1, lng1 = centroids[i]
                    lat2, lng2 = centroids[j]
                    # Approximate distance in meters
                    centroid_dist[i, j] = np.sqrt(
                        ((lat1 - lat2) * 111000) ** 2 +
                        ((lng1 - lng2) * 111000 * np.cos(np.radians(lat1))) ** 2
                    )

        # Solve cluster-level TSP using greedy
        cluster_seq, _ = self._greedy_tsp(centroid_dist)

        # Remove depot (index 0) and adjust indices
        cluster_order = [i - 1 for i in cluster_seq if i > 0]

        return cluster_order

    def merge_sequences(
        self,
        cluster_order: List[int],
        cluster_results: List[ClusterResult]
    ) -> List[int]:
        """
        Merge cluster sequences into global sequence.

        Args:
            cluster_order: Order of visiting clusters.
            cluster_results: Results from cluster-level optimization.

        Returns:
            Global sequence of node indices.
        """
        global_sequence = []

        for cluster_idx in cluster_order:
            result = cluster_results[cluster_idx]
            local_seq = result.local_sequence
            node_indices = result.node_indices

            # Map local indices to global indices
            for local_idx in local_seq:
                global_idx = node_indices[local_idx]
                global_sequence.append(global_idx)

        return global_sequence

    def optimize(
        self,
        locations: List[Tuple[float, float]],
        priorities: List[float],
        dist_matrix: np.ndarray,
        congestion_weights: np.ndarray,
        traffic_level: str = 'low',
        depot_idx: int = 0
    ) -> Tuple[List[int], float, float, Dict]:
        """
        Main entry point for hierarchical optimization.

        Args:
            locations: List of (lat, lng) coordinates.
            priorities: Priority weights for each location.
            dist_matrix: Distance matrix.
            congestion_weights: Congestion multipliers matrix.
            traffic_level: Traffic level.
            depot_idx: Index of the depot/start location.

        Returns:
            Tuple of (sequence, total_cost, solve_time, metadata).
        """
        n = len(locations)
        start_time = time.time()

        # Check if clustering needed
        if not self.should_use_clustering(n):
            if self.optimizer is not None:
                seq, cost, solve_time = self.optimizer.optimize(
                    dist_matrix, priorities, congestion_weights, traffic_level
                )
            else:
                seq, cost = self._greedy_tsp(dist_matrix)
                solve_time = time.time() - start_time

            return seq, cost, solve_time, {"method": "direct", "clusters": 0}

        logger.info(f"Using hierarchical optimization for {n} nodes")

        # Convert locations to numpy array
        coords = np.array(locations)

        # Step 1: Cluster locations (excluding depot)
        non_depot_indices = [i for i in range(n) if i != depot_idx]
        non_depot_coords = coords[non_depot_indices]

        clusters, centroids = self.cluster_locations(non_depot_coords)

        # Map cluster indices back to global indices
        clusters = [[non_depot_indices[i] for i in cluster] for cluster in clusters]

        cluster_sizes = [len(c) for c in clusters]
        logger.info(f"Created {len(clusters)} clusters: {cluster_sizes}")

        # Step 2: Solve TSP within each cluster
        cluster_results = []
        for i, cluster_indices in enumerate(clusters):
            result = self.solve_cluster_tsp(
                cluster_indices, dist_matrix, priorities,
                congestion_weights, traffic_level, coords
            )
            result.cluster_id = i
            cluster_results.append(result)

        # Step 3: Optimize cluster visitation order
        depot_coord = (locations[depot_idx][0], locations[depot_idx][1])
        cluster_order = self.optimize_cluster_order(cluster_results, depot_coord)

        # Step 4: Merge sequences
        global_sequence = self.merge_sequences(cluster_order, cluster_results)

        # Prepend depot if not already there
        if global_sequence[0] != depot_idx:
            global_sequence = [depot_idx] + global_sequence

        # Calculate total cost
        total_cost = 0.0
        lambda_factor = 2.0 if traffic_level == 'high' else 0.5

        for i in range(len(global_sequence) - 1):
            src, dst = global_sequence[i], global_sequence[i + 1]
            total_cost += dist_matrix[src, dst] * (1 + lambda_factor * congestion_weights[src, dst])

        solve_time = time.time() - start_time

        metadata = {
            "method": "hierarchical",
            "clusters": len(clusters),
            "cluster_sizes": cluster_sizes,
            "cluster_times": [r.solve_time for r in cluster_results],
            "cluster_order": cluster_order,
        }

        logger.info(f"Hierarchical optimization complete: {solve_time:.2f}s, cost={total_cost:.2f}")

        return global_sequence, total_cost, solve_time, metadata


def create_hierarchical_optimizer(
    qubo_optimizer: Optional[Any] = None,
    cluster_threshold: int = 40,
    max_cluster_size: int = 40
) -> HierarchicalOptimizer:
    """
    Factory function to create a hierarchical optimizer.

    Args:
        qubo_optimizer: Optional QUBOOptimizer instance for sub-problems.
        cluster_threshold: Use clustering when n > this.
        max_cluster_size: Maximum nodes per cluster.

    Returns:
        Configured HierarchicalOptimizer instance.
    """
    config = ClusterConfig(
        cluster_threshold=cluster_threshold,
        max_cluster_size=max_cluster_size
    )
    return HierarchicalOptimizer(config=config, qubo_optimizer=qubo_optimizer)
