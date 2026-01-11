"""
QUBO Optimizer for Quantum Traffic Optimization.

This module provides QUBO (Quadratic Unconstrained Binary Optimization) encoding
for the TSP/VRP problem and solvers including Qiskit QAOA and classical fallbacks.
"""

import time
import warnings
from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

# Try to import Qiskit components
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.primitives import Sampler
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn("Qiskit not available. Using classical solvers only.")


class QUBOOptimizer:
    """
    QUBO formulation and QAOA solver for delivery sequencing.
    
    Encodes the Traveling Salesman Problem (TSP) as a QUBO and solves it
    using either Qiskit QAOA or classical simulated annealing.
    
    Attributes:
        n_layers: Number of QAOA layers (p parameter).
        seed: Random seed for reproducibility.
        timeout: Maximum optimization time in seconds.
        lambda_high: Congestion penalty weight for high traffic.
        lambda_low: Congestion penalty weight for low/medium traffic.
    """
    
    def __init__(
        self,
        n_layers: int = 3,
        seed: int = 42,
        timeout: float = 5.0,
        lambda_high: float = 2.0,
        lambda_low: float = 0.5
    ):
        """
        Initialize the QUBOOptimizer.
        
        Args:
            n_layers: Number of QAOA layers (default: 3).
            seed: Random seed for reproducibility (default: 42).
            timeout: Maximum solve time in seconds (default: 5.0).
            lambda_high: λ for high traffic (default: 2.0).
            lambda_low: λ for low/medium traffic (default: 0.5).
        """
        self.n_layers = n_layers
        self.seed = seed
        self.timeout = timeout
        self.lambda_high = lambda_high
        self.lambda_low = lambda_low
        
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)
    
    def encode_qubo(
        self,
        dist_matrix: np.ndarray,
        priorities: List[float],
        congestion_weights: np.ndarray,
        traffic_level: str = 'low'
    ) -> Dict[Tuple[int, int], float]:
        """
        Encode TSP as QUBO problem.
        
        Uses one-hot encoding: x_{i,p} = 1 if city i is visited at position p.
        
        Objective:
            min Σ dist[i,j] * x_{i,p} * x_{j,p+1}  (distance cost)
            + λ * Σ congestion[i,j] * x_{i,p} * x_{j,p+1}  (congestion penalty)
            + A * Σ (constraint violations)  (feasibility)
        
        Args:
            dist_matrix: Distance matrix (n x n).
            priorities: Priority weights for each location.
            congestion_weights: Congestion multipliers matrix (n x n).
            traffic_level: Traffic level for λ selection.
            
        Returns:
            QUBO dict mapping (var_i, var_j) to coefficient.
        """
        n = len(dist_matrix)
        qubo: Dict[Tuple[int, int], float] = {}
        
        # Adaptive lambda based on traffic
        lam = self.lambda_high if traffic_level == 'high' else self.lambda_low
        
        # Penalty for constraint violations (should be larger than max cost)
        max_dist = np.max(dist_matrix) if dist_matrix.size > 0 else 1000
        A = max_dist * 10  # Large penalty for constraint violations
        
        def var_idx(city: int, position: int) -> int:
            """Convert (city, position) to variable index."""
            return city * n + position
        
        # Distance objective: minimize total travel distance
        # For consecutive positions p and p+1
        for p in range(n - 1):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        idx_i = var_idx(i, p)
                        idx_j = var_idx(j, p + 1)
                        
                        # Distance cost
                        cost = dist_matrix[i, j]
                        
                        # Add congestion penalty
                        cost += lam * congestion_weights[i, j] * dist_matrix[i, j]
                        
                        # Add priority bonus (negative cost for high priority)
                        # Earlier positions get bonus for high-priority items
                        if p < n // 2:
                            cost -= priorities[j] * 0.1 * max_dist
                        
                        key = (min(idx_i, idx_j), max(idx_i, idx_j))
                        qubo[key] = qubo.get(key, 0) + cost
        
        # Constraint 1: Each city visited exactly once
        # Σ_p x_{i,p} = 1 for each city i
        for i in range(n):
            for p1 in range(n):
                idx1 = var_idx(i, p1)
                
                # Linear term: -A (to encourage x=1)
                qubo[(idx1, idx1)] = qubo.get((idx1, idx1), 0) - A
                
                # Quadratic penalty for multiple positions
                for p2 in range(p1 + 1, n):
                    idx2 = var_idx(i, p2)
                    key = (idx1, idx2)
                    qubo[key] = qubo.get(key, 0) + 2 * A
        
        # Constraint 2: Each position has exactly one city
        # Σ_i x_{i,p} = 1 for each position p
        for p in range(n):
            for i1 in range(n):
                idx1 = var_idx(i1, p)
                
                # Linear term already added above
                
                # Quadratic penalty for multiple cities at same position
                for i2 in range(i1 + 1, n):
                    idx2 = var_idx(i2, p)
                    key = (min(idx1, idx2), max(idx1, idx2))
                    qubo[key] = qubo.get(key, 0) + 2 * A
        
        return qubo
    
    def solve_qaoa(
        self,
        qubo: Dict[Tuple[int, int], float],
        n: int
    ) -> Tuple[List[int], float]:
        """
        Solve QUBO using Qiskit QAOA.
        
        Args:
            qubo: QUBO dictionary.
            n: Number of cities.
            
        Returns:
            Tuple of (best_sequence, cost).
        """
        if not QISKIT_AVAILABLE:
            return self.solve_simulated_annealing(qubo, n)
        
        n_qubits = n * n  # One qubit per (city, position) pair
        
        if n_qubits > 16:
            # Too many qubits for efficient simulation, fall back to SA
            warnings.warn(f"QAOA with {n_qubits} qubits too large, using SA")
            return self.solve_simulated_annealing(qubo, n)
        
        try:
            # Build QAOA circuit
            gamma = [Parameter(f'γ_{i}') for i in range(self.n_layers)]
            beta = [Parameter(f'β_{i}') for i in range(self.n_layers)]
            
            qc = QuantumCircuit(n_qubits)
            
            # Initial state: equal superposition
            qc.h(range(n_qubits))
            
            # QAOA layers
            for layer in range(self.n_layers):
                # Cost unitary
                for (i, j), coeff in qubo.items():
                    if i == j:
                        # Linear term: RZ rotation
                        qc.rz(2 * gamma[layer] * coeff, i)
                    else:
                        # Quadratic term: CNOT-RZ-CNOT
                        qc.cx(i, j)
                        qc.rz(2 * gamma[layer] * coeff, j)
                        qc.cx(i, j)
                
                # Mixer unitary
                for q in range(n_qubits):
                    qc.rx(2 * beta[layer], q)
            
            qc.measure_all()
            
            # Use Sampler for execution
            sampler = Sampler()
            
            def get_counts_from_result(result):
                """Extract counts from Sampler result (handles API versions)."""
                try:
                    # Qiskit 1.x+ API
                    if hasattr(result, '__getitem__'):
                        pub_result = result[0]
                        if hasattr(pub_result, 'data'):
                            data = pub_result.data
                            if hasattr(data, 'meas'):
                                return data.meas.get_counts()
                            elif hasattr(data, 'c'):
                                return data.c.get_counts()
                    # Direct counts access
                    if hasattr(result, 'quasi_dists'):
                        # Convert quasi-probabilities to counts
                        quasi = result.quasi_dists[0]
                        total = 1000
                        return {format(k, f'0{n*n}b'): int(v * total)
                                for k, v in quasi.items() if v > 0.001}
                except Exception:
                    pass
                return None

            def objective(params):
                """Objective function for classical optimizer."""
                param_dict = {}
                for i in range(self.n_layers):
                    param_dict[gamma[i]] = params[i]
                    param_dict[beta[i]] = params[self.n_layers + i]

                bound_circuit = qc.assign_parameters(param_dict)

                try:
                    job = sampler.run([bound_circuit], shots=1000)
                    result = job.result()

                    # Compute expected cost
                    counts = get_counts_from_result(result)
                    if counts is None:
                        return float('inf')

                    exp_cost = 0.0
                    total_counts = sum(counts.values())

                    for bitstring, count in counts.items():
                        cost = self._evaluate_solution(bitstring, qubo, n)
                        exp_cost += cost * count / total_counts

                    return exp_cost
                except Exception:
                    return float('inf')
            
            # Classical optimization
            n_params = 2 * self.n_layers
            initial_params = self.rng.uniform(0, np.pi, n_params)
            
            result = minimize(
                objective,
                initial_params,
                method='COBYLA',
                options={'maxiter': 50}
            )
            
            # Get best solution from final circuit
            final_params = {}
            for i in range(self.n_layers):
                final_params[gamma[i]] = result.x[i]
                final_params[beta[i]] = result.x[self.n_layers + i]
            
            bound_circuit = qc.assign_parameters(final_params)
            job = sampler.run([bound_circuit], shots=1000)
            final_result = job.result()

            counts = get_counts_from_result(final_result)
            if counts is None or len(counts) == 0:
                raise ValueError("Could not extract counts from QAOA result")
            best_bitstring = max(counts, key=lambda x: -self._evaluate_solution(x, qubo, n))
            
            sequence = self._decode_solution(best_bitstring, n)
            cost = self._compute_route_cost(sequence, qubo, n)
            
            return sequence, cost
            
        except Exception as e:
            warnings.warn(f"QAOA failed: {e}. Falling back to SA.")
            return self.solve_simulated_annealing(qubo, n)
    
    def solve_simulated_annealing(
        self,
        qubo: Dict[Tuple[int, int], float],
        n: int,
        max_iter: int = 10000,
        temp_init: float = 1000.0,
        temp_min: float = 1.0,
        cooling_rate: float = 0.995
    ) -> Tuple[List[int], float]:
        """
        Solve QUBO using classical simulated annealing.
        
        Args:
            qubo: QUBO dictionary.
            n: Number of cities.
            max_iter: Maximum iterations.
            temp_init: Initial temperature.
            temp_min: Minimum temperature.
            cooling_rate: Temperature decay rate.
            
        Returns:
            Tuple of (best_sequence, cost).
        """
        # Start with random permutation
        current = list(range(n))
        self.rng.shuffle(current)
        current_cost = self._evaluate_permutation(current, qubo, n)
        
        best = current.copy()
        best_cost = current_cost
        
        temp = temp_init
        start_time = time.time()
        
        for _ in range(max_iter):
            if time.time() - start_time > self.timeout:
                break
            
            if temp < temp_min:
                break
            
            # Generate neighbor by swapping two cities
            new = current.copy()
            i, j = self.rng.choice(n, 2, replace=False)
            new[i], new[j] = new[j], new[i]
            
            new_cost = self._evaluate_permutation(new, qubo, n)
            
            # Accept or reject
            delta = new_cost - current_cost
            if delta < 0 or self.rng.random() < np.exp(-delta / temp):
                current = new
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost
            
            temp *= cooling_rate
        
        return best, best_cost
    
    def solve_brute_force(
        self,
        dist_matrix: np.ndarray,
        priorities: List[float],
        congestion_weights: np.ndarray,
        traffic_level: str = 'low'
    ) -> Tuple[List[int], float]:
        """
        Solve by exhaustive enumeration (for n <= 8).
        
        Args:
            dist_matrix: Distance matrix.
            priorities: Priority weights.
            congestion_weights: Congestion multipliers.
            traffic_level: Traffic level.
            
        Returns:
            Tuple of (best_sequence, cost).
        """
        n = len(dist_matrix)
        
        if n > 8:
            raise ValueError(f"Brute force only for n<=8, got n={n}")
        
        lam = self.lambda_high if traffic_level == 'high' else self.lambda_low
        
        best_perm = None
        best_cost = float('inf')
        
        for perm in permutations(range(n)):
            cost = 0.0
            for i in range(len(perm) - 1):
                src, dst = perm[i], perm[i + 1]
                cost += dist_matrix[src, dst] * (1 + lam * congestion_weights[src, dst])
            
            # Add priority considerations
            for pos, city in enumerate(perm):
                if pos < n // 2:
                    cost -= priorities[city] * 0.1 * np.max(dist_matrix)
            
            if cost < best_cost:
                best_cost = cost
                best_perm = list(perm)
        
        return best_perm, best_cost
    
    def solve_greedy(
        self,
        dist_matrix: np.ndarray,
        start_idx: int = 0
    ) -> Tuple[List[int], float]:
        """
        Solve using greedy nearest neighbor heuristic.
        
        Args:
            dist_matrix: Distance matrix.
            start_idx: Starting location index.
            
        Returns:
            Tuple of (sequence, cost).
        """
        n = len(dist_matrix)
        visited = [False] * n
        sequence = [start_idx]
        visited[start_idx] = True
        total_cost = 0.0
        
        current = start_idx
        for _ in range(n - 1):
            # Find nearest unvisited
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
    
    def optimize(
        self,
        dist_matrix: np.ndarray,
        priorities: List[float],
        congestion_weights: np.ndarray,
        traffic_level: str = 'low',
        use_qaoa: bool = False
    ) -> Tuple[List[int], float, float]:
        """
        Main entry point: encode and solve with fallback.

        Strategy:
        - n <= 8: Use brute force (optimal and deterministic)
        - n > 8: Use simulated annealing
        - use_qaoa=True: Explicitly request QAOA for demonstration (slower)

        Args:
            dist_matrix: Distance matrix (n x n).
            priorities: Priority weights for each location.
            congestion_weights: Congestion multipliers matrix.
            traffic_level: Traffic level ('low', 'medium', 'high').
            use_qaoa: Whether to use QAOA (default: False, use brute force).

        Returns:
            Tuple of (best_sequence, cost, solve_time).
        """
        n = len(dist_matrix)
        start_time = time.time()

        # Encode QUBO (needed for cost evaluation in all paths)
        qubo = self.encode_qubo(dist_matrix, priorities, congestion_weights, traffic_level)

        # Strategy selection based on problem size
        # Prefer brute force for small problems (deterministic and optimal)
        if n <= 8 and not use_qaoa:
            # Use brute force for small-medium problems (guaranteed optimal)
            sequence, cost = self.solve_brute_force(
                dist_matrix, priorities, congestion_weights, traffic_level
            )
        elif use_qaoa and QISKIT_AVAILABLE and n <= 4:
            # Use QAOA only when explicitly requested (for demonstration)
            sequence, cost = self.solve_qaoa(qubo, n)
        elif n <= 8:
            # Fallback to brute force if QAOA requested but not available
            sequence, cost = self.solve_brute_force(
                dist_matrix, priorities, congestion_weights, traffic_level
            )
        else:
            # Use simulated annealing for larger problems
            sequence, cost = self.solve_simulated_annealing(qubo, n)

        solve_time = time.time() - start_time

        # Validate sequence
        if not self._is_valid_permutation(sequence, n):
            # Fallback to greedy
            sequence, _ = self.solve_greedy(dist_matrix)
            cost = self._evaluate_permutation(sequence, qubo, n)

        return sequence, cost, solve_time
    
    def _evaluate_solution(
        self,
        bitstring: str,
        qubo: Dict[Tuple[int, int], float],
        n: int
    ) -> float:
        """Evaluate QUBO cost for a bitstring."""
        cost = 0.0
        bits = [int(b) for b in bitstring[::-1]]  # Reverse for qubit ordering
        
        for (i, j), coeff in qubo.items():
            if i < len(bits) and j < len(bits):
                if i == j:
                    cost += coeff * bits[i]
                else:
                    cost += coeff * bits[i] * bits[j]
        
        return cost
    
    def _decode_solution(self, bitstring: str, n: int) -> List[int]:
        """Decode bitstring to city sequence."""
        bits = [int(b) for b in bitstring[::-1]]
        
        sequence = []
        for p in range(n):
            for i in range(n):
                idx = i * n + p
                if idx < len(bits) and bits[idx] == 1:
                    sequence.append(i)
                    break
        
        # Fill missing cities
        all_cities = set(range(n))
        in_sequence = set(sequence)
        missing = list(all_cities - in_sequence)
        
        for pos in range(n):
            if pos >= len(sequence):
                if missing:
                    sequence.append(missing.pop())
        
        return sequence
    
    def _evaluate_permutation(
        self,
        perm: List[int],
        qubo: Dict[Tuple[int, int], float],
        n: int
    ) -> float:
        """Evaluate QUBO cost for a permutation."""
        # Convert permutation to bitstring
        bits = [0] * (n * n)
        for pos, city in enumerate(perm):
            bits[city * n + pos] = 1
        
        cost = 0.0
        for (i, j), coeff in qubo.items():
            if i < len(bits) and j < len(bits):
                if i == j:
                    cost += coeff * bits[i]
                else:
                    cost += coeff * bits[i] * bits[j]
        
        return cost
    
    def _compute_route_cost(
        self,
        sequence: List[int],
        qubo: Dict[Tuple[int, int], float],
        n: int
    ) -> float:
        """Compute route cost from QUBO."""
        return self._evaluate_permutation(sequence, qubo, n)
    
    def _is_valid_permutation(self, sequence: List[int], n: int) -> bool:
        """Check if sequence is a valid permutation."""
        if len(sequence) != n:
            return False
        return set(sequence) == set(range(n))


def demo():
    """Demonstrate the QUBOOptimizer functionality."""
    print("=" * 60)
    print("QUBOOptimizer Demo")
    print("=" * 60)
    
    # Create sample distance matrix
    n = 5
    np.random.seed(42)
    
    # Symmetric distance matrix
    dist_matrix = np.random.rand(n, n) * 1000
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    np.fill_diagonal(dist_matrix, 0)
    
    print(f"\nDistance matrix ({n}x{n}):")
    print(np.round(dist_matrix, 1))
    
    # Priorities and congestion
    priorities = [1.0, 2.0, 3.0, 1.5, 2.5]
    congestion = np.ones((n, n)) * 1.5
    
    # Initialize optimizer
    optimizer = QUBOOptimizer(n_layers=3, seed=42)
    
    # Solve with different methods
    print("\n" + "-" * 40)
    
    # Brute force (optimal)
    bf_seq, bf_cost = optimizer.solve_brute_force(
        dist_matrix, priorities, congestion, 'medium'
    )
    print(f"Brute force: {bf_seq}, cost: {bf_cost:.2f}")
    
    # Greedy
    greedy_seq, greedy_cost = optimizer.solve_greedy(dist_matrix)
    print(f"Greedy: {greedy_seq}, cost: {greedy_cost:.2f}")
    
    # Main optimizer
    opt_seq, opt_cost, opt_time = optimizer.optimize(
        dist_matrix, priorities, congestion, 'medium'
    )
    print(f"Optimizer: {opt_seq}, cost: {opt_cost:.2f}, time: {opt_time:.3f}s")
    
    # Performance comparison
    print("\n" + "-" * 40)
    print("Performance comparison:")
    if greedy_cost > 0:
        improvement = (greedy_cost - opt_cost) / greedy_cost * 100
        print(f"  Improvement over greedy: {improvement:.1f}%")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    demo()
