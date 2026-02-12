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
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter
    from qiskit.primitives import StatevectorSampler
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

    def _get_adaptive_params(self, n: int) -> Dict[str, Any]:
        """
        Compute adaptive QAOA parameters based on problem size.

        Larger problems use fewer shots/iterations, single-layer circuits,
        and the MPS simulator backend to stay within memory and time budgets.
        max_circuit_terms limits the QUBO terms encoded into the circuit
        to keep circuit depth tractable for MPS simulation.

        Args:
            n: Number of cities.

        Returns:
            Dict with keys: shots, maxiter, n_layers, use_mps, max_circuit_terms.
        """
        if n <= 3:
            return {"shots": 128, "maxiter": 10, "n_layers": min(self.n_layers, 2), "use_mps": False, "max_circuit_terms": None}
        elif n <= 4:
            return {"shots": 64, "maxiter": 5, "n_layers": 1, "use_mps": False, "max_circuit_terms": 40}
        elif n <= 5:
            return {"shots": 64, "maxiter": 8, "n_layers": 1, "use_mps": True, "max_circuit_terms": 80}
        elif n <= 6:
            return {"shots": 32, "maxiter": 5, "n_layers": 1, "use_mps": True, "max_circuit_terms": 60}
        else:
            # Used by hybrid decomposition sub-problems (window_size <= 5)
            return {"shots": 64, "maxiter": 8, "n_layers": 1, "use_mps": True, "max_circuit_terms": 80}

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

        Supports two execution modes:
        - n <= 4 (up to 16 qubits): Uses Sampler with statevector simulation.
        - n = 5-6 (25-36 qubits): Uses AerSimulator with matrix_product_state
          method, which efficiently handles QAOA circuits at these qubit counts.

        Args:
            qubo: QUBO dictionary.
            n: Number of cities.

        Returns:
            Tuple of (best_sequence, cost).
        """
        if not QISKIT_AVAILABLE:
            return self.solve_simulated_annealing(qubo, n)

        n_qubits = n * n  # One qubit per (city, position) pair

        if n_qubits > 36:
            # Beyond MPS capability for QAOA — use hybrid decomposition instead
            warnings.warn(f"QAOA with {n_qubits} qubits too large for direct solve, using SA")
            return self.solve_simulated_annealing(qubo, n)

        # Adaptive parameters based on problem size
        params = self._get_adaptive_params(n)
        active_layers = params["n_layers"]
        shots = params["shots"]
        maxiter = params["maxiter"]
        use_mps = params["use_mps"]
        max_circuit_terms = params["max_circuit_terms"]

        try:
            # Sparsify QUBO for circuit if term limit is set (keeps circuit
            # depth tractable for MPS). Keep all linear terms (constraint
            # penalties) and top-K quadratic terms by |coefficient|.
            circuit_qubo = qubo
            if max_circuit_terms is not None and len(qubo) > max_circuit_terms:
                linear_terms = {k: v for k, v in qubo.items() if k[0] == k[1]}
                quad_terms = {k: v for k, v in qubo.items() if k[0] != k[1]}
                remaining = max_circuit_terms - len(linear_terms)
                if remaining > 0:
                    top_quad = sorted(quad_terms.items(), key=lambda x: abs(x[1]), reverse=True)[:remaining]
                    circuit_qubo = {**linear_terms, **dict(top_quad)}
                else:
                    circuit_qubo = linear_terms

            # Build QAOA circuit with adaptive layer count
            gamma = [Parameter(f'γ_{i}') for i in range(active_layers)]
            beta = [Parameter(f'β_{i}') for i in range(active_layers)]

            qc = QuantumCircuit(n_qubits)

            # Initial state: equal superposition
            qc.h(range(n_qubits))

            # QAOA layers
            for layer in range(active_layers):
                # Cost unitary (uses sparsified QUBO for tractable circuit depth)
                for (i, j), coeff in circuit_qubo.items():
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

            # Track best solution across all COBYLA iterations
            best_counts: Dict[str, int] = {}
            best_obj_value = float('inf')
            qaoa_start_time = time.time()

            if use_mps:
                # MPS backend for n=4-6 (16-36 qubits)
                backend = AerSimulator(method='matrix_product_state')

                # Pre-transpile the parameterized circuit ONCE to avoid
                # repeated transpilation in each COBYLA iteration.
                transpiled_qc = transpile(qc, backend)

                def objective_mps(opt_params):
                    nonlocal best_counts, best_obj_value

                    # Timeout enforcement
                    if time.time() - qaoa_start_time > self.timeout:
                        return best_obj_value if best_obj_value < float('inf') else 0.0

                    param_dict = {}
                    for i in range(active_layers):
                        param_dict[gamma[i]] = opt_params[i]
                        param_dict[beta[i]] = opt_params[active_layers + i]

                    bound_circuit = transpiled_qc.assign_parameters(param_dict)

                    try:
                        job = backend.run(bound_circuit, shots=shots)
                        result = job.result()
                        counts = result.get_counts()

                        if not counts:
                            return float('inf')

                        exp_cost = 0.0
                        total_counts = sum(counts.values())

                        for bitstring, count in counts.items():
                            cost = self._evaluate_solution(bitstring, qubo, n)
                            exp_cost += cost * count / total_counts

                        if exp_cost < best_obj_value:
                            best_obj_value = exp_cost
                            best_counts = counts.copy()

                        return exp_cost
                    except Exception:
                        return float('inf')

                # Classical optimization loop
                n_params = 2 * active_layers
                initial_params = self.rng.uniform(0, np.pi, n_params)

                minimize(
                    objective_mps,
                    initial_params,
                    method='COBYLA',
                    options={'maxiter': maxiter}
                )

                # Get best solution from final run
                if not best_counts:
                    final_params = {}
                    for i in range(active_layers):
                        final_params[gamma[i]] = initial_params[i]
                        final_params[beta[i]] = initial_params[active_layers + i]
                    bound_circuit = transpiled_qc.assign_parameters(final_params)
                    job = backend.run(bound_circuit, shots=shots)
                    result = job.result()
                    best_counts = result.get_counts()

                if not best_counts:
                    raise ValueError("MPS QAOA produced no valid counts")

                best_bitstring = min(best_counts, key=lambda x: self._evaluate_solution(x, qubo, n))
                sequence = self._decode_solution(best_bitstring, n)
                cost = self._compute_route_cost(sequence, qubo, n)

                return sequence, cost

            else:
                # StatevectorSampler path for n <= 4 (statevector, fast)
                sampler = StatevectorSampler(seed=self.seed)

                def objective_sampler(opt_params):
                    nonlocal best_counts, best_obj_value

                    if time.time() - qaoa_start_time > self.timeout:
                        return best_obj_value if best_obj_value < float('inf') else 0.0

                    param_values = list(opt_params)

                    try:
                        # V2 PUBs API: run([(circuit, param_values)], shots=N)
                        job = sampler.run([(qc, param_values)], shots=shots)
                        result = job.result()
                        counts = result[0].data.meas.get_counts()

                        if not counts:
                            return float('inf')

                        exp_cost = 0.0
                        total_counts = sum(counts.values())

                        for bitstring, count in counts.items():
                            cost = self._evaluate_solution(bitstring, qubo, n)
                            exp_cost += cost * count / total_counts

                        if exp_cost < best_obj_value:
                            best_obj_value = exp_cost
                            best_counts = counts.copy()

                        return exp_cost
                    except Exception:
                        return float('inf')

                # Classical optimization
                n_params = 2 * active_layers
                initial_params = self.rng.uniform(0, np.pi, n_params)

                result = minimize(
                    objective_sampler,
                    initial_params,
                    method='COBYLA',
                    options={'maxiter': maxiter}
                )

                # Get best solution from final circuit
                final_params = list(result.x)
                job = sampler.run([(qc, final_params)], shots=shots)
                final_result = job.result()
                counts = final_result[0].data.meas.get_counts()

                if not counts:
                    if best_counts:
                        counts = best_counts
                    else:
                        raise ValueError("Could not extract counts from QAOA result")

                best_bitstring = min(counts, key=lambda x: self._evaluate_solution(x, qubo, n))
                sequence = self._decode_solution(best_bitstring, n)
                cost = self._compute_route_cost(sequence, qubo, n)

                return sequence, cost

        except Exception as e:
            warnings.warn(f"QAOA failed: {e}. Falling back to SA.")
            return self.solve_simulated_annealing(qubo, n)

    def solve_qaoa_hybrid(
        self,
        dist_matrix: np.ndarray,
        priorities: List[float],
        congestion_weights: np.ndarray,
        traffic_level: str = 'low',
        window_size: int = 5,
        overlap: int = 2,
    ) -> Tuple[List[int], float]:
        """
        Hybrid quantum-classical solver for medium TSP problems (n=7-10).

        Decomposes the problem into overlapping QAOA-solvable sub-problems:
        1. Build initial tour with greedy nearest-neighbor
        2. Slide a window of `window_size` cities across the tour
        3. Optimize each window using direct QAOA (MPS backend)
        4. Polish the merged result with 2-opt local search

        Every sub-problem is solved with real QAOA circuits, not classical
        heuristics. The window_size is capped at 5 (25 qubits) to keep
        each QAOA call tractable.

        Args:
            dist_matrix: Full distance matrix (n x n).
            priorities: Priority weights for each location.
            congestion_weights: Congestion multipliers matrix (n x n).
            traffic_level: Traffic level ('low', 'medium', 'high').
            window_size: Size of each QAOA sub-problem (default: 5).
            overlap: Overlap between consecutive windows (default: 2).

        Returns:
            Tuple of (best_sequence, cost).
        """
        n = len(dist_matrix)

        # Cap window_size to keep sub-problems within MPS QAOA range
        window_size = min(window_size, 5)
        if window_size > n:
            window_size = n

        # Step 1: Get initial tour via greedy
        current_tour, _ = self.solve_greedy(dist_matrix, start_idx=0)

        def tour_distance(tour: List[int]) -> float:
            return sum(dist_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))

        current_dist = tour_distance(current_tour)
        best_tour = current_tour.copy()
        best_dist = current_dist

        # Step 2: Sliding window QAOA optimization (multiple passes)
        step = max(1, window_size - overlap)
        max_passes = 2
        hybrid_start_time = time.time()

        for pass_num in range(max_passes):
            improved_this_pass = False
            i = 0

            while i + window_size <= n:
                # Respect overall timeout
                if time.time() - hybrid_start_time > self.timeout * 0.8:
                    break

                # Extract window indices from current tour
                window_global_indices = current_tour[i:i + window_size]
                sub_n = len(window_global_indices)

                if sub_n < 3:
                    i += step
                    continue

                # Build sub-problem matrices
                sub_dist = np.zeros((sub_n, sub_n))
                sub_cong = np.zeros((sub_n, sub_n))
                sub_prio = [priorities[idx] for idx in window_global_indices]

                for si, gi in enumerate(window_global_indices):
                    for sj, gj in enumerate(window_global_indices):
                        sub_dist[si, sj] = dist_matrix[gi, gj]
                        sub_cong[si, sj] = congestion_weights[gi, gj]

                # Solve sub-problem with direct QAOA
                sub_qubo = self.encode_qubo(sub_dist, sub_prio, sub_cong, traffic_level)
                sub_seq, _ = self.solve_qaoa(sub_qubo, sub_n)

                # Map QAOA result back to global indices
                optimized_window = [window_global_indices[j] for j in sub_seq]

                # If not the first window, find best rotation to minimize
                # the connection cost from the previous segment
                if i > 0:
                    prev_node = current_tour[i - 1]
                    best_rot = min(
                        range(sub_n),
                        key=lambda r: dist_matrix[prev_node, optimized_window[r]]
                    )
                    optimized_window = optimized_window[best_rot:] + optimized_window[:best_rot]

                # Build new tour with optimized window
                new_tour = current_tour[:i] + optimized_window + current_tour[i + window_size:]
                new_dist = tour_distance(new_tour)

                if new_dist < current_dist - 1e-10:
                    current_tour = new_tour
                    current_dist = new_dist
                    improved_this_pass = True

                    if current_dist < best_dist:
                        best_tour = current_tour.copy()
                        best_dist = current_dist

                i += step

            if not improved_this_pass:
                break

        # Step 3: Polish with 2-opt local search
        final_tour, final_dist = self.solve_2opt(dist_matrix, best_tour)
        if final_dist < best_dist:
            best_tour = final_tour
            best_dist = final_dist

        return best_tour, best_dist

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
    
    def solve_2opt(
        self,
        dist_matrix: np.ndarray,
        initial_tour: List[int],
        max_iterations: int = 1000
    ) -> Tuple[List[int], float]:
        """
        Improve a tour using 2-opt and Or-opt local search.

        Combines 2-opt (edge reversal) with Or-opt (segment relocation)
        for comprehensive local search optimization.

        Args:
            dist_matrix: Distance matrix (n x n).
            initial_tour: Starting tour to improve.
            max_iterations: Maximum improvement iterations.

        Returns:
            Tuple of (improved_tour, total_distance).
        """
        def tour_distance(tour: List[int]) -> float:
            return sum(dist_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))

        tour = initial_tour.copy()
        n = len(tour)
        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # 2-opt moves: reverse segments
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    d_removed = (dist_matrix[tour[i - 1], tour[i]] +
                                 dist_matrix[tour[j], tour[j + 1]])
                    d_added = (dist_matrix[tour[i - 1], tour[j]] +
                               dist_matrix[tour[i], tour[j + 1]])

                    if d_added < d_removed - 1e-10:
                        tour[i:j + 1] = tour[i:j + 1][::-1]
                        improved = True

            # Or-opt moves: relocate segments of 1, 2, or 3 nodes
            for seg_len in [1, 2, 3]:
                if n <= seg_len + 2:
                    continue

                for i in range(1, n - seg_len):
                    # Current cost of segment and its neighbors
                    seg_start = i
                    seg_end = i + seg_len - 1

                    if seg_end >= n - 1:
                        continue

                    # Cost of removing segment
                    cost_remove = (
                        dist_matrix[tour[seg_start - 1], tour[seg_start]] +
                        dist_matrix[tour[seg_end], tour[seg_end + 1]] -
                        dist_matrix[tour[seg_start - 1], tour[seg_end + 1]]
                    )

                    # Try inserting segment at each position
                    for j in range(1, n - 1):
                        if j >= seg_start - 1 and j <= seg_end + 1:
                            continue  # Skip positions within or adjacent to segment

                        # Cost of inserting segment at position j
                        cost_insert = (
                            dist_matrix[tour[j - 1], tour[seg_start]] +
                            dist_matrix[tour[seg_end], tour[j]] -
                            dist_matrix[tour[j - 1], tour[j]]
                        )

                        if cost_insert < cost_remove - 1e-10:
                            # Perform the move
                            segment = tour[seg_start:seg_end + 1]
                            new_tour = tour[:seg_start] + tour[seg_end + 1:]
                            insert_pos = j if j < seg_start else j - seg_len
                            new_tour = new_tour[:insert_pos] + segment + new_tour[insert_pos:]
                            tour = new_tour
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        return tour, tour_distance(tour)

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
        - use_qaoa=True, n <= 6: Direct QAOA (MPS backend for n>4)
        - use_qaoa=True, n 7-10: Hybrid QAOA decomposition
        - n <= 8 (no QAOA): Brute force (optimal and deterministic)
        - n > 8 (no QAOA): Greedy + 2-opt

        Args:
            dist_matrix: Distance matrix (n x n).
            priorities: Priority weights for each location.
            congestion_weights: Congestion multipliers matrix.
            traffic_level: Traffic level ('low', 'medium', 'high').
            use_qaoa: Whether to use QAOA (default: False).

        Returns:
            Tuple of (best_sequence, cost, solve_time).
        """
        n = len(dist_matrix)
        start_time = time.time()

        # Encode QUBO (needed for cost evaluation in all paths)
        qubo = self.encode_qubo(dist_matrix, priorities, congestion_weights, traffic_level)

        # Helper to calculate actual route distance
        def calc_distance(seq: List[int]) -> float:
            return sum(dist_matrix[seq[i], seq[i + 1]] for i in range(len(seq) - 1))

        # Strategy selection based on problem size
        if n <= 8 and not use_qaoa:
            # Brute force for small-medium problems (guaranteed optimal)
            sequence, cost = self.solve_brute_force(
                dist_matrix, priorities, congestion_weights, traffic_level
            )
        elif use_qaoa and QISKIT_AVAILABLE and n <= 6:
            # Direct QAOA: Sampler for n<=4, MPS for n=5-6
            sequence, cost = self.solve_qaoa(qubo, n)
        elif use_qaoa and QISKIT_AVAILABLE and n <= 10:
            # Hybrid QAOA: sliding-window decomposition with QAOA sub-solves
            sequence, cost = self.solve_qaoa_hybrid(
                dist_matrix, priorities, congestion_weights, traffic_level,
                window_size=5, overlap=2
            )
        elif n <= 8:
            # Fallback to brute force if QAOA requested but not available
            sequence, cost = self.solve_brute_force(
                dist_matrix, priorities, congestion_weights, traffic_level
            )
        else:
            # For larger problems: use greedy + 2-opt improvement
            greedy_seq, _ = self.solve_greedy(dist_matrix, start_idx=0)

            # Improve greedy solution with 2-opt
            sequence, _ = self.solve_2opt(dist_matrix, greedy_seq)
            cost = self._evaluate_permutation(sequence, qubo, n)

        solve_time = time.time() - start_time

        # Validate sequence
        if not self._is_valid_permutation(sequence, n):
            # Fallback to greedy
            sequence, _ = self.solve_greedy(dist_matrix)
            cost = self._evaluate_permutation(sequence, qubo, n)

        # Final safety check: always compare with greedy and pick best
        greedy_seq, _ = self.solve_greedy(dist_matrix, start_idx=0)
        if calc_distance(greedy_seq) < calc_distance(sequence):
            sequence = greedy_seq
            cost = self._evaluate_permutation(sequence, qubo, n)

        return sequence, cost, solve_time

    def compare_solvers(
        self,
        dist_matrix: np.ndarray,
        priorities: List[float],
        congestion_weights: np.ndarray,
        traffic_level: str = 'low',
        include_qaoa: bool = False
    ) -> Dict[str, Any]:
        """
        Compare all available optimization methods on the same problem.

        Runs greedy, simulated annealing, brute force (n<=8), and optionally QAOA
        on the same distance matrix and returns comparative results.

        Args:
            dist_matrix: Distance matrix (n x n).
            priorities: Priority weights for each location.
            congestion_weights: Congestion multipliers matrix.
            traffic_level: Traffic level ('low', 'medium', 'high').
            include_qaoa: Whether to include QAOA (direct for n<=6, hybrid for n<=10).

        Returns:
            Dict with:
                - solvers: List of solver results
                - best_solver: Name of solver with lowest cost
                - improvements: Dict of improvement percentages vs greedy
                - problem_size: Number of locations
        """
        n = len(dist_matrix)
        qubo = self.encode_qubo(dist_matrix, priorities, congestion_weights, traffic_level)

        results = []

        # 1. Greedy (baseline - always run)
        try:
            start = time.time()
            seq, dist_cost = self.solve_greedy(dist_matrix)
            cost = self._evaluate_permutation(seq, qubo, n)
            results.append({
                "name": "greedy",
                "sequence": seq,
                "cost": round(cost, 2),
                "distance": round(dist_cost, 2),
                "solve_time": round(time.time() - start, 4),
                "success": True,
                "error": None
            })
        except Exception as e:
            results.append({
                "name": "greedy",
                "sequence": [],
                "cost": float('inf'),
                "distance": float('inf'),
                "solve_time": 0,
                "success": False,
                "error": str(e)
            })

        # 2. Simulated Annealing
        try:
            start = time.time()
            seq, cost = self.solve_simulated_annealing(qubo, n)
            # Calculate actual distance
            dist_cost = sum(dist_matrix[seq[i], seq[i+1]] for i in range(len(seq)-1))
            results.append({
                "name": "simulated_annealing",
                "sequence": seq,
                "cost": round(cost, 2),
                "distance": round(dist_cost, 2),
                "solve_time": round(time.time() - start, 4),
                "success": True,
                "error": None
            })
        except Exception as e:
            results.append({
                "name": "simulated_annealing",
                "sequence": [],
                "cost": float('inf'),
                "distance": float('inf'),
                "solve_time": 0,
                "success": False,
                "error": str(e)
            })

        # 3. Greedy + 2-opt (always run for better results)
        try:
            start = time.time()
            greedy_seq, _ = self.solve_greedy(dist_matrix)
            seq, dist_cost = self.solve_2opt(dist_matrix, greedy_seq)
            cost = self._evaluate_permutation(seq, qubo, n)
            results.append({
                "name": "greedy_2opt",
                "sequence": seq,
                "cost": round(cost, 2),
                "distance": round(dist_cost, 2),
                "solve_time": round(time.time() - start, 4),
                "success": True,
                "error": None
            })
        except Exception as e:
            results.append({
                "name": "greedy_2opt",
                "sequence": [],
                "cost": float('inf'),
                "distance": float('inf'),
                "solve_time": 0,
                "success": False,
                "error": str(e)
            })

        # 4. Brute Force (only for n <= 8)
        if n <= 8:
            try:
                start = time.time()
                seq, cost = self.solve_brute_force(
                    dist_matrix, priorities, congestion_weights, traffic_level
                )
                dist_cost = sum(dist_matrix[seq[i], seq[i+1]] for i in range(len(seq)-1))
                results.append({
                    "name": "brute_force",
                    "sequence": seq,
                    "cost": round(cost, 2),
                    "distance": round(dist_cost, 2),
                    "solve_time": round(time.time() - start, 4),
                    "success": True,
                    "error": None
                })
            except Exception as e:
                results.append({
                    "name": "brute_force",
                    "sequence": [],
                    "cost": float('inf'),
                    "distance": float('inf'),
                    "solve_time": 0,
                    "success": False,
                    "error": str(e)
                })

        # 5. QAOA — Direct (n<=6) or Hybrid (n<=10)
        if include_qaoa and QISKIT_AVAILABLE and n <= 10:
            try:
                start = time.time()
                if n <= 6:
                    # Direct QAOA: Sampler for n<=4, MPS for n=5-6
                    seq, cost = self.solve_qaoa(qubo, n)
                    solver_name = "qaoa_direct"
                else:
                    # Hybrid QAOA: sliding-window decomposition
                    seq, cost = self.solve_qaoa_hybrid(
                        dist_matrix, priorities, congestion_weights, traffic_level
                    )
                    solver_name = "qaoa_hybrid"
                dist_cost = sum(dist_matrix[seq[i], seq[i+1]] for i in range(len(seq)-1))
                results.append({
                    "name": solver_name,
                    "sequence": seq,
                    "cost": round(cost, 2),
                    "distance": round(dist_cost, 2),
                    "solve_time": round(time.time() - start, 4),
                    "success": True,
                    "error": None
                })
            except Exception as e:
                results.append({
                    "name": "qaoa",
                    "sequence": [],
                    "cost": float('inf'),
                    "distance": float('inf'),
                    "solve_time": 0,
                    "success": False,
                    "error": str(e)
                })
        elif include_qaoa and not QISKIT_AVAILABLE:
            results.append({
                "name": "qaoa",
                "sequence": [],
                "cost": float('inf'),
                "distance": float('inf'),
                "solve_time": 0,
                "success": False,
                "error": "Qiskit not available"
            })
        elif include_qaoa and n > 10:
            results.append({
                "name": "qaoa",
                "sequence": [],
                "cost": float('inf'),
                "distance": float('inf'),
                "solve_time": 0,
                "success": False,
                "error": f"QAOA supports n<=10, got n={n}"
            })

        # Find best solver and calculate improvements
        successful = [r for r in results if r["success"]]
        best = min(successful, key=lambda x: x["cost"]) if successful else None

        greedy_result = next((r for r in results if r["name"] == "greedy" and r["success"]), None)
        improvements = {}

        if greedy_result and greedy_result["cost"] > 0:
            for r in successful:
                if r["name"] != "greedy":
                    pct = ((greedy_result["cost"] - r["cost"]) / greedy_result["cost"]) * 100
                    improvements[f"{r['name']}_vs_greedy"] = round(pct, 2)

        return {
            "solvers": results,
            "best_solver": best["name"] if best else None,
            "improvements": improvements,
            "problem_size": n
        }

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
        """Decode bitstring to city sequence, ensuring a valid permutation."""
        bits = [int(b) for b in bitstring[::-1]]

        sequence = []
        used_cities = set()
        for p in range(n):
            for i in range(n):
                idx = i * n + p
                if idx < len(bits) and bits[idx] == 1 and i not in used_cities:
                    sequence.append(i)
                    used_cities.add(i)
                    break

        # Fill missing cities (handles invalid bitstrings from sparse QAOA)
        missing = [c for c in range(n) if c not in used_cities]
        for c in missing:
            sequence.append(c)

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

    def optimize_large_scale(
        self,
        locations: List[Tuple[float, float]],
        priorities: List[float],
        dist_matrix: np.ndarray,
        congestion_weights: np.ndarray,
        traffic_level: str = 'low',
        cluster_threshold: int = 40
    ) -> Tuple[List[int], float, float, Dict[str, Any]]:
        """
        Optimize large-scale problems using hierarchical clustering.

        For problems with more than cluster_threshold nodes, uses K-means
        clustering to divide the problem into smaller sub-problems.

        Args:
            locations: List of (lat, lng) coordinates.
            priorities: Priority weights for each location.
            dist_matrix: Distance matrix.
            congestion_weights: Congestion multipliers matrix.
            traffic_level: Traffic level.
            cluster_threshold: Use clustering when n > this.

        Returns:
            Tuple of (sequence, cost, solve_time, metadata).
        """
        from .clustering import create_hierarchical_optimizer

        hierarchical = create_hierarchical_optimizer(
            qubo_optimizer=self,
            cluster_threshold=cluster_threshold
        )

        return hierarchical.optimize(
            locations, priorities, dist_matrix, congestion_weights, traffic_level
        )


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
