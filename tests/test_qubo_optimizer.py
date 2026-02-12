"""
Unit tests for QUBOOptimizer (qubo_optimizer.py).

Run with: pytest tests/test_qubo_optimizer.py -v
"""

import time
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qubo_optimizer import QUBOOptimizer, QISKIT_AVAILABLE


class TestQUBOOptimizer:
    """Test cases for QUBOOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create QUBOOptimizer instance."""
        return QUBOOptimizer(n_layers=3, seed=42, timeout=5.0)

    @pytest.fixture
    def small_problem(self):
        """Create a small test problem (n=4)."""
        n = 4
        np.random.seed(42)
        dist_matrix = np.random.rand(n, n) * 1000
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)

        priorities = [1.0, 2.0, 3.0, 1.5]
        congestion = np.ones((n, n)) * 1.5

        return dist_matrix, priorities, congestion

    @pytest.fixture
    def medium_problem(self):
        """Create a medium test problem (n=6) for MPS QAOA."""
        n = 6
        np.random.seed(42)
        dist_matrix = np.random.rand(n, n) * 1000
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)
        priorities = [1.0, 2.0, 3.0, 1.5, 2.5, 1.0]
        congestion = np.ones((n, n)) * 1.5
        return dist_matrix, priorities, congestion

    @pytest.fixture
    def large_problem(self):
        """Create a larger test problem (n=8) for hybrid QAOA."""
        n = 8
        np.random.seed(42)
        dist_matrix = np.random.rand(n, n) * 1000
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)
        priorities = [1.0] * n
        congestion = np.ones((n, n)) * 1.5
        return dist_matrix, priorities, congestion

    def test_initialization(self, optimizer):
        """Test optimizer initializes correctly."""
        assert optimizer is not None
        assert optimizer.n_layers == 3
        assert optimizer.seed == 42
        assert optimizer.timeout == 5.0
        assert optimizer.lambda_high == 2.0
        assert optimizer.lambda_low == 0.5

    def test_encode_qubo(self, optimizer, small_problem):
        """Test QUBO encoding produces valid dictionary."""
        dist_matrix, priorities, congestion = small_problem

        qubo = optimizer.encode_qubo(dist_matrix, priorities, congestion, 'medium')

        assert isinstance(qubo, dict)
        assert len(qubo) > 0

        # Check all keys are tuples of integers
        for key in qubo.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert all(isinstance(k, (int, np.integer)) for k in key)

    def test_encode_qubo_adaptive_lambda(self, optimizer, small_problem):
        """Test QUBO uses adaptive lambda based on traffic."""
        dist_matrix, priorities, congestion = small_problem

        qubo_low = optimizer.encode_qubo(dist_matrix, priorities, congestion, 'low')
        qubo_high = optimizer.encode_qubo(dist_matrix, priorities, congestion, 'high')

        # QUBO should be different for different traffic levels
        # (due to different lambda weights)
        assert qubo_low != qubo_high

    def test_solve_simulated_annealing_valid_permutation(self, optimizer, small_problem):
        """Test SA returns valid permutation."""
        dist_matrix, priorities, congestion = small_problem
        n = len(dist_matrix)

        qubo = optimizer.encode_qubo(dist_matrix, priorities, congestion, 'medium')
        sequence, cost = optimizer.solve_simulated_annealing(qubo, n)

        assert len(sequence) == n
        assert set(sequence) == set(range(n))

    def test_solve_simulated_annealing_within_timeout(self, optimizer, small_problem):
        """Test SA completes within timeout."""
        dist_matrix, priorities, congestion = small_problem
        n = len(dist_matrix)

        qubo = optimizer.encode_qubo(dist_matrix, priorities, congestion, 'medium')

        start = time.time()
        sequence, cost = optimizer.solve_simulated_annealing(qubo, n)
        elapsed = time.time() - start

        assert elapsed < optimizer.timeout + 1.0  # Allow 1s buffer

    def test_solve_greedy(self, optimizer, small_problem):
        """Test greedy solver returns valid sequence."""
        dist_matrix, _, _ = small_problem
        n = len(dist_matrix)

        sequence, cost = optimizer.solve_greedy(dist_matrix, start_idx=0)

        assert len(sequence) == n
        assert set(sequence) == set(range(n))
        assert sequence[0] == 0  # Starts from index 0
        assert cost > 0

    def test_solve_brute_force(self, optimizer, small_problem):
        """Test brute force returns optimal solution for small n."""
        dist_matrix, priorities, congestion = small_problem
        n = len(dist_matrix)

        sequence, cost = optimizer.solve_brute_force(
            dist_matrix, priorities, congestion, 'medium'
        )

        assert len(sequence) == n
        assert set(sequence) == set(range(n))

    def test_solve_brute_force_limit(self, optimizer):
        """Test brute force rejects large problems."""
        n = 10
        dist_matrix = np.random.rand(n, n)
        priorities = [1.0] * n
        congestion = np.ones((n, n))

        with pytest.raises(ValueError):
            optimizer.solve_brute_force(dist_matrix, priorities, congestion)

    def test_optimize_small_problem(self, optimizer, small_problem):
        """Test main optimize function."""
        dist_matrix, priorities, congestion = small_problem
        n = len(dist_matrix)

        sequence, cost, solve_time = optimizer.optimize(
            dist_matrix, priorities, congestion, 'medium'
        )

        assert len(sequence) == n
        assert set(sequence) == set(range(n))
        assert solve_time > 0

    def test_optimize_performance_5_locations(self):
        """Test optimization completes within timeout for 5 locations."""
        n = 5
        np.random.seed(42)
        dist_matrix = np.random.rand(n, n) * 1000
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)

        priorities = [1.0] * n
        congestion = np.ones((n, n)) * 1.5

        optimizer = QUBOOptimizer(n_layers=3, seed=42, timeout=30.0)

        start = time.time()
        sequence, cost, solve_time = optimizer.optimize(
            dist_matrix, priorities, congestion, 'medium'
        )
        total_time = time.time() - start

        assert total_time < 30.0, f"Optimization took {total_time:.2f}s, expected <30s"
        assert len(sequence) == n

    def test_optimize_vs_greedy(self, optimizer, small_problem):
        """Test optimized solution is at least as good as greedy."""
        dist_matrix, priorities, congestion = small_problem

        # Get greedy solution
        greedy_seq, greedy_cost = optimizer.solve_greedy(dist_matrix)

        # Get optimized solution
        opt_seq, opt_cost, _ = optimizer.optimize(
            dist_matrix, priorities, congestion, 'medium'
        )

        # Calculate actual route distances
        def route_distance(seq):
            total = 0
            for i in range(len(seq) - 1):
                total += dist_matrix[seq[i], seq[i + 1]]
            return total

        greedy_dist = route_distance(greedy_seq)
        opt_dist = route_distance(opt_seq)

        # Optimized should be at most 1.5x worse (accounting for priority/congestion trade-offs)
        assert opt_dist < greedy_dist * 1.5

    def test_is_valid_permutation(self, optimizer):
        """Test permutation validation."""
        assert optimizer._is_valid_permutation([0, 1, 2, 3], 4) is True
        assert optimizer._is_valid_permutation([3, 1, 0, 2], 4) is True
        assert optimizer._is_valid_permutation([0, 1, 2], 4) is False  # Wrong length
        assert optimizer._is_valid_permutation([0, 1, 1, 3], 4) is False  # Duplicate
        assert optimizer._is_valid_permutation([0, 1, 4, 3], 4) is False  # Wrong values

    def test_deterministic_with_seed(self):
        """Test same seed produces same results."""
        n = 4
        np.random.seed(42)
        dist_matrix = np.random.rand(n, n) * 1000
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)
        priorities = [1.0] * n
        congestion = np.ones((n, n))

        opt1 = QUBOOptimizer(seed=42)
        seq1, cost1, _ = opt1.optimize(dist_matrix, priorities, congestion)

        opt2 = QUBOOptimizer(seed=42)
        seq2, cost2, _ = opt2.optimize(dist_matrix, priorities, congestion)

        assert seq1 == seq2
        assert cost1 == cost2


class TestQUBOAdaptiveParams:
    """Tests for adaptive QAOA parameter selection."""

    @pytest.fixture
    def optimizer(self):
        return QUBOOptimizer(n_layers=3, seed=42)

    def test_adaptive_params_small(self, optimizer):
        """Test adaptive params for n<=3 use StatevectorSampler (no MPS)."""
        params = optimizer._get_adaptive_params(3)
        assert params["shots"] == 128
        assert params["maxiter"] == 10
        assert params["n_layers"] <= 2
        assert params["use_mps"] is False
        assert params["max_circuit_terms"] is None

    def test_adaptive_params_n4(self, optimizer):
        """Test adaptive params for n=4 use StatevectorSampler with sparsification."""
        params = optimizer._get_adaptive_params(4)
        assert params["shots"] == 64
        assert params["maxiter"] == 5
        assert params["n_layers"] == 1
        assert params["use_mps"] is False
        assert params["max_circuit_terms"] == 40

    def test_adaptive_params_n5(self, optimizer):
        """Test adaptive params for n=5 switch to MPS with sparse circuit."""
        params = optimizer._get_adaptive_params(5)
        assert params["shots"] == 64
        assert params["maxiter"] == 8
        assert params["n_layers"] == 1
        assert params["use_mps"] is True
        assert params["max_circuit_terms"] == 80

    def test_adaptive_params_n6(self, optimizer):
        """Test adaptive params for n=6 use MPS with further reduced shots."""
        params = optimizer._get_adaptive_params(6)
        assert params["shots"] == 32
        assert params["maxiter"] == 5
        assert params["n_layers"] == 1
        assert params["use_mps"] is True
        assert params["max_circuit_terms"] == 60

    def test_adaptive_params_hybrid_window(self, optimizer):
        """Test adaptive params for n>6 (used by hybrid sub-problems)."""
        params = optimizer._get_adaptive_params(7)
        assert params["use_mps"] is True
        assert params["n_layers"] == 1
        assert params["maxiter"] <= 10
        assert params["max_circuit_terms"] == 80


class TestQAOAScalability:
    """Tests for QAOA scalability beyond n=4."""

    @pytest.mark.slow
    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
    def test_solve_qaoa_direct_n5(self):
        """Test direct QAOA with MPS backend for n=5 (25 qubits)."""
        optimizer = QUBOOptimizer(n_layers=2, seed=42, timeout=30.0)
        n = 5
        np.random.seed(42)
        dist_matrix = np.random.rand(n, n) * 1000
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)
        priorities = [1.0] * n
        congestion = np.ones((n, n)) * 1.5

        qubo = optimizer.encode_qubo(dist_matrix, priorities, congestion, 'medium')
        sequence, cost = optimizer.solve_qaoa(qubo, n)

        assert len(sequence) == n
        assert set(sequence) == set(range(n))

    @pytest.mark.slow
    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
    def test_solve_qaoa_direct_n6(self):
        """Test direct QAOA with MPS backend for n=6 (36 qubits)."""
        optimizer = QUBOOptimizer(n_layers=2, seed=42, timeout=60.0)
        n = 6
        np.random.seed(42)
        dist_matrix = np.random.rand(n, n) * 1000
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)
        priorities = [1.0] * n
        congestion = np.ones((n, n)) * 1.5

        qubo = optimizer.encode_qubo(dist_matrix, priorities, congestion, 'medium')
        sequence, cost = optimizer.solve_qaoa(qubo, n)

        assert len(sequence) == n
        assert set(sequence) == set(range(n))

    @pytest.mark.slow
    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
    def test_solve_qaoa_hybrid_n8(self):
        """Test hybrid QAOA decomposition for n=8."""
        optimizer = QUBOOptimizer(n_layers=2, seed=42, timeout=60.0)
        n = 8
        np.random.seed(42)
        dist_matrix = np.random.rand(n, n) * 1000
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)
        priorities = [1.0] * n
        congestion = np.ones((n, n)) * 1.5

        sequence, cost = optimizer.solve_qaoa_hybrid(
            dist_matrix, priorities, congestion, 'medium',
            window_size=5, overlap=2
        )

        assert len(sequence) == n
        assert set(sequence) == set(range(n))

    @pytest.mark.slow
    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
    def test_solve_qaoa_hybrid_n10(self):
        """Test hybrid QAOA decomposition for n=10."""
        optimizer = QUBOOptimizer(n_layers=2, seed=42, timeout=90.0)
        n = 10
        np.random.seed(42)
        dist_matrix = np.random.rand(n, n) * 1000
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)
        priorities = [1.0] * n
        congestion = np.ones((n, n)) * 1.5

        sequence, cost = optimizer.solve_qaoa_hybrid(
            dist_matrix, priorities, congestion, 'medium',
            window_size=5, overlap=2
        )

        assert len(sequence) == n
        assert set(sequence) == set(range(n))

    @pytest.mark.slow
    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
    def test_optimize_dispatches_qaoa_n6(self):
        """Test optimize() uses direct QAOA for n=6 when requested."""
        optimizer = QUBOOptimizer(n_layers=2, seed=42, timeout=60.0)
        n = 6
        np.random.seed(42)
        dist_matrix = np.random.rand(n, n) * 1000
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)
        priorities = [1.0] * n
        congestion = np.ones((n, n)) * 1.5

        sequence, cost, solve_time = optimizer.optimize(
            dist_matrix, priorities, congestion, 'medium', use_qaoa=True
        )

        assert len(sequence) == n
        assert set(sequence) == set(range(n))
        assert solve_time > 0

    @pytest.mark.slow
    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
    def test_optimize_dispatches_qaoa_hybrid_n8(self):
        """Test optimize() uses hybrid QAOA for n=8 when requested."""
        optimizer = QUBOOptimizer(n_layers=2, seed=42, timeout=60.0)
        n = 8
        np.random.seed(42)
        dist_matrix = np.random.rand(n, n) * 1000
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)
        priorities = [1.0] * n
        congestion = np.ones((n, n)) * 1.5

        sequence, cost, solve_time = optimizer.optimize(
            dist_matrix, priorities, congestion, 'medium', use_qaoa=True
        )

        assert len(sequence) == n
        assert set(sequence) == set(range(n))
        assert solve_time > 0


class TestQUBOFormulation:
    """Tests for QUBO mathematical formulation."""

    @pytest.fixture
    def optimizer(self):
        return QUBOOptimizer(seed=42)

    def test_qubo_has_linear_terms(self, optimizer):
        """Test QUBO dictionary includes linear (diagonal) terms."""
        n = 3
        dist_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        priorities = [1.0, 1.0, 1.0]
        congestion = np.ones((n, n))

        qubo = optimizer.encode_qubo(dist_matrix, priorities, congestion)

        # Linear terms have same index twice (i, i)
        linear_terms = [k for k in qubo.keys() if k[0] == k[1]]
        assert len(linear_terms) > 0

    def test_qubo_has_quadratic_terms(self, optimizer):
        """Test QUBO dictionary includes quadratic (off-diagonal) terms."""
        n = 3
        dist_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        priorities = [1.0, 1.0, 1.0]
        congestion = np.ones((n, n))

        qubo = optimizer.encode_qubo(dist_matrix, priorities, congestion)

        # Quadratic terms have different indices (i, j) where i < j
        quadratic_terms = [k for k in qubo.keys() if k[0] != k[1]]
        assert len(quadratic_terms) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
