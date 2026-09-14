"""Tests for quantum-inspired portfolio optimisation: QHRP and QUBO selection."""

from __future__ import annotations

import numpy as np
import pytest
from cpz_quant.portfolio.optimization import (
    QUBOResult,
    build_permutation_qubo,
    build_portfolio_qubo,
    hierarchical_risk_parity,
    quantum_inspired_hrp,
    qubo_portfolio_selection,
)
from cpz_quant.portfolio.quantum_backends import (
    ClassicalBruteForceSolver,
    get_solver,
)

# ── Fixtures ─────────────────────────────────────────────────────────

SAMPLE_RETURNS = {
    "SPY": [0.01, -0.005, 0.008, 0.002, -0.003, 0.006, -0.001, 0.004, 0.003, -0.002] * 25,
    "TLT": [-0.002, 0.004, -0.001, 0.003, 0.005, -0.003, 0.002, -0.004, 0.001, 0.003] * 25,
    "GLD": [0.003, 0.001, -0.002, 0.004, 0.002, -0.001, 0.003, 0.001, -0.003, 0.002] * 25,
    "VNQ": [0.005, -0.003, 0.004, -0.001, -0.004, 0.007, -0.002, 0.003, 0.002, -0.005] * 25,
    "EFA": [0.004, -0.006, 0.005, 0.001, -0.002, 0.003, -0.003, 0.002, 0.004, -0.001] * 25,
}

SMALL_RETURNS = {
    "A": [0.01, -0.005, 0.008, 0.002, -0.003] * 50,
    "B": [-0.002, 0.004, -0.001, 0.003, 0.005] * 50,
    "C": [0.003, 0.001, -0.002, 0.004, 0.002] * 50,
}


# ── QHRP Tests ───────────────────────────────────────────────────────


class TestQuantumInspiredHRP:
    def test_heuristic_basic(self):
        result = quantum_inspired_hrp(SAMPLE_RETURNS, backend="heuristic")
        assert result.method == "quantum_inspired_hrp"
        assert len(result.weights) == 5
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert all(w >= 0 for w in result.weights.values())
        assert result.info["backend"] == "heuristic"

    def test_brute_force_small(self):
        result = quantum_inspired_hrp(SMALL_RETURNS, backend="brute_force")
        assert result.method == "quantum_inspired_hrp"
        assert len(result.weights) == 3
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert result.info["backend"] == "brute_force"

    def test_differs_from_standard_hrp(self):
        qhrp = quantum_inspired_hrp(SAMPLE_RETURNS, backend="heuristic")
        hrp = hierarchical_risk_parity(SAMPLE_RETURNS)
        assert qhrp.method == "quantum_inspired_hrp"
        assert hrp.method == "hierarchical_risk_parity"
        assert qhrp.weights != hrp.weights

    def test_single_asset(self):
        result = quantum_inspired_hrp({"SPY": [0.01, -0.005, 0.008] * 50})
        assert result.weights == {"SPY": 1.0}

    def test_custom_solver(self):
        solver = ClassicalBruteForceSolver()
        result = quantum_inspired_hrp(SMALL_RETURNS, solver=solver)
        assert result.method == "quantum_inspired_hrp"
        assert result.info["backend"] == "custom_solver"
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            quantum_inspired_hrp(SAMPLE_RETURNS, backend="nonsense")

    def test_nan_inf_returns_do_not_crash(self):
        """Regression: NaN/Inf in returns should not crash linkage."""
        dirty = {
            "A": [0.01, float("nan"), 0.008, 0.002, -0.003] * 50,
            "B": [-0.002, 0.004, float("inf"), 0.003, 0.005] * 50,
            "C": [0.003, 0.001, -0.002, 0.004, 0.002] * 50,
        }
        result = quantum_inspired_hrp(dirty, backend="heuristic")
        assert result.method == "quantum_inspired_hrp"
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert all(np.isfinite(w) for w in result.weights.values())

    def test_nan_inf_hrp_does_not_crash(self):
        """Regression: hierarchical_risk_parity must survive NaN/Inf returns."""
        dirty = {
            "X": [0.01, float("nan"), 0.008, 0.002, -0.003] * 50,
            "Y": [-0.002, 0.004, float("inf"), 0.003, 0.005] * 50,
            "Z": [0.003, 0.001, -0.002, 0.004, 0.002] * 50,
        }
        result = hierarchical_risk_parity(dirty)
        assert result.method == "hierarchical_risk_parity"
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert all(np.isfinite(w) for w in result.weights.values())

    def test_constant_returns_do_not_crash(self):
        """Zero-variance asset should not produce NaN in distance matrix."""
        flat = {
            "CONST": [0.0] * 250,
            "GOOD": [0.01, -0.005, 0.008, 0.002, -0.003] * 50,
        }
        result = hierarchical_risk_parity(flat)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        result_q = quantum_inspired_hrp(flat, backend="heuristic")
        assert abs(sum(result_q.weights.values()) - 1.0) < 1e-6


# ── QUBO Portfolio Selection Tests ───────────────────────────────────


class TestQUBOPortfolioSelection:
    def test_basic(self):
        result = qubo_portfolio_selection(
            SAMPLE_RETURNS,
            target_k=3,
            backend="brute_force",
        )
        assert isinstance(result, QUBOResult)
        assert result.method == "qubo_portfolio_selection"
        assert len(result.selected_assets) == 3
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_cardinality(self):
        for k in [1, 2, 4, 5]:
            result = qubo_portfolio_selection(
                SAMPLE_RETURNS,
                target_k=k,
                backend="brute_force",
            )
            non_zero = sum(1 for w in result.weights.values() if w > 0)
            assert non_zero == k

    def test_target_k_exceeds_n(self):
        result = qubo_portfolio_selection(
            SMALL_RETURNS,
            target_k=10,
            backend="brute_force",
        )
        assert len(result.selected_assets) <= 3

    def test_info_fields(self):
        result = qubo_portfolio_selection(
            SAMPLE_RETURNS,
            target_k=2,
            backend="brute_force",
        )
        assert result.info["target_k"] == 2
        assert result.info["actual_k"] == 2
        assert result.info["backend"] == "brute_force"

    def test_custom_solver(self):
        solver = ClassicalBruteForceSolver()
        result = qubo_portfolio_selection(SMALL_RETURNS, target_k=2, solver=solver)
        assert result.info["backend"] == "custom_solver"
        assert len(result.selected_assets) == 2

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            qubo_portfolio_selection(SAMPLE_RETURNS, backend="nonsense")


# ── QUBO Matrix Tests ────────────────────────────────────────────────


class TestBuildQUBO:
    def test_portfolio_qubo_shape(self):
        mu = np.array([0.10, 0.12, 0.07, 0.09, 0.11])
        cov = np.eye(5) * 0.04
        Q = build_portfolio_qubo(mu, cov, target_k=2)
        assert Q.shape == (5, 5)
        np.testing.assert_allclose(Q, Q.T, atol=1e-10)

    def test_permutation_qubo_shape(self):
        similarity = np.abs(np.random.randn(4, 4))
        similarity = (similarity + similarity.T) / 2.0
        Q = build_permutation_qubo(similarity)
        assert Q.shape == (16, 16)
        np.testing.assert_allclose(Q, Q.T, atol=1e-10)


# ── Solver Protocol Tests ────────────────────────────────────────────


class TestSolverProtocol:
    def test_brute_force_solver(self):
        Q = np.array([[-1, 2], [2, -1]], dtype=np.float64)
        solver = ClassicalBruteForceSolver()
        x = solver.solve(Q)
        assert x.shape == (2,)
        assert all(xi in (0, 1) for xi in x)

    def test_brute_force_too_large(self):
        Q = np.zeros((25, 25))
        solver = ClassicalBruteForceSolver()
        with pytest.raises(ValueError, match="1..20 variables"):
            solver.solve(Q)

    def test_get_solver_factory(self):
        solver = get_solver("brute_force")
        assert isinstance(solver, ClassicalBruteForceSolver)

    def test_get_solver_unknown(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_solver("quantum_unicorn")


# ── Simulated Annealing Tests (skip if not installed) ────────────────


class TestSimulatedAnnealing:
    @pytest.fixture(autouse=True)
    def _check_dwave(self):
        try:
            import dimod  # noqa: F401
            import neal  # noqa: F401
        except ImportError:
            pytest.skip("dwave-neal/dimod not installed")

    def test_sa_qubo_selection(self):
        result = qubo_portfolio_selection(
            SAMPLE_RETURNS,
            target_k=3,
            backend="simulated_annealing",
        )
        assert result.method == "qubo_portfolio_selection"
        assert len(result.selected_assets) > 0

    def test_sa_qhrp(self):
        result = quantum_inspired_hrp(
            SMALL_RETURNS,
            backend="simulated_annealing",
        )
        assert result.method == "quantum_inspired_hrp"
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6


# ── Braket Solver Tests (mocked) ─────────────────────────────────────


class TestBraketSolverMocked:
    def test_braket_local_requires_no_s3_folder(self):
        from cpz_quant.portfolio.quantum_backends import BraketSolver

        assert BraketSolver(s3_folder=None).device_kind == "local"

    def test_braket_device_aliases(self):
        from cpz_quant.portfolio.quantum_backends import BraketSolver

        for device in ("ionq", "rigetti", "sv1", "dm1", "quera"):
            with pytest.raises(NotImplementedError, match="disabled"):
                BraketSolver(device_arn=device, s3_folder=("test-bucket", "test-prefix"))

    def test_braket_cost_estimation(self):
        from cpz_quant.portfolio.quantum_backends import BraketSolver

        assert BraketSolver(device="local").estimated_cost == 0

    def test_actual_local_qaoa_and_energy_convention(self):
        pytest.importorskip("braket")
        from cpz_quant.portfolio.quantum_backends import BraketSolver, _bqm_from_qubo

        q = np.array([[-1.0, 0.3], [0.3, -0.7]])
        bqm = _bqm_from_qubo(q)
        for x in (np.array([a, b]) for a in (0, 1) for b in (0, 1)):
            assert bqm.energy(dict(enumerate(x))) == pytest.approx(x @ q @ x)
        a = BraketSolver(seed=7, shots=100, max_iter=8)
        b = BraketSolver(seed=7, shots=100, max_iter=8)
        x = a.solve(q)
        assert np.array_equal(x, b.solve(q))
        assert x @ q @ x == pytest.approx(
            min(np.array([i, j]) @ q @ np.array([i, j]) for i in (0, 1) for j in (0, 1))
        )
        assert a.last_meta.cost_source == "free"
        assert a.last_meta.n_tasks <= 8

    def test_invalid_matrix_is_rejected(self):
        for q in (np.empty((0, 0)), np.array([[float("nan")]]), np.ones((2, 3))):
            with pytest.raises(ValueError):
                ClassicalBruteForceSolver().solve(q)
