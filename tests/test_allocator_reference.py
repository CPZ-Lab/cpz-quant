"""Hierarchical and risk-budgeting allocators against published references.

Reference weights come from ``tests/fixtures/allocator_reference.npz``
(PyPortfolioOpt and riskfolio-lib outputs on frozen returns; see
``tests/reference/generate_allocator_fixture.py``).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from cpz_quant.portfolio.optimization import (
    _build_matrices,
    _quasi_block_diag_objective,
    build_permutation_qubo,
    hierarchical_equal_risk_contribution,
    hierarchical_risk_parity,
    nested_clustered_optimization,
    quantum_inspired_hrp,
    risk_parity,
    schur_complementary_allocation,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "allocator_reference.npz"


@pytest.fixture(scope="module")
def ref():
    with np.load(FIXTURE, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


@pytest.fixture(scope="module")
def rets(ref):
    names = [str(n) for n in ref["names"]]
    return {k: ref["returns"][:, i].tolist() for i, k in enumerate(names)}


def _vector(weights, names):
    return np.array([weights[k] for k in names])


# ── HRP: Lopez de Prado (2016), #3016 ────────────────────────────────


@pytest.mark.parametrize("method", ["single", "ward"])
def test_hrp_matches_pyportfolioopt(ref, rets, method):
    # The paper splits each cluster by inverse VARIANCE: alpha = 1 - V1/(V1+V2).
    # The old inverse-volatility split was 8 to 10 points off on this data.
    w = hierarchical_risk_parity(rets, linkage_method=method).weights
    np.testing.assert_allclose(_vector(w, list(rets)), ref[f"hrp_{method}"], atol=1e-6)


def test_hrp_two_assets_is_inverse_variance():
    # Hand computation: with two assets the single split gives
    # w_1 = 1 - var_1 / (var_1 + var_2) = var_2 / (var_1 + var_2).
    rng = np.random.default_rng(4)
    a, b = rng.normal(0, 0.01, 500), rng.normal(0, 0.02, 500)
    w = hierarchical_risk_parity({"a": a.tolist(), "b": b.tolist()}).weights
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    assert w["a"] == pytest.approx(vb / (va + vb), abs=1e-6)


def test_hrp_single_asset_is_fully_invested():
    # Used to crash inside scipy's linkage on an empty distance matrix.
    r = hierarchical_risk_parity({"SPY": [0.01, -0.02, 0.005, 0.003] * 30})
    assert r.weights == {"SPY": 1.0}


def test_schur_gamma_zero_still_equals_fixed_hrp(ref, rets):
    w = schur_complementary_allocation(rets, gamma=0.0, linkage_method="single").weights
    np.testing.assert_allclose(_vector(w, list(rets)), ref["hrp_single"], atol=1e-6)


# ── HERC: Raffinot (2018), #3016 ─────────────────────────────────────


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_herc_follows_the_dendrogram_like_riskfolio(ref, rets, k):
    # Splitting the ordered cluster list in halves instead of following the
    # tree left these weights up to 36 points off at k = 5.
    r = hierarchical_equal_risk_contribution(rets, linkage_method="ward", n_clusters=k)
    np.testing.assert_allclose(_vector(r.weights, list(rets)), ref[f"herc_ward_k{k}"], atol=1e-6)
    assert r.info["n_clusters"] == k


def test_herc_rejects_impossible_cluster_count(rets):
    with pytest.raises(ValueError, match="n_clusters"):
        hierarchical_equal_risk_contribution(rets, n_clusters=99)


# ── Quantum-inspired HRP seriation, #3017 ────────────────────────────


def _three_groups(seed=5, n_days=600):
    """9 assets in 3 groups, listed so that the groups interleave."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.01, (n_days, 3))
    return {
        f"g{k % 3}_{k}": (f[:, k % 3] + rng.normal(0, 0.004 * (1 + k % 4), n_days)).tolist()
        for k in range(9)
    }


def _similarity(rets):
    _, _, cov = _build_matrices(rets)
    sd = np.sqrt(np.diag(cov))
    return np.abs(cov / np.outer(sd, sd))


def _objective_of_order(sim, order):
    positions = np.empty(len(order), dtype=int)
    positions[np.asarray(order)] = np.arange(len(order))
    return _quasi_block_diag_objective(sim, positions)


def _exact_minimum(sim):
    """min over all position vectors of sum_ij A_ij (p_i - p_j)^2, by enumeration."""
    n = sim.shape[0]
    perms = np.array(list(itertools.permutations(range(n))), dtype=float)
    best = np.inf
    for chunk in np.array_split(perms, max(1, len(perms) // 20_000)):
        diff = chunk[:, :, None] - chunk[:, None, :]
        best = min(best, float(np.min(np.einsum("ij,kij->k", sim, diff * diff))))
    return best


def test_qhrp_brute_force_uses_the_optimal_order_forwards():
    rets = _three_groups()
    sim = _similarity(rets)
    r = quantum_inspired_hrp(rets, backend="brute_force")
    order = r.info["permutation"]
    best = _exact_minimum(sim)
    # Reading the solved positions as an order (backwards) scored 222.9 vs 52.0.
    assert _objective_of_order(sim, order) == pytest.approx(best, rel=1e-12)
    assert r.info["objective_value"] == pytest.approx(best, rel=1e-12)
    groups = [int(str(list(rets)[i])[1]) for i in order]
    assert groups[0:3] == [groups[0]] * 3 and groups[3:6] == [groups[3]] * 3


def test_qhrp_heuristic_reorders_a_dense_matrix():
    rets = _three_groups()
    sim = _similarity(rets)
    order = quantum_inspired_hrp(rets, backend="heuristic").info["permutation"]
    # Reverse Cuthill-McKee returned the columns reversed (W = 199.3 against
    # an exact minimum of 52.0). The heuristic lands within 0.01% of it here
    # and keeps each group together.
    assert _objective_of_order(sim, order) == pytest.approx(_exact_minimum(sim), rel=1e-4)
    groups = [int(str(list(rets)[i])[1]) for i in order]
    assert groups[0:3] == [groups[0]] * 3 and groups[3:6] == [groups[3]] * 3


@pytest.mark.parametrize("backend", ["heuristic", "brute_force"])
def test_qhrp_weights_do_not_depend_on_column_order(backend):
    rets = _three_groups()
    base = quantum_inspired_hrp(rets, backend=backend).weights
    rng = np.random.default_rng(0)
    for _ in range(4 if backend == "heuristic" else 2):
        shuffled = {k: rets[k] for k in rng.permutation(list(rets))}
        w = quantum_inspired_hrp(shuffled, backend=backend).weights
        assert max(abs(w[k] - base[k]) for k in rets) < 1e-9


def test_qhrp_qubo_solution_is_decoded_as_positions():
    rets = {k: v for k, v in list(_three_groups().items())[:4]}
    sim = _similarity(rets)
    n = len(rets)
    best = min(itertools.permutations(range(n)),
               key=lambda p: _quasi_block_diag_objective(sim, np.array(p)))

    class OneHot:
        """Stands in for an annealer that found the optimum positions."""

        def solve(self, Q):
            assert Q.shape == build_permutation_qubo(sim).shape
            x = np.zeros(n * n)
            for asset, position in enumerate(best):
                x[asset * n + position] = 1.0
            return x

    r = quantum_inspired_hrp(rets, solver=OneHot())
    assert r.info["permutation"] == [int(i) for i in np.argsort(best)]
    assert r.info["qubo_feasible"] is True


def test_qhrp_split_is_inverse_variance():
    # Two assets: the only split gives the HRP inverse-variance weights.
    rng = np.random.default_rng(9)
    a, b = rng.normal(0, 0.01, 400), rng.normal(0, 0.03, 400)
    w = quantum_inspired_hrp({"a": a.tolist(), "b": b.tolist()}, backend="brute_force").weights
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    assert w["a"] == pytest.approx(vb / (va + vb), abs=1e-6)


# ── Risk parity, #3124 ───────────────────────────────────────────────


def _stock_bond(rho=-0.6, seed=2020):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 252)
    return {"SPY": (0.021 * z[:, 0]).tolist(), "TLT": (0.0135 * z[:, 1]).tolist()}


@pytest.mark.parametrize("rho", [-0.6, -0.9, 0.3])
def test_risk_parity_two_assets_closed_form(rho):
    # Closed form: equal risk contributions for two assets need
    # w_1 sigma_1 = w_2 sigma_2 whatever the correlation, so
    # w_SPY = sigma_TLT / (sigma_SPY + sigma_TLT). The old multiplicative
    # update took a square root of a negative contribution and gave NaN.
    pair = _stock_bond(rho)
    r = risk_parity(pair)
    s_spy, s_tlt = (np.std(pair[k], ddof=1) for k in ("SPY", "TLT"))
    assert r.weights["SPY"] == pytest.approx(s_tlt / (s_spy + s_tlt), abs=1e-9)
    assert r.weights["TLT"] == pytest.approx(s_spy / (s_spy + s_tlt), abs=1e-9)
    assert np.isfinite(r.sharpe_ratio) and r.volatility > 0


def test_risk_parity_matches_riskfolio(ref):
    names = ["EQ1", "EQ2", "BD1", "BD2", "GLD"]
    all_names = [str(n) for n in ref["names"]]
    rets = {k: ref["returns"][:, all_names.index(k)].tolist() for k in names}
    r = risk_parity(rets)
    np.testing.assert_allclose(_vector(r.weights, names), ref["risk_parity"], atol=1e-5)
    contributions = np.array(list(r.info["risk_contributions"].values()))
    np.testing.assert_allclose(contributions, 0.2, atol=1e-9)


def test_risk_parity_budget_must_cover_every_asset():
    with pytest.raises(ValueError, match="budget"):
        risk_parity(_stock_bond(), budget={"SPY": 0.5})


def test_risk_parity_budget_is_honoured():
    r = risk_parity(_stock_bond(), budget={"SPY": 0.25, "TLT": 0.75})
    assert r.info["risk_contributions"]["SPY"] == pytest.approx(0.25, abs=1e-9)


# ── NCO max-Sharpe in a falling market, #3131 ────────────────────────


def _falling_market(seed=22, n_days=252):
    rng = np.random.default_rng(seed)
    eq, rates = rng.normal(0, 0.012, n_days), rng.normal(0, 0.006, n_days)
    cols = {
        "SPY": eq - 1.2e-3, "QQQ": 1.3 * eq + rng.normal(0, 0.005, n_days) - 1.6e-3,
        "EFA": 0.9 * eq + rng.normal(0, 0.005, n_days) - 8e-4,
        "IEF": -0.1 * eq + 0.6 * rates + rng.normal(0, 0.002, n_days) - 6e-4,
        "TLT": -0.3 * eq + 1.5 * rates + rng.normal(0, 0.004, n_days) - 1.1e-3,
    }
    return {k: v.tolist() for k, v in cols.items()}


def test_nco_max_sharpe_refuses_when_no_tangency_portfolio_exists():
    rets = _falling_market()
    _, mu, cov = _build_matrices(rets)
    # 1' Sigma^-1 mu < 0: normalising Sigma^-1 mu flips it into the
    # lowest-Sharpe mix, which is what NCO used to return.
    assert np.ones(len(mu)) @ np.linalg.solve(cov, mu) < 0
    with pytest.raises(ValueError, match="no fully invested maximum-Sharpe"):
        nested_clustered_optimization(rets, objective="max_sharpe", n_clusters=1)
    with pytest.raises(ValueError, match="no fully invested maximum-Sharpe"):
        nested_clustered_optimization(rets, objective="max_sharpe")


def test_nco_max_sharpe_one_cluster_is_the_tangency_portfolio():
    rng = np.random.default_rng(3)
    rets = {k: (rng.normal(0, 0.01, 500) + d).tolist()
            for k, d in zip("ABCD", (4e-4, 6e-4, 3e-4, 5e-4))}
    _, mu, cov = _build_matrices(rets)
    expected = np.linalg.solve(cov, mu)
    expected /= expected.sum()
    r = nested_clustered_optimization(rets, objective="max_sharpe", n_clusters=1)
    np.testing.assert_allclose(_vector(r.weights, list(rets)), expected, atol=1e-6)
