"""Tests for NCO + Schur complementary allocation."""

from __future__ import annotations

import numpy as np
import pytest
from cpz_quant.portfolio.optimization import (
    _build_matrices,
    hierarchical_risk_parity,
    nested_clustered_optimization,
    schur_complementary_allocation,
)


def _returns(n=800, seed=1):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 0.01, n)
    f2 = rng.normal(0, 0.01, n)
    cols = {
        "A": (f1 + rng.normal(0, 0.003, n)),
        "B": (f1 + rng.normal(0, 0.004, n)),
        "C": (f2 + rng.normal(0, 0.003, n)),
        "D": (f2 + rng.normal(0, 0.005, n)),
        "E": (rng.normal(0, 0.02, n)),
    }
    return {k: v.tolist() for k, v in cols.items()}


def _port_var(returns, weights):
    ids, mu, cov = _build_matrices(returns)
    w = np.array([weights[i] for i in ids])
    return float(w @ cov @ w)


class TestNCO:
    def test_min_variance_valid_and_low_var(self):
        rets = _returns()
        r = nested_clustered_optimization(rets, objective="min_variance")
        assert sum(r.weights.values()) == pytest.approx(1.0, abs=1e-6)
        n = len(r.weights)
        eq = {k: 1.0 / n for k in r.weights}
        assert _port_var(rets, r.weights) <= _port_var(rets, eq) + 1e-12

    def test_max_sharpe_runs(self):
        r = nested_clustered_optimization(_returns(), objective="max_sharpe")
        assert sum(r.weights.values()) == pytest.approx(1.0, abs=1e-6)
        assert r.info["n_clusters"] >= 2

    def test_single_asset(self):
        r = nested_clustered_optimization({"A": list(np.random.default_rng(0).normal(0, 0.01, 100))})
        assert list(r.weights.values())[0] == pytest.approx(1.0)


class TestSchur:
    def test_gamma_zero_matches_hrp(self):
        rets = _returns()
        schur0 = schur_complementary_allocation(rets, gamma=0.0, linkage_method="single")
        hrp = hierarchical_risk_parity(rets, linkage_method="single")
        for k in hrp.weights:
            assert schur0.weights[k] == pytest.approx(hrp.weights[k], abs=1e-6)

    def test_gamma_one_differs_and_valid(self):
        rets = _returns()
        schur1 = schur_complementary_allocation(rets, gamma=1.0)
        assert sum(schur1.weights.values()) == pytest.approx(1.0, abs=1e-6)
        schur0 = schur_complementary_allocation(rets, gamma=0.0)
        diff = sum(abs(schur1.weights[k] - schur0.weights[k]) for k in schur1.weights)
        assert diff > 1e-4  # Schur augmentation changes the allocation

    def test_weights_nonnegative(self):
        r = schur_complementary_allocation(_returns(), gamma=0.5)
        assert all(w >= -1e-9 for w in r.weights.values())
