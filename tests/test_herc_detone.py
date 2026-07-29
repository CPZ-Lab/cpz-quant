"""Tests for HERC allocator + detoned covariance."""

from __future__ import annotations

import numpy as np
import pytest
from cpz_quant.portfolio.covariance import detone_cov
from cpz_quant.portfolio.optimization import (
    hierarchical_equal_risk_contribution,
    hierarchical_risk_parity,
)


def _two_cluster_returns(n=800, seed=1):
    """Two clusters: {A,B} co-move, {C,D} co-move, clusters near-independent."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 0.01, n)
    f2 = rng.normal(0, 0.01, n)
    a = f1 + rng.normal(0, 0.003, n)
    b = f1 + rng.normal(0, 0.003, n)
    c = f2 + rng.normal(0, 0.003, n)
    d = f2 + rng.normal(0, 0.003, n)
    return {"A": a.tolist(), "B": b.tolist(), "C": c.tolist(), "D": d.tolist()}


class TestHERC:
    def test_valid_allocation(self):
        r = hierarchical_equal_risk_contribution(_two_cluster_returns())
        w = np.array(list(r.weights.values()))
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= -1e-9).all()
        assert r.info["n_clusters"] >= 2

    def test_balances_across_clusters(self):
        # Each 2-asset cluster should receive a comparable total to the other.
        r = hierarchical_equal_risk_contribution(_two_cluster_returns(), n_clusters=2)
        w = r.weights
        cluster1 = w["A"] + w["B"]
        cluster2 = w["C"] + w["D"]
        assert cluster1 == pytest.approx(cluster2, abs=0.2)

    def test_matches_hrp_budget(self):
        rets = _two_cluster_returns()
        herc = hierarchical_equal_risk_contribution(rets)
        hrp = hierarchical_risk_parity(rets)
        assert sum(herc.weights.values()) == pytest.approx(1.0, abs=1e-6)
        assert sum(hrp.weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_single_asset(self):
        r = hierarchical_equal_risk_contribution({"A": list(np.random.default_rng(0).normal(0, 0.01, 100))})
        assert list(r.weights.values())[0] == pytest.approx(1.0)


def _market_factor_returns(n=1000, seed=2):
    """All assets load on a common market factor + idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0, 0.01, n)
    out = {}
    for k in "ABCD":
        out[k] = (0.8 * mkt + 0.5 * rng.normal(0, 0.01, n)).tolist()
    return out


class TestDetone:
    def test_shape_and_symmetry(self):
        cov = detone_cov(_market_factor_returns(), n_detones=1, annualize=False)
        assert cov.shape == (4, 4)
        assert np.allclose(cov, cov.T, atol=1e-10)

    def test_removes_dominant_market_mode(self):
        rets = _market_factor_returns()
        from cpz_quant.portfolio.covariance import sample_cov

        def corr_of(c):
            d = np.sqrt(np.diag(c))
            return c / np.outer(d, d)

        base = corr_of(sample_cov(rets, annualize=False))
        det = corr_of(detone_cov(rets, n_detones=1, annualize=False))
        # The dominant (market) eigenvalue is removed, and residual cross-corr drops.
        assert np.max(np.linalg.eigvalsh(det)) < np.max(np.linalg.eigvalsh(base))

        def avg_abs_offdiag(corr):
            return np.mean(np.abs(corr[np.triu_indices_from(corr, 1)]))
        assert avg_abs_offdiag(det) < avg_abs_offdiag(base)
