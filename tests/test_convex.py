"""Tests for the optional cvxpy convex backend."""

from __future__ import annotations

import numpy as np
import pytest
from cpz_quant.portfolio import Constraints
from cpz_quant.portfolio.convex import (
    _MIP_SOLVERS,
    cardinality_constrained_cvx,
    has_cvxpy,
    mean_cvar_cvx,
    mean_variance_cvx,
    robust_mean_variance_cvx,
)

pytestmark = pytest.mark.skipif(not has_cvxpy(), reason="cvxpy not installed")


def _returns(n_assets: int = 4, T: int = 400, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    mu = np.linspace(0.0002, 0.001, n_assets)
    data = rng.normal(mu, 0.01, size=(T, n_assets))
    return {f"A{i}": data[:, i].tolist() for i in range(n_assets)}


def test_mean_variance_cvx_fully_invested_long_only():
    res = mean_variance_cvx(_returns(), constraints=Constraints(long_only=True))
    w = np.array(list(res.weights.values()))
    assert res.method == "mean_variance_cvx"
    assert abs(w.sum() - 1.0) < 1e-5
    assert (w >= -1e-8).all()
    assert res.volatility > 0


def test_mean_variance_cvx_target_return_binds():
    rets = _returns()
    lo = mean_variance_cvx(rets, target_return=0.02, constraints=Constraints(long_only=True))
    hi = mean_variance_cvx(rets, target_return=0.15, constraints=Constraints(long_only=True))
    # A tighter return floor cannot reduce variance.
    assert hi.volatility >= lo.volatility - 1e-6
    assert hi.expected_return >= 0.15 * 100 - 1e-3  # OptResult reports percent


def test_mean_variance_cvx_turnover_constraint():
    rets = _returns()
    prev = {k: 1.0 / 4 for k in rets}
    res = mean_variance_cvx(
        rets,
        constraints=Constraints(long_only=True, max_turnover=0.10),
        prev_weights=prev,
    )
    turnover = sum(abs(res.weights[k] - prev[k]) for k in rets)
    assert turnover <= 0.10 + 1e-5


def test_mean_variance_cvx_turnover_without_prev_raises():
    with pytest.raises(ValueError, match="prev_weights"):
        mean_variance_cvx(_returns(), constraints=Constraints(max_turnover=0.1))


def test_mean_cvar_cvx_basic():
    res = mean_cvar_cvx(_returns(), alpha=0.95, constraints=Constraints(long_only=True))
    w = np.array(list(res.weights.values()))
    assert abs(w.sum() - 1.0) < 1e-5
    assert res.info["cvar_daily"] >= res.info["var_daily"] - 1e-9
    assert 0 < res.info["alpha"] < 1


def test_mean_cvar_cvx_bad_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        mean_cvar_cvx(_returns(), alpha=0.3)


@pytest.mark.parametrize("uncertainty", ["ellipsoidal", "box"])
def test_robust_mean_variance_cvx(uncertainty):
    res = robust_mean_variance_cvx(
        _returns(), uncertainty=uncertainty, kappa=1.5,
        constraints=Constraints(long_only=True),
    )
    w = np.array(list(res.weights.values()))
    assert abs(w.sum() - 1.0) < 1e-5
    assert res.info["worst_case_penalty"] >= 0


def test_robust_mean_variance_cvx_bad_set_raises():
    with pytest.raises(ValueError, match="uncertainty"):
        robust_mean_variance_cvx(_returns(), uncertainty="banana")


def test_robust_kappa_zero_matches_plain_mvo():
    rets = _returns()
    plain = mean_variance_cvx(rets, constraints=Constraints(long_only=True))
    robust = robust_mean_variance_cvx(rets, kappa=0.0, constraints=Constraints(long_only=True))
    for k in rets:
        assert abs(plain.weights[k] - robust.weights[k]) < 1e-3


def _mip_available() -> bool:
    import cvxpy as cp

    return any(s in cp.installed_solvers() for s in _MIP_SOLVERS)


@pytest.mark.skipif(not has_cvxpy() or not _mip_available(), reason="no MIP solver")
def test_cardinality_constrained_cvx():
    res = cardinality_constrained_cvx(_returns(n_assets=6), max_assets=3, min_position=0.05)
    held = [w for w in res.weights.values() if w > 1e-6]
    assert len(held) <= 3
    assert all(w >= 0.05 - 1e-5 for w in held)
    assert res.info["n_selected"] == len(held)


def test_cardinality_without_mip_raises_cleanly():
    if _mip_available():
        pytest.skip("MIP solver installed; failure path not reachable")
    with pytest.raises(RuntimeError, match="mixed-integer"):
        cardinality_constrained_cvx(_returns(), max_assets=2)


def test_cardinality_bad_args_raise():
    if not _mip_available():
        pytest.skip("needs MIP solver to reach validation")
    with pytest.raises(ValueError, match="max_assets"):
        cardinality_constrained_cvx(_returns(n_assets=4), max_assets=9)
