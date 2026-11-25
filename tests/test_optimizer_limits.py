"""Optimisers honour their limits, fail loudly, and match exact references.

#3020 (exposure caps, failed solves, mean-risk max_return, entropy pooling),
#3079 (cvxpy optimisers dropping limits), #3126 (minimum variance at a 1.0
gross cap), #3049 (robust mean-variance uncertainty scale), #3046 (alpha-risk-
cost costs and transfer coefficient), #3144 (mean-risk input types).
"""

from __future__ import annotations

import numpy as np
import pytest
from cpz_quant.portfolio.costs import CostModel
from cpz_quant.portfolio.entropy_pooling import entropy_pooling, mean_view_rows
from cpz_quant.portfolio.optimization import (
    Constraints,
    _build_matrices,
    alpha_risk_cost_optimize,
    max_diversification,
    max_sharpe,
    mean_cvar,
    min_tracking_error,
    min_variance,
    robust_mvo,
    turnover_penalized,
)
from cpz_quant.portfolio.risk_measures import RiskMeasure, compute_risk, mean_risk_optimize


def _etfs(n_days=750, seed=11):
    rng = np.random.default_rng(seed)
    eq, rates = rng.normal(0, 0.011, n_days), rng.normal(0, 0.006, n_days)
    gold, cmdty = rng.normal(0, 0.009, n_days), rng.normal(0, 0.012, n_days)
    cols = {
        "SPY": eq + rng.normal(0, 0.002, n_days) + 4e-4,
        "QQQ": 1.25 * eq + rng.normal(0, 0.005, n_days) + 5e-4,
        "EFA": 0.85 * eq + rng.normal(0, 0.005, n_days) + 2e-4,
        "EEM": 0.95 * eq + rng.normal(0, 0.009, n_days) + 1e-4,
        "IEF": -0.12 * eq + 0.55 * rates + rng.normal(0, 0.0015, n_days) + 1e-4,
        "TLT": -0.3 * eq + 1.4 * rates + rng.normal(0, 0.004, n_days) + 1.5e-4,
        "GLD": 0.05 * eq + 0.2 * rates + gold + 2e-4,
        "DBC": 0.35 * eq + cmdty + 1e-4,
    }
    return {k: v.tolist() for k, v in cols.items()}


def _gross(weights):
    return sum(abs(v) for v in weights.values())


# ── #3020: exposure caps and failed solves ───────────────────────────

LOOSE_BOUNDS_UNIT_CAP = Constraints(min_weight=-5, max_weight=5, max_gross_exposure=1.0)


@pytest.mark.parametrize("run", [
    lambda r, c: robust_mvo(r, constraints=c),
    lambda r, c: mean_cvar(r, constraints=c),
    lambda r, c: max_diversification(r, constraints=c),
    lambda r, c: turnover_penalized(r, {k: 1 / len(r) for k in r}, constraints=c),
    lambda r, c: min_tracking_error(r, {"TLT": 1.0}, constraints=c),
], ids=["robust_mvo", "mean_cvar", "max_diversification", "turnover_penalized",
        "min_tracking_error"])
def test_gross_exposure_cap_binds(run):
    # robust_mvo came back at 30x gross under a 1.0 cap on this data.
    result = run(_etfs(), LOOSE_BOUNDS_UNIT_CAP)
    assert _gross(result.weights) <= 1.0 + 1e-6
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-7)


@pytest.mark.parametrize("run", [
    lambda r, c: robust_mvo(r, constraints=c),
    lambda r, c: mean_cvar(r, constraints=c),
    lambda r, c: max_diversification(r, constraints=c),
    lambda r, c: min_tracking_error(r, {"TLT": 1.0}, constraints=c),
], ids=["robust_mvo", "mean_cvar", "max_diversification", "min_tracking_error"])
@pytest.mark.parametrize("limit", [
    {"sector_limits": {"GLD": 0.05}},
    {"factor_limits": {"market": (0.0, 0.5)}},
    {"max_tracking_error": 0.01},
])
def test_unsupported_limits_raise(run, limit):
    with pytest.raises(NotImplementedError, match=next(iter(limit))):
        run(_etfs(), Constraints(long_only=True, **limit))


def test_mean_cvar_unreachable_target_raises():
    # A 60% target returned a 27% portfolio with no error.
    rets = _etfs()
    with pytest.raises(RuntimeError, match="target_return 0.600000") as err:
        mean_cvar(rets, target_return=0.60, constraints=Constraints(long_only=True))
    # Long-only and fully invested: the best attainable return is the top asset's.
    top = max(np.mean(v) * 252 for v in rets.values())
    assert f"allow is {top:.6f}" in str(err.value)


def test_mean_cvar_matches_the_exact_cvxpy_program():
    cvx = pytest.importorskip("cpz_quant.portfolio.convex")
    if not cvx.has_cvxpy():
        pytest.skip("cvxpy not installed")
    rets = _etfs()
    c = Constraints(long_only=True, max_weight=0.4)
    ours = mean_cvar(rets, confidence=0.95, target_return=0.05, constraints=c)
    ref = cvx.mean_cvar_cvx(rets, alpha=0.95, target_return=0.05, constraints=c)
    assert ours.info["cvar_daily"] == pytest.approx(ref.info["cvar_daily"], rel=1e-4)
    assert np.array(list(ours.weights.values())) @ _build_matrices(rets)[1] >= 0.05 - 1e-7


def test_failed_solve_is_not_returned_as_a_portfolio(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "cpz_quant.portfolio.optimization.sp_opt.minimize",
        lambda *a, **k: SimpleNamespace(success=False, x=np.ones(16) / 8, message="Iteration limit"),
    )
    with pytest.raises(RuntimeError, match="Iteration limit"):
        robust_mvo(_etfs(), constraints=Constraints(long_only=True))


def test_mean_risk_max_return_needs_and_honours_a_risk_limit():
    R = np.column_stack(list(_etfs().values()))
    with pytest.raises(ValueError, match="max_risk"):
        mean_risk_optimize(R, objective="max_return")
    limit = 0.015
    r = mean_risk_optimize(R, objective="max_return", risk_measure=RiskMeasure.CVAR,
                           max_risk=limit)
    w = np.array(list(r.weights.values()))
    assert compute_risk(R @ w, RiskMeasure.CVAR) <= limit * (1 + 1e-6)
    assert max(w) < 0.99  # no longer everything in the single best asset


def test_entropy_pooling_meets_small_daily_views():
    # Views on daily means of 1e-4 were missed by up to 8% with converged=True.
    R = np.column_stack(list(_etfs().values()))
    views = {0: 2e-4, 5: 1e-4}
    A, b = mean_view_rows(R, views)
    res = entropy_pooling(n_scenarios=R.shape[0], a_eq=A, b_eq=b)
    assert res.converged
    np.testing.assert_allclose(res.posterior @ R[:, list(views)], b, rtol=1e-6)
    assert res.max_view_error <= 1e-9


def test_entropy_pooling_reports_an_impossible_view():
    R = np.column_stack(list(_etfs().values()))
    A, b = mean_view_rows(R, {0: 0.05})  # above every scenario
    res = entropy_pooling(n_scenarios=R.shape[0], a_eq=A, b_eq=b)
    assert not res.converged
    assert res.max_view_error > 1e-3


# ── #3079: cvxpy optimisers ──────────────────────────────────────────


def _cvx():
    cvx = pytest.importorskip("cpz_quant.portfolio.convex")
    if not cvx.has_cvxpy():
        pytest.skip("cvxpy not installed")
    return cvx


@pytest.mark.parametrize("name", ["mean_variance_cvx", "mean_cvar_cvx", "robust_mean_variance_cvx"])
@pytest.mark.parametrize("limit", [
    {"sector_limits": {"GLD": 0.05}},
    {"factor_limits": {"market": (0.0, 0.5)}},
    {"max_tracking_error": 0.01},
])
def test_cvx_optimisers_do_not_drop_limits(name, limit):
    # They used to return the no-limit weights exactly.
    fn = getattr(_cvx(), name)
    with pytest.raises(NotImplementedError, match=next(iter(limit))):
        fn(_etfs(), constraints=Constraints(long_only=True, max_weight=0.45, **limit))


@pytest.mark.parametrize("name", ["mean_variance_cvx", "mean_cvar_cvx", "robust_mean_variance_cvx"])
def test_cvx_positive_min_weight_applies_when_long_only(name):
    # Only w >= 0 was added for long_only, so a 10% floor was ignored.
    r = getattr(_cvx(), name)(_etfs(), constraints=Constraints(long_only=True, min_weight=0.1))
    assert min(r.weights.values()) >= 0.1 - 1e-5  # solver accuracy


def test_cvx_cardinality_rejects_a_positive_min_weight():
    with pytest.raises(ValueError, match="min_position"):
        _cvx().cardinality_constrained_cvx(_etfs(), max_assets=3,
                                           constraints=Constraints(long_only=True, min_weight=0.1))


def test_cvx_net_exposure_below_full_investment_raises():
    with pytest.raises(ValueError, match="max_net_exposure"):
        _cvx().mean_variance_cvx(_etfs(), constraints=Constraints(max_net_exposure=0.5))


# ── #3049: robust mean-variance uncertainty scale ────────────────────


@pytest.mark.parametrize("uncertainty", ["ellipsoidal", "box"])
def test_robust_penalty_is_one_standard_error_of_the_annual_mean(uncertainty):
    cvx = _cvx()
    rets = {k: v for k, v in _etfs().items() if k in ("SPY", "TLT", "GLD", "EEM", "DBC")}
    r = cvx.robust_mean_variance_cvx(rets, uncertainty=uncertainty, kappa=1.0)
    ids = list(rets)
    w = np.array([r.weights[k] for k in ids])
    R = np.column_stack([rets[k] for k in ids])
    t = R.shape[0]
    # Hand computation: mu = 252 * daily mean, so SE(annual mean) =
    # 252 * sd(daily) / sqrt(T). The old cov / T scale was sqrt(252) too small.
    if uncertainty == "ellipsoidal":
        expected = 252 * np.std(R @ w, ddof=1) / np.sqrt(t)
    else:
        expected = np.abs(w) @ (252 * np.std(R, axis=0, ddof=1) / np.sqrt(t))
    assert r.info["worst_case_penalty"] == pytest.approx(expected, rel=1e-4)


# ── #3126: minimum variance at a 1.0 gross cap ───────────────────────


def _cvxpy_min_variance(rets):
    cp = pytest.importorskip("cvxpy")
    _, _, cov = _build_matrices(rets)
    w = cp.Variable(cov.shape[0])
    cp.Problem(cp.Minimize(cp.quad_form(w, cov)), [cp.sum(w) == 1, w >= 0]).solve()
    return float(np.sqrt(w.value @ cov @ w.value) * 100)


def test_min_variance_reaches_the_true_minimum_in_any_column_order():
    rets = _etfs()
    true_vol = _cvxpy_min_variance(rets)
    c = Constraints(long_only=True, max_gross_exposure=1.0)
    rng = np.random.default_rng(1)
    for _ in range(8):
        shuffled = {k: rets[k] for k in rng.permutation(list(rets))}
        # Previously 2.51% vs 1.44% here, and 0 of 20 orders reached the minimum.
        assert min_variance(shuffled, constraints=c).volatility == pytest.approx(true_vol, abs=1e-4)


def test_max_sharpe_unchanged_by_a_redundant_unit_gross_cap():
    rets = _etfs()
    free = max_sharpe(rets, constraints=Constraints(long_only=True, max_gross_exposure=None))
    capped = max_sharpe(rets, constraints=Constraints(long_only=True, max_gross_exposure=1.0))
    assert capped.sharpe_ratio == pytest.approx(free.sharpe_ratio, abs=1e-6)


# ── #3046: alpha-risk-cost ───────────────────────────────────────────


def _arc_inputs(seed=7, n=8):
    rng = np.random.default_rng(seed)
    ids = [f"A{i}" for i in range(n)]
    B = rng.normal(0, 0.15, (n, 3))
    cov = B @ B.T + np.diag(rng.uniform(0.01, 0.04, n))
    alphas = dict(zip(ids, rng.normal(0.01, 0.02, n)))
    current = {ids[i]: 0.25 for i in range(4)}
    return ids, cov, alphas, current


def _cvxpy_arc(ids, cov, alphas, current, rate, risk_aversion=2.0):
    cp = pytest.importorskip("cvxpy")
    a = np.array([alphas[i] for i in ids])
    w0 = np.array([current.get(i, 0.0) for i in ids])
    w = cp.Variable(len(ids))
    objective = a @ w - 0.5 * risk_aversion * cp.quad_form(w, cov) - rate * cp.norm1(w - w0)
    cp.Problem(cp.Maximize(objective), [cp.sum(w) == 1, w >= 0]).solve()
    return w.value


@pytest.mark.parametrize("model,rate", [
    (None, 0.0),
    (CostModel(spread_bps=5, impact_coefficient=0), 2.5e-4),
    (CostModel(spread_bps=300, commission_bps=50, impact_coefficient=0), 0.02),
])
def test_arc_costs_shape_the_trade_like_cvxpy(model, rate):
    # Every cost model gave the same weights (each asset was charged a fixed
    # spread whatever it traded); with 200 bps per unit traded the exact
    # program trades 0.36, not 1.0.
    ids, cov, alphas, current = _arc_inputs()
    r = alpha_risk_cost_optimize(alphas, cov, current_weights=current, cost_model=model,
                                 risk_aversion=2.0, constraints=Constraints(long_only=True))
    w = np.array([r.optimal_weights[i] for i in ids])
    expected = _cvxpy_arc(ids, cov, alphas, current, rate)
    np.testing.assert_allclose(w, expected, atol=2e-5)
    w0 = np.array([current.get(i, 0.0) for i in ids])
    assert r.expected_cost == pytest.approx(100 * rate * np.abs(w - w0).sum(), abs=1e-4)


def test_arc_no_cost_model_means_zero_cost():
    ids, cov, alphas, current = _arc_inputs()
    r = alpha_risk_cost_optimize(alphas, cov, current_weights=current,
                                 constraints=Constraints(long_only=True))
    assert r.expected_cost == 0.0


def test_arc_size_dependent_impact_needs_market_inputs():
    ids, cov, alphas, current = _arc_inputs()
    with pytest.raises(ValueError, match="adv"):
        alpha_risk_cost_optimize(alphas, cov, current_weights=current, cost_model=CostModel())
    small_adv = alpha_risk_cost_optimize(
        alphas, cov, current_weights=current, cost_model=CostModel(), risk_aversion=2.0,
        constraints=Constraints(long_only=True), portfolio_value=1e9,
        adv={i: 2e6 for i in ids}, daily_volatility={i: 0.02 for i in ids})
    free = alpha_risk_cost_optimize(alphas, cov, current_weights=current, risk_aversion=2.0,
                                    constraints=Constraints(long_only=True))

    def turnover(r):
        return sum(abs(r.optimal_weights[i] - current.get(i, 0.0)) for i in ids)

    assert turnover(small_adv) < turnover(free) - 0.1


def test_arc_transfer_coefficient_is_a_correlation():
    ids, cov, alphas, current = _arc_inputs()
    a = np.array([alphas[i] for i in ids])
    r = alpha_risk_cost_optimize(alphas, cov, constraints=Constraints(long_only=True))
    w = np.array([r.optimal_weights[i] for i in ids])
    # Clarke, de Silva & Thorley: corr(Sigma^1/2 w, Sigma^-1/2 alpha)
    # = alpha'w / (sqrt(alpha' Sigma^-1 alpha) * sqrt(w' Sigma w)).
    expected = (a @ w) / (np.sqrt(a @ np.linalg.solve(cov, a)) * np.sqrt(w @ cov @ w))
    assert r.transfer_coefficient == pytest.approx(expected, abs=1e-4)
    assert -1.0 <= r.transfer_coefficient <= 1.0
    # When the unconstrained optimum is itself fully invested (alpha =
    # lambda * Sigma * w_target), nothing is lost: TC = 1. The old formula
    # reported values such as 5.66 on comparable inputs.
    w_target = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
    ideal = dict(zip(ids, 2.0 * cov @ w_target))
    r = alpha_risk_cost_optimize(ideal, cov, risk_aversion=2.0)
    np.testing.assert_allclose([r.optimal_weights[i] for i in ids], w_target, atol=1e-5)
    assert r.transfer_coefficient == pytest.approx(1.0, abs=1e-4)


def test_arc_rejects_misaligned_inputs():
    ids, cov, alphas, current = _arc_inputs()
    with pytest.raises(ValueError, match="cov"):
        alpha_risk_cost_optimize(alphas, cov[:4, :4])
    with pytest.raises(ValueError, match="current_weights"):
        alpha_risk_cost_optimize(alphas, cov, current_weights={"ZZZ": 1.0})


# ── #3144: mean-risk accepts frames ──────────────────────────────────


@pytest.mark.parametrize("kind", ["dict", "polars", "pandas"])
def test_mean_risk_accepts_frames_and_keeps_tickers(kind):
    tickers = ["SPY", "TLT", "GLD", "EEM", "IEF"]
    all_rets = _etfs()
    data = {k: all_rets[k] for k in tickers}
    if kind == "polars":
        pl = pytest.importorskip("polars")
        data = pl.DataFrame(data)
    elif kind == "pandas":
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame(data)
    r = mean_risk_optimize(data, risk_measure=RiskMeasure.VARIANCE)
    assert list(r.weights) == tickers
    R = np.column_stack([all_rets[k] for k in tickers])
    ref = mean_risk_optimize(R, risk_measure=RiskMeasure.VARIANCE, asset_ids=tickers)
    assert r.weights == ref.weights


def test_mean_risk_rejects_missing_values():
    data = {k: list(v) for k, v in _etfs().items()}
    data["SPY"][0] = float("nan")
    with pytest.raises(ValueError, match="'SPY' at row 0"):
        mean_risk_optimize(data)
