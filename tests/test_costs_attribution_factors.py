"""Cost model, turnover, capacity, attribution and factor-model fixes.

Each test names the CPZ-Lab/cpzai issue it pins down and shows the hand
computation its reference value comes from.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from cpz_quant.portfolio import (
    CostModel,
    WalkForward,
    alpha_beta_decomposition,
    alpha_decay_capacity,
    brinson_fachler,
    cross_validate,
    factor_risk_decomposition,
    fundamental_factors,
    grid_search,
    hierarchical_risk_parity,
    risk_attribution,
    rolling_attribution,
    total_cost,
    turnover_analysis,
)
from cpz_quant.portfolio.risk_measures import RiskMeasure, compute_risk

# ── #3045: turnover cost is a share of the book ─────────────────────────


def _swap_cost(weights_from, weights_to):
    return turnover_analysis(
        weights_from, weights_to, cost_model=CostModel(),
        adv={s: 100e6 for s in "ABC"}, volatility={s: 0.02 for s in "ABC"},
        portfolio_value=50e6,
    )


def test_3045_three_percent_swap_on_a_50m_book():
    res = _swap_cost({"A": 0.50, "B": 0.50}, {"A": 0.47, "B": 0.53})
    # Each leg trades $1.5M: half spread 2.5 bps + 0.1 * 0.02 * sqrt(1.5M / 100M) * 1e4
    # = 2.5 + 2.4495 = 4.9495 bps of its notional, i.e. $742.42 per leg.
    leg_bps = 2.5 + 0.1 * 0.02 * math.sqrt(1.5e6 / 100e6) * 1e4
    assert res.per_asset_cost_bps["A"] == pytest.approx(leg_bps, abs=1e-4)
    dollars = 2 * 1.5e6 * leg_bps / 1e4  # $1,484.85
    assert res.estimated_cost_usd == pytest.approx(dollars, abs=0.01)
    assert res.estimated_cost_bps == pytest.approx(dollars / 50e6 * 1e4, abs=1e-4)  # 0.297 bps


def test_3045_a_tiny_top_up_adds_its_dollar_cost_not_its_full_spread():
    base = _swap_cost({"A": 0.50, "B": 0.50}, {"A": 0.47, "B": 0.53})
    topped = _swap_cost({"A": 0.50, "B": 0.50, "C": 0.0}, {"A": 0.47, "B": 0.5299, "C": 0.0001})
    # The $5,000 top-up costs about 2.5 bps of $5,000 = $1.25 plus a little impact.
    assert topped.estimated_cost_usd - base.estimated_cost_usd < 3.0
    assert topped.estimated_cost_bps - base.estimated_cost_bps < 0.01


# ── #3070: cost model semantics ──────────────────────────────────────────


def test_3070_estimate_is_one_leg_and_says_so():
    cm = CostModel(spread_bps=8, commission_bps=1)
    one_leg = 4.0 + 0.1 * 0.02 * math.sqrt(2e6 / 80e6) * 1e4 + 1.0  # 8.1623 bps
    assert cm.estimate(2e6, 80e6, 0.02) == pytest.approx(one_leg, rel=1e-12)
    assert "one trade (one leg)" in CostModel.estimate.__doc__
    assert "round trip" in CostModel.estimate.__doc__


def test_3070_permanent_fraction_splits_the_impact():
    impact = 0.1 * 0.02 * math.sqrt(2e6 / 80e6) * 1e4
    for pf in (0.0, 0.5, 1.0):
        b = CostModel(permanent_fraction=pf).breakdown(2e6, 80e6, 0.02)
        assert b.permanent_cost_bps == pytest.approx(pf * impact, rel=1e-12)
        assert b.temporary_cost_bps == pytest.approx((1 - pf) * impact, rel=1e-12)
        assert b.total_cost_bps == pytest.approx(2.5 + impact, rel=1e-12)


def test_3070_settings_almgren_chriss_does_not_use_are_refused():
    with pytest.raises(ValueError, match="impact_coefficient"):
        CostModel(impact_model="almgren_chriss", impact_coefficient=1.0)
    with pytest.raises(ValueError, match="impact_exponent"):
        CostModel(impact_model="almgren_chriss", impact_exponent=0.6)
    with pytest.raises(ValueError, match="impact_coefficient"):
        total_cost(2e6, 80e6, 0.02, impact_model="almgren_chriss", impact_coefficient=0.3)
    assert CostModel(impact_model="almgren_chriss").estimate(2e6, 80e6, 0.02) > 2.5


def test_3070_unknown_model_name_raises():
    with pytest.raises(ValueError, match="Unknown impact_model 'Sqrt'"):
        CostModel(impact_model="Sqrt")


@pytest.mark.parametrize("model", ["sqrt", "linear", "almgren_chriss"])
def test_3070_a_sell_costs_what_a_buy_costs(model):
    cm = CostModel(impact_model=model)
    sell = cm.estimate(-2e6, 80e6, 0.02)
    assert isinstance(sell, float)
    assert sell == pytest.approx(cm.estimate(2e6, 80e6, 0.02), rel=1e-12)
    assert total_cost(-2e6, 80e6, 0.02, impact_model="linear").total_cost_bps == pytest.approx(
        2.5 + 0.1 * (2e6 / 80e6) * 1e4)  # 27.5 bps, not a 22.5 bps rebate


# ── #3050: capacity counts short legs ────────────────────────────────────


def _long_short(thin_weight):
    return [
        {"symbol": "BIG", "weight": 0.5, "adv_usd": 5e9, "volatility": 0.015},
        {"symbol": "THIN", "weight": thin_weight, "adv_usd": 3e6, "volatility": 0.03},
    ]


def test_3050_thin_short_leg_sets_capacity():
    cap = alpha_decay_capacity(500, 4.0, _long_short(-0.5))
    # 10% of $3M ADV / 0.5 weight = $600,000.
    assert cap.participation_capacity_usd == pytest.approx(600_000.0)
    assert cap.capacity_usd == pytest.approx(600_000.0)
    assert cap.bottleneck_symbol == "THIN"
    assert {p["symbol"] for p in cap.per_symbol} == {"BIG", "THIN"}
    same_long = alpha_decay_capacity(500, 4.0, _long_short(0.5))
    assert cap.capacity_usd == same_long.capacity_usd


def test_3050_position_without_adv_is_refused():
    book = _long_short(-0.5)
    book[1].pop("adv_usd")
    with pytest.raises(ValueError, match="THIN"):
        alpha_decay_capacity(500, 4.0, book)


# ── #3132: Brinson-Fachler does not drop holdings ────────────────────────


def _total(w, r):
    return sum(w[a] * r[a] for a in w)


def test_3132_unmapped_holding_raises():
    pw = {"AAPL": 0.40, "JPM": 0.45, "TSLA": 0.15}
    pr = {"AAPL": 0.05, "JPM": 0.02, "TSLA": 0.12}
    bw, br = {"AAPL": 0.5, "JPM": 0.5}, {"AAPL": 0.04, "JPM": 0.02}
    with pytest.raises(ValueError, match="TSLA"):
        brinson_fachler(pw, pr, bw, br, {"AAPL": "Tech", "JPM": "Fin"})
    mapped = brinson_fachler(pw, pr, bw, br, {"AAPL": "Tech", "JPM": "Fin", "TSLA": "Auto"})
    assert mapped.total_active_return == pytest.approx((_total(pw, pr) - _total(bw, br)) * 100)


def test_3132_net_short_sector_counts():
    pw = {"AAPL": 0.7, "JPM": 0.5, "UNH": -0.2}
    pr = {"AAPL": 0.05, "JPM": 0.02, "UNH": -0.10}
    bw = {"AAPL": 0.4, "JPM": 0.4, "UNH": 0.2}
    br = {"AAPL": 0.04, "JPM": 0.02, "UNH": -0.10}
    res = brinson_fachler(pw, pr, bw, br, {"AAPL": "Tech", "JPM": "Fin", "UNH": "Health"})
    # Portfolio 0.035 + 0.010 + 0.020 = 6.5%; benchmark 0.016 + 0.008 - 0.020 = 0.4%.
    assert res.total_active_return == pytest.approx(6.1)
    # The net-short Health sector returns -10% on -0.2 weight: its selection is
    # w_b * (R_p - R_b) = 0.2 * 0 = 0, allocation (-0.4) * (-0.10 - 0.004) = +4.16%.
    assert res.per_sector["Health"]["allocation"] == pytest.approx(4.16)


def test_3132_zero_net_sector_with_return_raises():
    pw = {"A": 0.6, "L": 0.2, "S": -0.2, "B": 0.4}
    pr = {"A": 0.01, "L": 0.05, "S": -0.05, "B": 0.0}
    with pytest.raises(ValueError, match="net to zero"):
        brinson_fachler(pw, pr, {"A": 0.5, "B": 0.5}, {"A": 0.01, "B": 0.0},
                        {"A": "X", "B": "Y", "L": "Pair", "S": "Pair"})


# ── #3029: information ratio, factor risk split, drawdown ────────────────


def _pair(seed=29):
    """Benchmark x and a defensive y = 0.0002 + 0.3 x + e with exact OLS fit.

    The noise is demeaned and orthogonal to x, so the CAPM alpha is exactly
    0.0002 a day (positive) while the mean active return is
    0.3 * mean(x) + 0.0002 - mean(x) < 0: the two definitions disagree in sign.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0006, 0.012, 750)
    e = rng.normal(0.0, 0.008, 750)
    xc = x - x.mean()
    e = e - e.mean()
    e = e - (e @ xc) / (xc @ xc) * xc
    return x, 0.0002 + 0.3 * x + e


def test_3029_information_ratio_is_active_return_over_tracking_error():
    x, y = _pair()
    res = alpha_beta_decomposition(list(y), list(x))
    assert res.alpha == pytest.approx(0.0002 * 252 * 100, rel=1e-6)  # 5.04 %/yr
    active = y - x
    ir = active.mean() * 252 / (active.std(ddof=1) * math.sqrt(252))
    assert ir < 0 < res.alpha  # positive CAPM alpha, negative active return
    assert res.information_ratio == pytest.approx(ir, abs=1e-4)
    roll = rolling_attribution(list(y), list(x), window=60)
    w = active[-60 - 1 : -1]
    assert roll["ir"][-1] == pytest.approx(w.mean() * 252 / (w.std(ddof=1) * math.sqrt(252)), abs=1e-4)


def _factor_case():
    B = np.array([[1.0, 0.5], [1.0, -0.5], [1.0, 0.2]])
    F = np.array([[0.04, 0.012], [0.012, 0.02]])
    D = np.array([0.01, 0.02, 0.015])
    w = {"a": 0.4, "b": 0.3, "c": 0.3}
    return B, F, D, w


def test_3029_per_factor_parts_add_up_to_the_factor_share():
    B, F, D, w = _factor_case()
    res = factor_risk_decomposition(w, B, F, D, factor_names=["mkt", "size"])
    # B'w = (1.0, 0.11); F B'w = (0.04132, 0.0142); Euler parts 0.04132 and 0.001562.
    assert res.per_factor_contribution["mkt"] == pytest.approx(0.04132, abs=1e-8)
    assert res.per_factor_contribution["size"] == pytest.approx(0.001562, abs=1e-8)
    assert sum(res.per_factor_contribution.values()) == pytest.approx(res.factor_variance, abs=1e-8)
    cov = B @ F @ B.T + np.diag(D)
    ra = risk_attribution(w, cov, factor_loadings=B, factor_cov=F, specific_var=D,
                          asset_ids=["a", "b", "c"], factor_names=["mkt", "size"])
    assert sum(ra.per_factor.values()) == pytest.approx(ra.factor_risk_pct, abs=0.02)


def test_3029_max_drawdown_counts_a_loss_from_the_first_return():
    # Wealth 1 -> 0.97 -> 0.9312: a 6.88% drawdown from the starting value.
    assert compute_risk([-0.03, -0.04], RiskMeasure.MAX_DRAWDOWN) == pytest.approx(0.0688)
    assert compute_risk([0.10, -0.05], RiskMeasure.MAX_DRAWDOWN) == pytest.approx(0.05)


# ── #3039: fundamental factor model is point in time ─────────────────────


def _noise_universe(seed, n_assets=60, days=504):
    rng = np.random.default_rng(seed)
    rets = {f"S{i}": rng.normal(0.0003, 0.02, days).tolist() for i in range(n_assets)}
    caps = {k: float(rng.uniform(1e9, 1e11)) for k in rets}
    return rets, caps


def test_3039_noise_shows_no_momentum_premium():
    ts = []
    for seed in range(8):
        rets, caps = _noise_universe(seed)
        m = fundamental_factors(rets, caps)
        assert m.start_index == 252 and len(m.factor_returns["momentum"]) == 504 - 252
        ts.append(m.t_stats["momentum"])
    # In-sample exposures gave t of about +5 on every seed (premium ~ 252 / days).
    assert abs(np.mean(ts)) < 1.5
    assert max(abs(t) for t in ts) < 3.5


def test_3039_exposures_never_use_later_returns():
    rets, caps = _noise_universe(1, n_assets=30, days=400)
    base = fundamental_factors(rets, caps, momentum_window=120)
    k = 300  # a day inside every earlier momentum window
    bumped = {s: list(v) for s, v in rets.items()}
    bumped["S0"][k] += 0.5
    after = fundamental_factors(bumped, caps, momentum_window=120)
    days_before = k - base.start_index
    for f in base.factor_returns:
        np.testing.assert_array_equal(
            base.factor_returns[f][:days_before], after.factor_returns[f][:days_before]
        )


def test_3039_missing_market_cap_is_refused():
    rets, caps = _noise_universe(2, n_assets=10, days=60)
    caps.pop("S3")
    with pytest.raises(KeyError, match="S3"):
        fundamental_factors(rets, caps)


# ── #3006: walk-forward validation accepts the frame the docs build ──────


def _returns_frame_data(seed=6):
    rng = np.random.default_rng(seed)
    return {k: rng.normal(0.0005, 0.01, 400).tolist() for k in ("AAPL", "MSFT", "TLT")}


@pytest.mark.parametrize("kind", ["polars", "pandas"])
def test_3006_cross_validate_accepts_a_dataframe(kind):
    data = _returns_frame_data()
    if kind == "polars":
        import polars as pl

        frame = pl.DataFrame(data)
    else:
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame(data)
    cv = WalkForward(n_splits=4, test_size=63)
    alloc = lambda train: hierarchical_risk_parity(train).weights  # noqa: E731
    from_frame = cross_validate(alloc, frame, cv=cv)
    from_dict = cross_validate(alloc, data, cv=cv)
    assert from_frame.oos_sharpe == pytest.approx(from_dict.oos_sharpe, rel=1e-12)
    assert from_frame.stability() == from_dict.stability()
    gs = grid_search(lambda linkage: (lambda tr: hierarchical_risk_parity(tr, linkage_method=linkage).weights),
                     {"linkage": ["single", "ward"]}, frame, cv)
    assert gs.best_params["linkage"] in ("single", "ward")
