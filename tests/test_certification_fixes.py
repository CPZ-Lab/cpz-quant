"""Certification analytics, regime, grade and factor-attribution fixes.

Each test names the CPZ-Lab/cpzai issue it pins down and shows where its
reference value comes from.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest
from cpz_quant.certification import (
    CertLive,
    CertMetrics,
    CertRigor,
    certify,
    compute_risk_analytics,
    factor_attribution,
    regime_conditional_performance,
)
from cpz_quant.certification.grade import score_capacity_cost, score_factor_independence


def _equity(rets, start=100.0):
    return list(start * np.cumprod(np.r_[1.0, 1.0 + np.asarray(rets, dtype=float)]))


def _business_days(start: dt.date, n: int):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


# ── #3029: time under water ──────────────────────────────────────────────


def test_3029_days_flat_at_a_high_are_not_under_water():
    rng = np.random.default_rng(29)
    rets = np.r_[np.zeros(200), [-0.01, -0.01, 0.03], rng.normal(0.002, 0.001, 60)]
    eq = np.asarray(_equity(rets))
    a = compute_risk_analytics(list(eq))
    peak = np.maximum.accumulate(eq)
    run = best = 0
    for below in eq < peak:  # hand count of consecutive days strictly below the peak
        run = run + 1 if below else 0
        best = max(best, run)
    assert best == 2
    assert a.max_time_under_water_days == best


# ── #3047: capture ratios, worst month, wiped-out accounts ───────────────


def _geo(r):
    return np.prod(1 + r) ** (1 / r.size) - 1


def test_3047_two_times_leveraged_fund_captures_about_two():
    rng = np.random.default_rng(47)
    b_rets = rng.normal(0.0003, 0.01, 1500)
    s_rets = 2.0 * b_rets
    a = compute_risk_analytics(_equity(s_rets), _equity(b_rets))
    up, dn = b_rets > 0, b_rets < 0
    # Morningstar: ratio of per-period geometric means over up (down) periods.
    assert a.up_capture == pytest.approx(_geo(s_rets[up]) / _geo(b_rets[up]), abs=1e-3)
    assert a.down_capture == pytest.approx(_geo(s_rets[dn]) / _geo(b_rets[dn]), abs=1e-3)
    assert 1.9 < a.up_capture < 2.0 < a.down_capture < 2.1


def test_3047_worst_month_includes_the_first_trading_day():
    dates = _business_days(dt.date(2025, 1, 2), 300)
    eq = np.full(300, 100.0)
    crash = dates.index("2025-06-02")  # first trading day of June
    eq[crash:] = 90.0
    eq = eq * (1 + np.arange(300) * 1e-7)  # avoid a flat curve
    a = compute_risk_analytics(list(eq), dates=dates)
    may_end = eq[crash - 1]
    june_end = eq[max(i for i, d in enumerate(dates) if d.startswith("2025-06"))]
    assert a.worst_month == pytest.approx(round(june_end / may_end - 1, 4))
    assert a.worst_month == pytest.approx(-0.10, abs=1e-3)


def test_3047_short_wiped_out_curve_still_shows_total_loss():
    eq = _equity(np.full(12, 0.002)) + [0.0]
    a = compute_risk_analytics(eq)
    assert a is not None  # too short for statistics, not too short to show a wipe-out
    assert a.max_drawdown == -1.0 and a.annualized_return == -1.0
    assert a.sharpe is None and a.var95 is None
    assert compute_risk_analytics(_equity(np.full(12, 0.002))) is None


def test_3047_regime_breakdown_counts_a_wipe_out():
    rng = np.random.default_rng(4700)
    eq = _equity(rng.normal(0.0005, 0.01, 200)) + [0.0, 0.0, 0.0]
    b = regime_conditional_performance(eq)
    days = sum(r.n_days for r in b.vol_regimes)
    worst = min(r.annualized_return for r in b.vol_regimes if r.n_days)
    assert worst == -1.0  # the regime holding the wipe-out day lost everything
    assert days == 200 - 21 + 1  # the -100% day is counted, nothing after it


def test_3047_wiped_out_curve_shows_total_loss():
    rng = np.random.default_rng(470)
    eq = _equity(rng.normal(0.0004, 0.01, 250)) + [0.0] * 15
    a = compute_risk_analytics(eq)
    assert a.max_drawdown == -1.0
    assert a.annualized_return == -1.0
    before = compute_risk_analytics(eq[:251])
    assert a.sharpe < before.sharpe


# ── #3048: regimes come from data known before the day ───────────────────


def test_3048_random_walk_is_not_regime_dependent():
    bulls, bears, stressed = [], [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        rw = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 2000))))
        b = regime_conditional_performance(rw, rw)
        trend = {r.name: r.sharpe for r in b.trend_regimes}
        vol = {r.name: r.sharpe for r in b.vol_regimes}
        bulls.append(trend["bull"])
        bears.append(trend["bear"])
        stressed.append(vol["stressed"])
    # Labelling a day by its own close gave bull +3.1 and bear -3.0 on every seed.
    assert abs(np.mean(bulls)) < 0.5 and abs(np.mean(bears)) < 0.5
    assert abs(np.mean(stressed)) < 0.5


def test_3048_a_crash_day_is_labelled_by_the_days_before_it():
    rng = np.random.default_rng(3048)
    rets = rng.normal(0.001, 0.0001, 150)  # every day up, quietly
    rets[100] = -0.20
    b = regime_conditional_performance(_equity(rets), vol_window=21)
    regimes = {r.name: r for r in b.vol_regimes}
    # The crash day's regime comes from the quiet days before it, so it is not
    # "stressed"; the stressed days are the ones after it, all up days. Using the
    # crash day's own move put it in "stressed".
    assert regimes["stressed"].hit_rate == 1.0
    assert min(r.hit_rate for r in regimes.values()) < 1.0


# ── #3069: missing inputs never score better; NumPy numbers count ────────


def test_3069_missing_beta_never_beats_a_measured_beta():
    missing, _ = score_factor_independence(CertMetrics(sharpe_ratio=1.0, alpha=0.05))
    for beta in (0.0, 0.5, 0.8, 1.2, 3.0):
        measured, _ = score_factor_independence(CertMetrics(sharpe_ratio=1.0, beta=beta, alpha=0.05))
        assert missing <= measured


def test_3069_missing_capacity_never_beats_a_measured_capacity():
    missing, _ = score_capacity_cost(CertRigor())
    for cap in (1_000.0, 20_000.0, 1e6, 1e9):
        assert missing <= score_capacity_cost(CertRigor(capacity=cap))[0]


def _rig():
    return CertRigor(deflated_sharpe=0.9, probabilistic_sharpe=0.95, oos_sharpe=1.0,
                     capacity=5e6, turnover=2.0)


def test_3069_dropping_beta_no_longer_lifts_the_grade():
    with_beta = certify(backtest_verified=True, rigor=_rig(),
                        metrics=CertMetrics(sharpe_ratio=1.2, max_drawdown=-15, beta=0.8, profit_factor=1.5))
    without = certify(backtest_verified=True, rigor=_rig(),
                      metrics=CertMetrics(sharpe_ratio=1.2, max_drawdown=-15, profit_factor=1.5))
    assert without.composite <= with_beta.composite


def test_3069_numpy_numbers_and_flags_are_numbers_and_flags():
    plain = certify(
        backtest_verified=True, has_high_severity_lookahead=False, rigor=_rig(),
        metrics=CertMetrics(sharpe_ratio=1.2, max_drawdown=-15.0, beta=0.3),
        live=CertLive(sharpe_ratio=1.0, total_trades=40, elapsed_days=60),
    )
    numpy_typed = certify(
        backtest_verified=np.bool_(True), has_high_severity_lookahead=np.bool_(False), rigor=_rig(),
        metrics=CertMetrics(sharpe_ratio=np.float32(1.2), max_drawdown=np.float32(-15.0),
                            beta=np.float64(0.3)),
        live=CertLive(sharpe_ratio=np.float32(1.0), total_trades=np.int64(40),
                      elapsed_days=np.int32(60)),
    )
    assert not plain.provisional and not numpy_typed.provisional
    assert numpy_typed.grade == plain.grade
    assert numpy_typed.composite == pytest.approx(plain.composite, abs=0.1)


# ── #3177: drawdown units ────────────────────────────────────────────────


def test_3177_wiped_out_fraction_is_refused_with_or_without_analytics():
    eq = _equity(np.full(250, 0.001)) + [0.0] * 15
    a = compute_risk_analytics(eq)
    assert a.max_drawdown == -1.0
    for analytics in (None, a):
        with pytest.raises(ValueError, match="percent"):
            certify(backtest_verified=True, rigor=_rig(), analytics=analytics,
                    metrics=CertMetrics(sharpe_ratio=1.2, max_drawdown=a.max_drawdown))
    ok = certify(backtest_verified=True, rigor=_rig(), analytics=a,
                 metrics=CertMetrics(sharpe_ratio=1.2, max_drawdown=a.max_drawdown * 100))
    assert {d.key: d for d in ok.dimensions}["risk"].detail.startswith("Max drawdown 100.0%")


def test_3177_fraction_matching_analytics_is_refused_at_any_size():
    # A leveraged book ending below zero: drawdown beyond -100% (-1.04 as a fraction).
    eq = _equity(np.full(250, 0.001)) + [-5.0]
    a = compute_risk_analytics(eq)
    assert a.max_drawdown < -1.0
    with pytest.raises(ValueError, match="fraction"):
        certify(backtest_verified=True, rigor=_rig(), analytics=a,
                metrics=CertMetrics(sharpe_ratio=1.2, max_drawdown=a.max_drawdown))


def test_3177_fractional_drawdown_is_refused():
    with pytest.raises(ValueError, match="percent"):
        certify(backtest_verified=True, rigor=_rig(),
                metrics=CertMetrics(sharpe_ratio=1.2, max_drawdown=-0.62))


def test_3177_percent_drawdown_scores_as_before():
    res = certify(backtest_verified=True, rigor=_rig(),
                  metrics=CertMetrics(sharpe_ratio=1.2, max_drawdown=-62))
    risk = {d.key: d for d in res.dimensions}["risk"]
    assert risk.detail.startswith("Max drawdown 62.0%")
    assert risk.score == 5  # 100 - 62 = 38 is below the 45 band: the floor score 5


def test_3177_sub_one_percent_drawdown_accepted_when_analytics_confirm():
    eq = _equity(np.r_[np.full(100, 0.0005), [-0.0062], np.full(100, 0.0005)])
    a = compute_risk_analytics(eq)
    assert a.max_drawdown == pytest.approx(-0.0062, abs=1e-4)
    res = certify(backtest_verified=True, rigor=_rig(), analytics=a,
                  metrics=CertMetrics(sharpe_ratio=1.2, max_drawdown=a.max_drawdown * 100))
    assert {d.key: d for d in res.dimensions}["risk"].detail.startswith("Max drawdown 0.6%")


# ── #3145: an exact factor fit has no alpha t-stat ───────────────────────


def test_3145_exact_fit_gives_undefined_not_absurd_stats():
    rng = np.random.default_rng(3145)
    f = {k: rng.normal(0.0003, 0.01, 504) for k in ("EFA", "IEF", "GLD")}
    y = 0.5 * f["EFA"] + 0.3 * f["IEF"] + 0.2 * f["GLD"]
    fa = factor_attribution(y, f)
    assert fa.r_squared == 1.0
    assert fa.alpha_daily == pytest.approx(0.0, abs=1e-12)
    assert fa.alpha_tstat is None  # undefined, and None keeps the result valid JSON
    assert fa.idiosyncratic_sharpe is None
    assert all(t is None for t in fa.beta_tstats.values())
    assert fa.betas == {"EFA": 0.5, "IEF": 0.3, "GLD": 0.2}


def test_3145_noisy_fit_still_has_finite_stats():
    rng = np.random.default_rng(31450)
    f = {k: rng.normal(0.0003, 0.01, 504) for k in ("EFA", "IEF")}
    y = 0.0002 + 0.6 * f["EFA"] + rng.normal(0, 0.002, 504)
    fa = factor_attribution(y, f)
    assert math.isfinite(fa.alpha_tstat) and fa.alpha_tstat > 0
    assert math.isfinite(fa.idiosyncratic_sharpe)
