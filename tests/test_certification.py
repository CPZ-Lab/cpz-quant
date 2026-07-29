"""Tests for cpz_quant.certification — analytics parity, grading, PBO, factor attribution.

All synthetic data, no external dependencies. Mirrors the edge engine test suite
so the SDK and hosted certification stay in lockstep.
"""

from __future__ import annotations

import numpy as np
import pytest
from cpz_quant.certification import (
    CertLive,
    CertMetrics,
    CertRigor,
    certify,
    compute_risk_analytics,
    factor_attribution,
    probability_of_backtest_overfitting,
)


def _equity(rets, start=100.0):
    eq = [start]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    return eq


OSC = [0.0005 + 0.01 * np.sin(i / 3) for i in range(400)]


# ── analytics ────────────────────────────────────────────────────────────────
class TestAnalytics:
    def test_short_series_returns_none(self):
        assert compute_risk_analytics(_equity(OSC[:10])) is None
        assert compute_risk_analytics([]) is None

    def test_full_block_on_realistic_series(self):
        a = compute_risk_analytics(_equity(OSC))
        assert a is not None
        assert a.observations == len(OSC)
        assert a.max_drawdown < 0
        assert a.max_time_under_water_days > 0
        assert a.cvar95 <= a.var95
        assert a.cvar99 <= a.cvar95
        assert a.min_trl_years > 0
        assert isinstance(a.significant, bool)

    def test_no_downside_series(self):
        a = compute_risk_analytics(_equity([0.0004] * 300))
        assert a.max_drawdown == 0.0
        assert a.sortino is None
        assert a.calmar is None

    def test_benchmark_identity_unit_beta(self):
        eq = _equity(OSC)
        a = compute_risk_analytics(eq, benchmark=eq)
        assert a.beta == pytest.approx(1.0, abs=0.02)
        assert a.correlation == pytest.approx(1.0, abs=0.02)

    def test_crisis_defensive_book(self):
        b_rets = [0.004] * 50 + [-0.018] * 20 + [0.003] * 60
        p_rets = [0.0002 if 50 <= i < 70 else 0.001 for i in range(len(b_rets))]
        a = compute_risk_analytics(_equity(p_rets), benchmark=_equity(b_rets))
        assert a.crisis_benchmark_return < -0.15
        assert a.crisis_return > a.crisis_benchmark_return
        assert a.down_capture < 0.5

    def test_worst_best_month_from_dates(self):
        import datetime as dt
        eq = _equity(OSC)
        base = dt.date(2020, 1, 1)
        dates = [(base + dt.timedelta(days=i)).isoformat() for i in range(len(eq))]
        a = compute_risk_analytics(eq, dates=dates)
        assert a.worst_month <= a.best_month


# ── grading (parity with certification.ts) ──────────────────────────────────
STRONG_RIGOR = CertRigor(deflated_sharpe=0.96, probabilistic_sharpe=0.97, num_trials=40,
                         oos_sharpe=1.4, oos_return=12, turnover=2, capacity=60_000_000,
                         parameter_stability=True)
STRONG_METRICS = CertMetrics(sharpe_ratio=1.6, max_drawdown=-8, volatility=0.12,
                             profit_factor=2.4, alpha=0.06, beta=0.15, total_trades=320)


class TestGrade:
    def test_provisional_strong_backtest_is_A_not_Aplus(self):
        r = certify(backtest_verified=True, metrics=STRONG_METRICS, rigor=STRONG_RIGOR)
        assert r.gates_passed and r.certified and r.provisional
        assert r.composite >= 90
        assert r.grade == "A"  # provisional caps at A

    def test_full_with_live_can_reach_Aplus(self):
        live = CertLive(sharpe_ratio=1.5, total_trades=60, elapsed_days=120, alpha=0.05)
        r = certify(backtest_verified=True, metrics=STRONG_METRICS, rigor=STRONG_RIGOR, live=live)
        assert not r.provisional
        assert r.grade == "A+"

    def test_gate_fail_not_certified(self):
        assert certify(backtest_verified=False, metrics=STRONG_METRICS, rigor=STRONG_RIGOR).grade == "Not Certified"
        assert certify(backtest_verified=True, metrics=STRONG_METRICS, rigor=None).grade == "Not Certified"
        neg = CertRigor(**{**STRONG_RIGOR.__dict__, "deflated_sharpe": -0.1})
        assert certify(backtest_verified=True, metrics=STRONG_METRICS, rigor=neg).grade == "Not Certified"

    def test_analytics_tail_penalty_lowers_risk(self):
        # A fat-tailed, weak-downside analytics block should not raise the grade.
        thin = compute_risk_analytics(_equity([0.001, -0.05] * 200))  # jagged, negative skew-ish
        base = certify(backtest_verified=True, metrics=CertMetrics(sharpe_ratio=0.5, max_drawdown=-25, profit_factor=1.1), rigor=STRONG_RIGOR)
        withA = certify(backtest_verified=True, metrics=CertMetrics(sharpe_ratio=0.5, max_drawdown=-25, profit_factor=1.1), rigor=STRONG_RIGOR, analytics=thin)
        risk_base = next(d.score for d in base.dimensions if d.key == "risk")
        risk_a = next(d.score for d in withA.dimensions if d.key == "risk")
        assert risk_a <= risk_base + 5  # analytics never inflate beyond the small bonus

    def test_engine_version(self):
        r = certify(backtest_verified=True, metrics=STRONG_METRICS, rigor=STRONG_RIGOR)
        assert r.engine_version == "1.1.0"


# ── PBO / CSCV ───────────────────────────────────────────────────────────────
class TestPBO:
    def test_random_noise_is_high_pbo(self):
        # Pure noise configs: the in-sample winner is random => PBO near 0.5.
        rng = np.random.default_rng(7)
        M = rng.normal(0, 0.01, size=(600, 20))
        res = probability_of_backtest_overfitting(M, n_splits=10)
        assert 0.0 <= res.pbo <= 1.0
        assert res.pbo > 0.25  # noise is not robust
        assert res.n_trials == 20

    def test_one_genuinely_good_config_lowers_pbo(self):
        rng = np.random.default_rng(11)
        M = rng.normal(0, 0.01, size=(600, 20))
        M[:, 0] += 0.004  # config 0 has a real, persistent edge
        res = probability_of_backtest_overfitting(M, n_splits=10)
        assert res.pbo < 0.2  # a genuinely superior config is rarely overfit

    def test_requires_multiple_trials(self):
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(np.zeros((100, 1)))


# ── factor attribution ───────────────────────────────────────────────────────
class TestFactors:
    def test_pure_alpha_has_zero_beta(self):
        rng = np.random.default_rng(3)
        mkt = rng.normal(0.0003, 0.01, 500)
        strat = rng.normal(0.0008, 0.004, 500)  # independent of market
        fa = factor_attribution(strat, {"MKT": mkt})
        assert abs(fa.betas["MKT"]) < 0.2
        assert fa.alpha_annualized > 0
        assert fa.idiosyncratic_sharpe > 0

    def test_pure_beta_has_no_alpha(self):
        rng = np.random.default_rng(5)
        mkt = rng.normal(0.0004, 0.012, 500)
        strat = 1.5 * mkt  # pure levered beta, no alpha
        fa = factor_attribution(strat, {"MKT": mkt})
        assert fa.betas["MKT"] == pytest.approx(1.5, abs=0.05)
        assert abs(fa.alpha_annualized) < 0.02
        assert fa.r_squared > 0.98
