"""Tests for the native risk-measure library + mean-risk optimizer."""

from __future__ import annotations

import numpy as np
import pytest
from cpz_quant.portfolio.risk_measures import (
    RiskMeasure,
    all_risk_measures,
    compute_risk,
    mean_risk_optimize,
)


@pytest.fixture
def rets():
    rng = np.random.default_rng(7)
    return rng.normal(0.0005, 0.012, 1000)


class TestComputeRisk:
    def test_all_measures_finite(self, rets):
        vals = all_risk_measures(rets)
        assert len(vals) == len(RiskMeasure)
        for k, v in vals.items():
            assert np.isfinite(v), k

    def test_tail_ordering_var_le_cvar_le_evar(self, rets):
        var = compute_risk(rets, RiskMeasure.VALUE_AT_RISK, beta=0.95)
        cvar = compute_risk(rets, RiskMeasure.CVAR, beta=0.95)
        evar = compute_risk(rets, RiskMeasure.EVAR, beta=0.95)
        # Coherent-measure ordering: VaR <= CVaR <= EVaR.
        assert var <= cvar + 1e-9
        assert cvar <= evar + 1e-9

    def test_dispersion_nonnegative(self, rets):
        for m in [RiskMeasure.VARIANCE, RiskMeasure.STANDARD_DEVIATION, RiskMeasure.SEMI_VARIANCE,
                  RiskMeasure.MAD, RiskMeasure.GINI_MEAN_DIFFERENCE, RiskMeasure.ULCER_INDEX]:
            assert compute_risk(rets, m) >= 0

    def test_max_drawdown_known(self):
        # +10%, then -50% peak-to-trough, then flat.
        r = np.array([0.1, -0.5, 0.0, 0.0])
        mdd = compute_risk(r, RiskMeasure.MAX_DRAWDOWN)
        assert mdd == pytest.approx(0.5, abs=1e-9)

    def test_no_drawdown_on_monotone_up(self):
        r = np.full(50, 0.002)
        assert compute_risk(r, RiskMeasure.MAX_DRAWDOWN) == pytest.approx(0.0, abs=1e-12)

    def test_cdar_ge_dar(self, rets):
        dar = compute_risk(rets, RiskMeasure.DRAWDOWN_AT_RISK, beta=0.95)
        cdar = compute_risk(rets, RiskMeasure.CDAR, beta=0.95)
        assert cdar >= dar - 1e-9

    def test_edar_ge_cdar(self, rets):
        cdar = compute_risk(rets, RiskMeasure.CDAR, beta=0.95)
        edar = compute_risk(rets, RiskMeasure.EDAR, beta=0.95)
        assert edar >= cdar - 1e-9


class TestMeanRiskOptimize:
    @pytest.fixture
    def assets(self):
        rng = np.random.default_rng(11)
        low = rng.normal(0.0004, 0.006, 800)   # low vol
        mid = rng.normal(0.0006, 0.015, 800)
        high = rng.normal(0.0008, 0.030, 800)   # high vol
        return np.column_stack([low, mid, high])

    def test_weights_valid(self, assets):
        r = mean_risk_optimize(assets, risk_measure=RiskMeasure.CVAR, objective="min_risk")
        w = np.array(list(r.weights.values()))
        assert r.success
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= -1e-6).all() and (w <= 1 + 1e-6).all()

    def test_min_variance_tilts_to_low_vol(self, assets):
        r = mean_risk_optimize(assets, risk_measure=RiskMeasure.VARIANCE, objective="min_risk",
                               asset_ids=["low", "mid", "high"])
        # Minimum-variance concentrates in the low-vol asset.
        assert r.weights["low"] > r.weights["high"]

    def test_min_cvar_runs_for_all_measures(self, assets):
        for m in [RiskMeasure.CVAR, RiskMeasure.CDAR, RiskMeasure.EVAR, RiskMeasure.MAD,
                  RiskMeasure.WORST_REALIZATION, RiskMeasure.ULCER_INDEX, RiskMeasure.SEMI_VARIANCE]:
            r = mean_risk_optimize(assets, risk_measure=m, objective="min_risk")
            w = np.array(list(r.weights.values()))
            assert w.sum() == pytest.approx(1.0, abs=1e-5), m

    def test_max_ratio_positive_return(self, assets):
        r = mean_risk_optimize(assets, risk_measure=RiskMeasure.CVAR, objective="max_ratio")
        assert r.expected_return > 0
