"""Tests for regime-conditional performance."""

from __future__ import annotations

import numpy as np
from cpz_quant.certification import regime_conditional_performance


def _equity(rets, start=100.0):
    eq = [start]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    return eq


class TestRegime:
    def test_short_series_returns_none(self):
        assert regime_conditional_performance(_equity([0.001] * 20)) is None

    def test_vol_regimes_present_and_partition_days(self):
        rng = np.random.default_rng(0)
        # alternating calm and stressed vol blocks
        rets = np.concatenate([
            rng.normal(0.0004, 0.004, 120),   # calm
            rng.normal(0.0004, 0.02, 120),    # stressed
            rng.normal(0.0004, 0.01, 120),    # normal
        ])
        b = regime_conditional_performance(_equity(rets))
        assert b is not None
        names = {r.name for r in b.vol_regimes}
        assert names == {"calm", "normal", "stressed"}
        # Shares are non-negative and roughly sum to 1.
        total_share = sum(r.share_of_days for r in b.vol_regimes)
        assert 0.9 <= total_share <= 1.0001
        # Stressed regime has the highest annualized vol.
        vols = {r.name: r.annualized_vol for r in b.vol_regimes}
        assert vols["stressed"] > vols["calm"]

    def test_consistency_flag_and_worst_regime(self):
        rng = np.random.default_rng(1)
        rets = rng.normal(0.0006, 0.008, 400)
        b = regime_conditional_performance(_equity(rets))
        assert isinstance(b.consistent_across_regimes, bool)
        assert b.worst_vol_regime in {"calm", "normal", "stressed"}

    def test_trend_regimes_when_benchmark_present(self):
        rng = np.random.default_rng(2)
        b_rets = np.concatenate([rng.normal(0.001, 0.01, 200), rng.normal(-0.001, 0.015, 200)])
        p_rets = rng.normal(0.0005, 0.008, 400)
        b = regime_conditional_performance(_equity(p_rets), benchmark=_equity(b_rets))
        trend_names = {r.name for r in b.trend_regimes}
        assert trend_names == {"bull", "bear"}
        assert sum(r.n_days for r in b.trend_regimes) > 0

    def test_no_benchmark_means_no_trend_regimes(self):
        b = regime_conditional_performance(_equity(np.random.default_rng(3).normal(0.0004, 0.01, 300)))
        assert b.trend_regimes == []
