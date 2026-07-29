"""Smoke tests for the optional plotly viz module."""

from __future__ import annotations

import numpy as np
import pytest

plotly = pytest.importorskip("plotly")

from cpz_quant.viz import (  # noqa: E402
    plot_corr_clusters,
    plot_drawdown,
    plot_frontier,
    plot_weights,
)


def _data(n_assets: int = 4, T: int = 250, seed: int = 5):
    rng = np.random.default_rng(seed)
    return {f"A{i}": rng.normal(0.0005, 0.01, T).tolist() for i in range(n_assets)}


def test_plot_weights():
    fig = plot_weights({"AAPL": 0.5, "TLT": 0.3, "GLD": 0.2})
    assert fig.data[0].orientation == "h"


def test_plot_frontier_marks_extremes():
    fig = plot_frontier(_data(), n_points=8)
    names = {t.name for t in fig.data}
    assert {"frontier", "min variance", "max Sharpe"} <= names


def test_plot_drawdown_two_panels():
    eq = (100 * np.cumprod(1 + np.random.default_rng(1).normal(0.0004, 0.01, 300))).tolist()
    fig = plot_drawdown(eq)
    assert len(fig.data) == 2


def test_plot_drawdown_too_short_raises():
    with pytest.raises(ValueError, match="at least 2"):
        plot_drawdown([100.0])


def test_plot_corr_clusters_square():
    fig = plot_corr_clusters(_data(n_assets=6))
    z = np.asarray(fig.data[0].z)
    assert z.shape == (6, 6)
