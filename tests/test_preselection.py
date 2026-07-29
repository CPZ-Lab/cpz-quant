"""Tests for pre-selection transformers."""

from __future__ import annotations

import numpy as np
from cpz_quant.portfolio.preselection import (
    drop_highly_correlated,
    drop_zero_variance,
    select_complete_assets,
    select_k_extremes,
    select_non_dominated,
)


def _universe(n=500, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0005, 0.01, n)
    return {
        "A": base.tolist(),
        "B": (base + rng.normal(0, 0.0005, n)).tolist(),   # ~duplicate of A (high corr)
        "C": rng.normal(0.0008, 0.02, n).tolist(),          # independent
        "D": rng.normal(0.0002, 0.03, n).tolist(),          # high vol, low return
        "FLAT": np.zeros(n).tolist(),                        # zero variance
    }


def test_drop_zero_variance():
    out = drop_zero_variance(_universe())
    assert "FLAT" not in out and "A" in out


def test_select_complete_assets():
    u = _universe()
    u["SHORT"] = u["C"][:100]  # short history
    out = select_complete_assets(u)
    assert "SHORT" not in out
    assert len(out) == len(u) - 1


def test_drop_highly_correlated():
    out = drop_highly_correlated(_universe(), threshold=0.9)
    # A and B are near-duplicates: exactly one should survive.
    assert ("A" in out) ^ ("B" in out)
    assert "C" in out


def test_select_k_extremes_top_sharpe():
    out = select_k_extremes(_universe(), k=2, measure="sharpe", highest=True)
    assert len(out) == 2


def test_select_k_extremes_bottom():
    out = select_k_extremes(_universe(), k=1, measure="volatility", highest=False)
    assert len(out) == 1


def test_select_non_dominated_removes_dominated():
    # E dominates F (higher return, lower vol) -> F removed.
    rng = np.random.default_rng(3)
    u = {
        "E": rng.normal(0.001, 0.01, 500).tolist(),
        "F": rng.normal(0.0002, 0.02, 500).tolist(),   # lower return, higher vol => dominated
        "G": rng.normal(0.0015, 0.03, 500).tolist(),   # highest return (non-dominated)
    }
    out = select_non_dominated(u)
    assert "F" not in out
    assert "E" in out and "G" in out
