"""A missing return is an error, never data (#3051).

Kendall's tau of a pair with a NaN is NaN (it became a correlation of 0),
and NaN compares False and sorts last (pre-selection kept none of a
percent-change frame, whose first row is always blank).
"""

from __future__ import annotations

import numpy as np
import pytest
from cpz_quant.portfolio.copula import (
    ClaytonCopula,
    GaussianCopula,
    GumbelCopula,
    StudentTCopula,
    fit_copula,
    pseudo_observations,
    synthetic_returns,
)
from cpz_quant.portfolio.preselection import (
    drop_highly_correlated,
    drop_zero_variance,
    select_complete_assets,
    select_k_extremes,
    select_non_dominated,
)


def _correlated_pair(seed=3, n_days=1000):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 0.01, n_days)
    return np.column_stack([1.2 * m + rng.normal(0, 0.004, n_days), m + rng.normal(0, 0.002, n_days)])


def test_clean_pair_reference():
    # Reference: rho = sin(pi * tau / 2) from scipy's Kendall tau (Lindskog et al.).
    from scipy import stats

    X = _correlated_pair()
    tau = stats.kendalltau(X[:, 0], X[:, 1])[0]
    assert GaussianCopula.fit(X).correlation[0, 1] == pytest.approx(np.sin(np.pi * tau / 2), abs=1e-6)


@pytest.mark.parametrize("fit", [
    GaussianCopula.fit, StudentTCopula.fit, ClaytonCopula.fit, GumbelCopula.fit,
    pseudo_observations,
    lambda X: fit_copula(X, "gaussian"),
    lambda X: synthetic_returns(X, 100, family="gaussian", random_state=0),
], ids=["gaussian", "student_t", "clayton", "gumbel", "pseudo_obs", "fit_copula", "synthetic"])
def test_copulas_reject_a_blank_day(fit):
    X = _correlated_pair()
    X[500, 1] = np.nan  # one blank day out of 1,000 turned a 0.9 correlation into 0
    with pytest.raises(ValueError, match="missing or non-finite"):
        fit(X)


def test_copula_rejects_a_constant_series():
    X = _correlated_pair()
    X[:, 1] = 0.0
    with pytest.raises(ValueError, match="constant"):
        GaussianCopula.fit(X)


def _pct_change_frame(n_assets=6, n_days=300, seed=4):
    """A pandas-style percent-change frame: the first row is blank."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(n_assets):
        r = rng.normal(2e-4 * (k + 1), 0.01 * (1 + k % 3), n_days)
        r[0] = np.nan
        out[f"E{k}"] = r.tolist()
    return out


@pytest.mark.parametrize("select", [
    drop_zero_variance,
    drop_highly_correlated,
    lambda r: select_k_extremes(r, k=3),
    lambda r: select_k_extremes(r, k=2, measure="volatility", highest=False),
    select_non_dominated,
], ids=["zero_variance", "highly_correlated", "top_sharpe", "lowest_vol", "non_dominated"])
def test_preselection_rejects_blank_rows(select):
    with pytest.raises(ValueError, match="missing or non-finite"):
        select(_pct_change_frame())


def test_preselection_on_clean_rows_picks_by_the_metric():
    rets = {k: v[1:] for k, v in _pct_change_frame().items()}
    R = np.column_stack(list(rets.values()))
    sharpe = R.mean(axis=0) / R.std(axis=0, ddof=1)
    expected = sorted(np.array(list(rets))[np.argsort(sharpe)[-3:]])
    assert sorted(select_k_extremes(rets, k=3)) == expected
    assert len(drop_zero_variance(rets)) == len(rets)


def test_select_complete_assets_counts_blanks_as_missing():
    rets = {k: v[1:] for k, v in _pct_change_frame().items()}
    rets["LATE"] = [float("nan")] * 50 + rets["E0"][50:]  # listed 50 days late
    assert "LATE" not in select_complete_assets(rets)
    assert len(select_complete_assets(rets)) == len(rets) - 1
