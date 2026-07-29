"""DataFrame-input parity: polars/pandas frames must give identical results to dicts."""

from __future__ import annotations

import numpy as np
import pytest
from cpz_quant import as_returns
from cpz_quant.portfolio import (
    drop_zero_variance,
    hierarchical_risk_parity,
    ledoit_wolf,
    mean_variance,
    risk_parity,
)

pl = pytest.importorskip("polars")


def _data(n_assets: int = 5, T: int = 300, seed: int = 9):
    rng = np.random.default_rng(seed)
    cols = {f"A{i}": rng.normal(0.0004, 0.01, T).tolist() for i in range(n_assets)}
    return cols


def test_as_returns_polars_numeric_only():
    d = _data()
    df = pl.DataFrame(d).with_columns(
        pl.Series("date", [f"2024-{1 + i % 12:02d}-01" for i in range(300)])
    )
    out = as_returns(df)
    assert set(out) == set(d)  # date column excluded
    assert out["A0"] == pytest.approx(d["A0"])


def test_as_returns_passthrough():
    d = _data()
    assert as_returns(d) is d
    arr = np.zeros((3, 3))
    assert as_returns(arr) is arr


def test_as_returns_no_numeric_raises():
    df = pl.DataFrame({"label": ["x", "y", "z"]})
    with pytest.raises(ValueError, match="no numeric columns"):
        as_returns(df)


@pytest.mark.parametrize("allocator", [hierarchical_risk_parity, risk_parity, mean_variance])
def test_polars_dict_parity(allocator):
    d = _data()
    df = pl.DataFrame(d)
    from_dict = allocator(d)
    from_frame = allocator(df)
    assert from_frame.weights == from_dict.weights
    assert from_frame.sharpe_ratio == from_dict.sharpe_ratio


def test_polars_keyword_call():
    d = _data()
    res = hierarchical_risk_parity(returns=pl.DataFrame(d))
    assert set(res.weights) == set(d)


def test_polars_covariance_and_preselection():
    d = _data()
    df = pl.DataFrame(d)
    lw_dict = ledoit_wolf(d)
    lw_frame = ledoit_wolf(df)
    np.testing.assert_allclose(lw_frame.covariance, lw_dict.covariance)
    kept = drop_zero_variance(df)
    assert set(kept) == set(d)


def test_pandas_dict_parity():
    pd = pytest.importorskip("pandas")
    d = _data()
    df = pd.DataFrame(d)
    from_dict = hierarchical_risk_parity(d)
    from_frame = hierarchical_risk_parity(df)
    assert from_frame.weights == from_dict.weights
