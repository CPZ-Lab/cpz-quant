"""Public wrapper contract for the extended indicators.

Every wrapper accepts a polars DataFrame, returns the latest non-NaN value
by default and a full-length ``pl.Series`` with ``series=True``, and agrees
with the underlying numpy kernel.
"""

from __future__ import annotations

import datetime as dt
import math

import cpz_quant.indicators as ind
import cpz_quant.indicators.trend as tr
import numpy as np
import polars as pl
import pytest

SINGLE = [
    "trima", "t3", "zlema", "hma", "alma", "vidya", "mcginley", "psar", "aroon_osc", "trix",
    "mass_index", "midpoint", "midprice", "stc", "cmo", "mom", "rocp", "rocr",
    "ultimate_oscillator", "awesome_oscillator", "apo", "ppo", "connors_rsi", "coppock", "dpo",
    "bop", "chaikin_volatility", "ulcer_index", "choppiness", "parkinson", "rogers_satchell",
    "yang_zhang", "vwma", "ad_line", "adosc", "force_index", "eom", "nvi", "pvi", "pvt",
    "linear_reg_value", "linear_reg_angle", "efficiency_ratio",
]
MULTI = [
    "ichimoku", "aroon", "vortex", "stochastic_slow", "stoch_rsi", "kst", "tsi",
    "fisher_transform", "rvgi", "elder_ray", "wavetrend", "dmi", "klinger", "pvo",
]


@pytest.fixture(scope="module")
def bars(ohlcv) -> pl.DataFrame:
    return pl.DataFrame({k: ohlcv[k] for k in ("open", "high", "low", "close", "volume")})


def _last_valid(s: pl.Series) -> float:
    arr = s.to_numpy()
    arr = arr[~np.isnan(arr)]
    return float(arr[-1]) if arr.size else math.nan


@pytest.mark.parametrize("name", SINGLE)
def test_single_output_contract(name, bars):
    fn = getattr(ind, name)
    full = fn(bars, series=True)
    assert isinstance(full, pl.Series) and full.dtype == pl.Float64
    assert len(full) == bars.height
    latest = fn(bars)
    assert isinstance(latest, float)
    assert latest == pytest.approx(_last_valid(full), rel=1e-12)
    assert name in ind.__all__


@pytest.mark.parametrize("name", MULTI)
def test_multi_output_contract(name, bars):
    fn = getattr(ind, name)
    full = list(fn(bars, series=True))
    latest = list(fn(bars))
    assert len(full) == len(latest) >= 2
    for s, v in zip(full, latest):
        assert isinstance(s, pl.Series) and len(s) == bars.height
        assert v == pytest.approx(_last_valid(s), rel=1e-12)
    assert name in ind.__all__


def test_wrappers_match_kernels(bars, ohlcv):
    np.testing.assert_array_equal(
        ind.hma(bars, 16, series=True).to_numpy(), tr.hma_series(ohlcv["close"], 16)
    )
    result = ind.dmi(bars, 14, series=True)
    assert result.adx.to_numpy()[-1] == pytest.approx(ind.dmi(bars, 14).adx)
    ichi = ind.ichimoku(bars)
    assert set(ichi.__slots__) == {"tenkan", "kijun", "senkou_a", "senkou_b"}


def test_source_and_column_overrides(bars):
    renamed = bars.rename({"close": "px"})
    assert ind.t3(renamed, col_close="px") == ind.t3(bars)
    assert ind.zlema(bars, source="hl2") != ind.zlema(bars)


def test_array_input_for_close_based_wrappers(ohlcv):
    arr = ohlcv["close"]
    assert ind.trima(arr, 10) == ind.trima(arr.tolist(), 10)


def test_frame_required_for_multi_column_wrappers(ohlcv):
    with pytest.raises(TypeError):
        ind.vwma(ohlcv["close"])


def test_vwap_session_anchor():
    ts = [
        dt.datetime(2026, 1, 5, 9, 30), dt.datetime(2026, 1, 5, 10, 30),
        dt.datetime(2026, 1, 6, 9, 30), dt.datetime(2026, 1, 6, 10, 30),
    ]
    df = pl.DataFrame({
        "ts": ts, "open": [1.0, 2, 3, 4], "high": [2.0, 3, 4, 5], "low": [1.0, 2, 3, 4],
        "close": [1.5, 2.5, 3.5, 4.5], "volume": [10.0, 30, 10, 10],
    })
    anchored = ind.vwap(df, session_col="ts", series=True).to_list()
    tp = [(h + lo + c) / 3 for h, lo, c in zip(df["high"], df["low"], df["close"])]
    assert anchored[0] == pytest.approx(tp[0])
    assert anchored[1] == pytest.approx((tp[0] * 10 + tp[1] * 30) / 40)
    assert anchored[2] == pytest.approx(tp[2])
    assert anchored[3] == pytest.approx((tp[2] + tp[3]) / 2)
    cumulative = ind.vwap(df, series=True).to_list()
    assert cumulative[2] != pytest.approx(anchored[2])


def test_candle_pattern_wrappers(bars):
    from cpz_quant.indicators.candles import CANDLE_PATTERNS, TALIB_NAMES

    assert len(ind.CANDLE_PATTERN_NAMES) == len(CANDLE_PATTERNS) == len(TALIB_NAMES) == 61
    o, h, lo, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    full = ind.candle_pattern(bars, "engulfing", series=True)
    assert full.dtype == pl.Int32 and len(full) == bars.height
    np.testing.assert_array_equal(full.to_numpy(), CANDLE_PATTERNS["engulfing"](o, h, lo, c))
    assert ind.candle_pattern(bars, "engulfing") == int(full[-1])
    loose = ind.candle_pattern(bars, "morning_star", penetration=0.0, series=True)
    np.testing.assert_array_equal(
        loose.to_numpy(), CANDLE_PATTERNS["morning_star"](o, h, lo, c, penetration=0.0)
    )
    frame = ind.candle_patterns(bars)
    assert frame.columns == list(ind.CANDLE_PATTERN_NAMES) and frame.height == bars.height
    subset = ind.candle_patterns(bars, ["doji", "hammer"])
    assert subset.columns == ["doji", "hammer"]
    with pytest.raises(ValueError):
        ind.candle_pattern(bars, "tweezer_top")
    with pytest.raises(TypeError):
        ind.candle_pattern(bars, "doji", penetration=0.3)
    with pytest.raises(TypeError):
        ind.candle_pattern(o, "doji")
