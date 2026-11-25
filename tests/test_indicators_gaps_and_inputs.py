"""Indicator behaviour on missing bars, chained inputs, flat or empty data,
pandas input and a few definition details.

Each test names the CPZ-Lab/cpzai issue it pins down. Reference values come
from TA-Lib 0.8.0 (dev dependency), pandas-style rolling means computed with
NumPy, or a hand computation written out in the test.
"""

from __future__ import annotations

import math
import warnings

import cpz_quant.indicators as ind
import cpz_quant.indicators.trend as tr
import cpz_quant.indicators.volatility as vl
import numpy as np
import polars as pl
import pytest

talib = pytest.importorskip("talib")


@pytest.fixture(scope="module")
def bars(ohlcv) -> pl.DataFrame:
    return pl.DataFrame({k: ohlcv[k] for k in ("open", "high", "low", "close", "volume")})


def _with_null(bars: pl.DataFrame, col: str, back: int) -> pl.DataFrame:
    values = bars[col].to_numpy().astype(float).copy()
    values[len(values) - back] = np.nan
    return bars.with_columns(pl.Series(col, values))


# ── #3040: a missing bar never yields a stale "latest" value ─────────────


def test_3040_sma_latest_ignores_a_gap_outside_its_window(bars):
    gapped = _with_null(bars, "close", 60)
    close = bars["close"].to_numpy()
    # pandas rolling(20).mean() at the last bar = mean of the last 20 closes.
    assert ind.sma(gapped, 20) == pytest.approx(close[-20:].mean(), rel=1e-12)
    full = ind.sma(gapped, 20, series=True).to_numpy()
    i_gap = len(close) - 60
    assert np.isnan(full[i_gap : i_gap + 20]).all()  # only windows holding the gap
    assert np.isfinite(full[i_gap + 20 :]).all()


@pytest.mark.parametrize("call", [
    lambda b: ind.ema(b, 20),
    lambda b: ind.macd(b).macd,
    lambda b: ind.macd(b).histogram,
    lambda b: ind.rsi(b, 14),
])
def test_3040_recursive_indicators_return_nan_not_a_pre_gap_value(bars, call):
    gapped = _with_null(bars, "close", 60)
    with pytest.warns(RuntimeWarning, match="latest value is NaN"):
        value = call(gapped)
    assert math.isnan(value)


def test_3040_matches_talib_which_also_gives_nan(bars):
    gapped = _with_null(bars, "close", 60)
    c = gapped["close"].to_numpy()
    assert math.isnan(talib.EMA(c, 20)[-1]) and math.isnan(talib.MACD(c)[0][-1])
    with pytest.warns(RuntimeWarning):
        assert math.isnan(ind.ema(gapped, 20))


def test_3040_null_high_does_not_freeze_atr(bars):
    gapped = _with_null(bars, "high", 60)
    h, lo, c = (gapped[k].to_numpy() for k in ("high", "low", "close"))
    assert math.isnan(talib.ATR(h, lo, c, 14)[-1])
    with pytest.warns(RuntimeWarning):
        assert math.isnan(ind.atr(gapped, 14))


def test_3040_clean_data_raises_no_warning(bars):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isfinite(ind.ema(bars, 20))


# ── #3052: indicators of indicators ──────────────────────────────────────


def test_3052_sma_of_rsi_matches_talib(ohlcv, assert_match):
    r = ind.rsi(ohlcv["close"], 9, series=True)
    ours = ind.sma(r, 5, series=True).to_numpy()
    ref = talib.SMA(talib.RSI(ohlcv["close"], 9), 5)
    assert_match(ours, ref, name="sma(rsi9, 5)")


def test_3052_bollinger_of_rsi_matches_talib(ohlcv, assert_match):
    r = ind.rsi(ohlcv["close"], 9, series=True)
    up, mid, lo = ind.bollinger(r, 20, series=True)
    ref_up, ref_mid, ref_lo = talib.BBANDS(talib.RSI(ohlcv["close"], 9), 20, 2.0, 2.0, 0)
    assert_match(mid.to_numpy(), ref_mid, name="bb mid")
    assert_match(up.to_numpy(), ref_up, name="bb up")
    assert_match(lo.to_numpy(), ref_lo, name="bb lo")


@pytest.mark.parametrize("fn", [ind.ema, ind.dema, ind.tema])
def test_3052_ema_family_of_rsi_starts_when_rsi_does(ohlcv, fn):
    r = ind.rsi(ohlcv["close"], 9, series=True).to_numpy()
    out = fn(r, 5, series=True).to_numpy()
    first = int(np.argmax(np.isfinite(r)))
    assert np.isnan(out[:first]).all()
    assert np.isfinite(out[first:]).all()


def test_3052_ema_of_rsi_is_the_recursion_from_the_first_rsi(ohlcv):
    r = ind.rsi(ohlcv["close"], 9, series=True).to_numpy()
    out = ind.ema(r, 5, series=True).to_numpy()
    first = int(np.argmax(np.isfinite(r)))
    a = 2.0 / 6.0
    expected = r[first]
    for i in range(first + 1, first + 4):
        expected = a * r[i] + (1 - a) * expected
    assert out[first] == r[first]
    assert out[first + 3] == pytest.approx(expected, rel=1e-12)


def test_3052_macd_of_rsi_has_values(ohlcv):
    r = ind.rsi(ohlcv["close"], 9, series=True)
    assert np.isfinite(ind.macd(r, series=True).macd.to_numpy()).sum() > 500


def test_3052_sma_of_atr_is_a_rolling_mean(bars):
    a = ind.atr(bars, 14, series=True).to_numpy()
    out = ind.sma(a, 5, series=True).to_numpy()
    first = int(np.argmax(np.isfinite(a)))
    assert out[first + 4] == pytest.approx(a[first : first + 5].mean(), rel=1e-12)
    assert np.isfinite(out[first + 4 :]).all()


def test_3052_rsi_of_sma_matches_talib_without_counting_warmup_as_flat(ohlcv, assert_match):
    s = ind.sma(ohlcv["close"], 30, series=True).to_numpy()
    ours = ind.rsi(s, 14, series=True).to_numpy()
    assert_match(ours, talib.RSI(s, 14), name="rsi(sma30)")
    w = ind.wma(ohlcv["close"], 20, series=True).to_numpy()
    assert_match(ind.rsi(w, 14, series=True).to_numpy(), talib.RSI(w, 14), name="rsi(wma20)")


def test_3052_stochastic_d_warmup_is_not_filled_with_50(bars, ohlcv):
    k, d = ind.stochastic(bars, 14, 3, series=True)
    k, d = k.to_numpy(), d.to_numpy()
    assert np.isnan(d[:15]).all()  # %K starts at 13, %D (3-bar mean) at 15
    assert d[15] == pytest.approx(k[13:16].mean(), rel=1e-12)
    _, fastd = talib.STOCHF(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14, 3, 0)
    np.testing.assert_allclose(d[15:], fastd[15:], rtol=1e-9, atol=1e-9)


# ── #3053: flat prices and empty input ───────────────────────────────────


def test_3053_rsi_of_a_flat_price_is_undefined_not_oversold():
    flat = np.full(40, 25.0)
    assert math.isnan(ind.rsi(flat))
    assert np.isnan(ind.rsi(flat, series=True).to_numpy()).all()


def test_3053_rsi_after_moves_then_flat_keeps_its_last_meaning():
    up_then_flat = np.r_[np.linspace(20, 30, 30), np.full(30, 30.0)]
    assert ind.rsi(up_then_flat) == pytest.approx(100.0)  # only gains: RSI 100 by definition


@pytest.mark.parametrize("name", ["rsi", "williams_r", "sma", "ema", "atr", "cci", "obv"])
def test_3053_empty_frame_raises_instead_of_returning_zero(bars, name):
    with pytest.raises(ValueError, match="no bars"):
        getattr(ind, name)(bars.head(0))


def test_3053_empty_frame_multi_output_raises(bars):
    with pytest.raises(ValueError, match="no bars"):
        ind.stochastic(bars.head(0))
    with pytest.raises(ValueError, match="no bars"):
        ind.macd(bars.head(0))


def test_3053_empty_frame_series_is_empty(bars):
    assert len(ind.rsi(bars.head(0), series=True)) == 0


# ── #3073: momentum() after rsi() ────────────────────────────────────────


def test_3073_momentum_survives_other_momentum_indicators(ohlcv):
    c = ohlcv["close"]
    before = ind.momentum(c, 10)
    ind.rsi(c)
    ind.macd(c)
    ind.stoch_rsi(c)
    assert ind.momentum(c, 10) == before
    assert before == pytest.approx((c[-1] / c[-11] - 1) * 100, rel=1e-12)


def test_3073_momentum_submodule_still_importable():
    import cpz_quant.indicators.momentum as mo

    assert callable(mo.rsi_series)
    from cpz_quant.indicators import momentum

    assert momentum(np.arange(1.0, 30.0), 5) == pytest.approx((29 / 24 - 1) * 100)


# ── #3072: pandas input, numpy obv, Rust Garman-Klass settings ───────────


@pytest.fixture(scope="module")
def pandas_bars(bars):
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({c: bars[c].to_numpy() for c in bars.columns})


@pytest.mark.parametrize("call", [
    lambda b: ind.realized_vol(b, 20),
    lambda b: ind.hurst(b),
    lambda b: ind.donchian(b, 20)[0],
    lambda b: ind.donchian(b, 20)[1],
    lambda b: ind.atr(b, 14),
    lambda b: ind.obv(b),
    lambda b: ind.mfi(b),
    lambda b: ind.cmf(b),
    lambda b: ind.sma(b, 20, source="hl2"),
    lambda b: ind.vwma(b),
])
def test_3072_pandas_frame_gives_the_polars_number(bars, pandas_bars, call):
    assert call(pandas_bars) == pytest.approx(call(bars), rel=1e-12)


def test_3072_donchian_upper_is_a_price_not_the_volume(bars, pandas_bars):
    upper, _ = ind.donchian(pandas_bars, 20)
    assert upper == pytest.approx(bars["high"].to_numpy()[-20:].max())


def test_3072_two_dimensional_array_is_refused(bars):
    arr = bars.select(["open", "high", "low", "close"]).to_numpy()
    with pytest.raises(ValueError, match="1-D"):
        ind.realized_vol(arr)


def test_3072_obv_without_volume_is_refused():
    with pytest.raises(TypeError, match="obv"):
        ind.obv(np.arange(1.0, 50.0))


def test_3072_derived_source_on_a_bare_series_is_refused():
    with pytest.raises(TypeError, match="source='hl2'"):
        ind.sma(np.arange(1.0, 50.0), 5, source="hl2")


def test_3072_rust_garman_klass_only_used_for_its_fixed_settings(bars, monkeypatch):
    calls = []

    def fake(o, h, low, c, period):
        calls.append(period)
        return [0.0] * len(c)

    monkeypatch.setattr(ind, "fast_garman_klass", fake)
    raw = ind.garman_klass(bars, 20, annualize=False)
    crypto = ind.garman_klass(bars, 20, trading_days=365)
    assert calls == []  # the kernel always annualises with 252 days
    o, h, lo, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    ref = vl.garman_klass_series(o, h, lo, c, 20, annualize=False)[-1]
    assert raw == pytest.approx(ref, rel=1e-12)
    assert crypto == pytest.approx(ref * math.sqrt(365), rel=1e-12)
    ind.garman_klass(bars, 20)
    assert calls == [20]


# ── #3071: Ichimoku displacement, Supertrend flips, EMA/MACD seeding ─────


def test_3071_ichimoku_refuses_a_backward_shift(bars):
    for d in (0, -3):
        with pytest.raises(ValueError, match="displacement"):
            ind.ichimoku(bars, displacement=d)


def test_3071_ichimoku_displacement_1_uses_no_later_bar(bars):
    t = 300
    full = ind.ichimoku(bars, displacement=1, series=True)
    part = ind.ichimoku(bars.head(t + 1), displacement=1, series=True)
    for a, b in zip(full, part):
        assert a.to_numpy()[t] == b.to_numpy()[t]


def _trend_then_crash():
    n = 40
    close = 100.0 + np.arange(n, dtype=float)
    high = close + 1.0
    low = close - 1.0
    # Bar 40 gaps far below the rising lower band.
    close = np.r_[close, 60.0]
    high = np.r_[high, 139.0]
    low = np.r_[low, 59.0]
    return high, low, close


def test_3071_supertrend_flips_on_the_crossing_bar():
    high, low, close = _trend_then_crash()
    st = tr.supertrend_series(high, low, close, 10, multiplier=3.0)
    atr = vl.atr_series(high, low, close, 10)
    t = len(close) - 1
    assert st[t - 1] < close[t - 1]  # long side before the crash
    # On the crash bar the line is the fresh upper band (hl2 + 3 * ATR).
    assert st[t] == pytest.approx((high[t] + low[t]) / 2 + 3.0 * atr[t], rel=1e-12)
    assert st[t] > close[t]


def test_3071_supertrend_sides_switch_only_on_a_band_cross(ohlcv):
    h, lo, c = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    st = tr.supertrend_series(h, lo, c, 10, multiplier=3.0)
    long_side = st < c
    flips = np.flatnonzero(long_side[1:] != long_side[:-1]) + 1
    assert flips.size > 3
    for t in flips:
        if np.isnan(st[t - 1]):
            continue
        # The bar that flips is the bar whose close crossed the previous line.
        assert (c[t] < st[t - 1]) if not long_side[t] else (c[t] > st[t - 1])


def test_3071_ema_and_macd_docstrings_state_their_seed():
    assert "Seeded with the first valid" in ind.ema.__doc__ and "TA-Lib" in ind.ema.__doc__
    assert "TA-Lib" in ind.macd.__doc__ and "seed" in ind.macd.__doc__


# ── #3054: Hurst exponent of a random walk is about 0.5 ──────────────────


def _fgn(n: int, hurst: float, rng: np.random.Generator) -> np.ndarray:
    """Fractional Gaussian noise by Cholesky factorisation of its autocovariance."""
    k = np.arange(n, dtype=float)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * hurst) - 2 * k ** (2 * hurst) + np.abs(k - 1) ** (2 * hurst))
    cov = gamma[np.abs(np.subtract.outer(np.arange(n), np.arange(n)))]
    return np.linalg.cholesky(cov) @ rng.normal(size=n)


def test_3054_random_walk_reads_about_half():
    rng = np.random.default_rng(3054)
    hs = [ind.hurst(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 750)))) for _ in range(20)]
    # Uncorrected R/S averaged 0.63 here; the corrected estimator centres on 0.5.
    assert abs(np.mean(hs) - 0.5) < 0.03
    assert sum(h > 0.5 for h in hs) < 16


def test_3054_mean_reverting_series_reads_below_half():
    rng = np.random.default_rng(54)
    hs = [ind.hurst(100 * np.exp(np.cumsum(0.01 * _fgn(750, 0.4, rng)))) for _ in range(8)]
    assert np.mean(hs) < 0.45


def test_3054_too_short_is_nan_not_a_made_up_half():
    assert math.isnan(ind.hurst(np.array([100.0, 101.0, 102.0])))


def test_3054_non_positive_prices_are_refused():
    with pytest.raises(ValueError, match="positive"):
        ind.hurst(np.r_[np.linspace(1, 2, 100), 0.0])


# ── Review follow-up: a NaN in a window never becomes a made-up number ───


def _gap_bars(bars, col, k):
    values = bars[col].to_numpy().astype(float).copy()
    values[k] = np.nan
    return bars.with_columns(pl.Series(col, values))


K = 300  # bar index of the missing value


@pytest.mark.parametrize("name, col, call, at", [
    ("cci", "close", lambda b: ind.cci(b, 20, series=True), K + 1),
    ("zscore", "close", lambda b: ind.zscore(b, 60, series=True), K + 1),
    ("roc", "close", lambda b: ind.roc(b, 10, series=True), K + 10),
    ("momentum", "close", lambda b: ind.momentum(b, 10, series=True), K + 10),
    ("stc", "close", lambda b: ind.stc(b, series=True), K + 1),
    ("mfi", "close", lambda b: ind.mfi(b, 14, series=True), K + 1),
    ("obv", "close", lambda b: ind.obv(b, series=True), K + 1),
    ("linreg_r2", "close", lambda b: ind.linear_reg(b, 20, series=True).r_squared, K + 1),
    ("volume_zscore", "volume", lambda b: ind.volume_zscore(b, 20, series=True), K + 1),
    ("williams_r", "high", lambda b: ind.williams_r(b, 14, series=True), K + 1),
    ("stochastic_k", "high", lambda b: ind.stochastic(b, 14, 3, series=True)[0], K + 1),
    ("fisher", "high", lambda b: ind.fisher_transform(b, 9, series=True)[0], K + 1),
    ("adx", "high", lambda b: ind.adx(b, 14, series=True), K + 1),
])
def test_review_nan_in_window_gives_nan(bars, name, col, call, at):
    clean = call(bars).to_numpy()
    gapped = call(_gap_bars(bars, col, K)).to_numpy()
    assert np.isfinite(clean[at])
    assert np.isnan(gapped[at]), f"{name} turned a missing bar into {gapped[at]}"


@pytest.fixture(scope="module")
def flat_bars():
    n = 60
    return pl.DataFrame({k: np.full(n, 25.0) for k in ("open", "high", "low", "close")}
                        ).with_columns(pl.Series("volume", np.full(n, 1e5)))


@pytest.mark.parametrize("call", [
    lambda b: ind.mfi(b, 14, series=True),
    lambda b: ind.ultimate_oscillator(b, series=True),
    lambda b: ind.stochastic(b, 14, 3, series=True)[0],
    lambda b: ind.williams_r(b, 14, series=True),
])
def test_review_flat_prices_are_undefined_not_a_reading(flat_bars, call):
    # 0 gains over 0 losses (or a zero range) is 0/0; these used to read 0, 50 or -50.
    assert np.isnan(call(flat_bars).to_numpy()).all()


def test_review_flat_after_moves_warns_and_returns_nan(bars, flat_bars):
    moving = bars.head(100)
    flat_tail = pl.concat([moving, flat_bars.with_columns(
        [pl.lit(float(moving["close"][-1])).alias(c) for c in ("open", "high", "low", "close")])])
    with pytest.warns(RuntimeWarning, match="latest value is NaN"):
        assert math.isnan(ind.williams_r(flat_tail, 14))


def test_review_momentum_pickles_and_documents_the_indicator(ohlcv):
    import pickle

    again = pickle.loads(pickle.dumps(ind.momentum))
    assert again is ind.momentum
    assert again(ohlcv["close"], 10) == ind.momentum(ohlcv["close"], 10)
    assert ind.momentum.__doc__.startswith("Simple price momentum")


def test_review_sma_matches_a_plain_window_mean_with_gaps():
    rng = np.random.default_rng(7)
    x = rng.normal(100, 5, 2000)
    x[[0, 1, 500, 1500]] = np.nan
    ours = tr.sma_series(x, 30)
    ref = np.full(x.size, np.nan)
    for t in range(29, x.size):
        w = x[t - 29 : t + 1]
        ref[t] = w.mean() if np.isfinite(w).all() else np.nan
    np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name, col, call, at", [
    ("cmo", "close", lambda b: ind.cmo(b, 14, series=True), K + 1),
    ("cmf_volume", "volume", lambda b: ind.cmf(b, 20, series=True), K + 1),
    ("cmf_high", "high", lambda b: ind.cmf(b, 20, series=True), K + 1),
    ("aroon_osc", "high", lambda b: ind.aroon_osc(b, 14, series=True), K + 1),
    ("aroon_up", "high", lambda b: ind.aroon(b, 14, series=True)[1], K + 1),
    ("aroon_down", "low", lambda b: ind.aroon(b, 14, series=True)[0], K + 1),
    ("nvi", "volume", lambda b: ind.nvi(b, series=True), K + 1),
    ("pvi", "close", lambda b: ind.pvi(b, series=True), K + 1),
])
def test_review_more_indicators_keep_a_gap_nan(bars, name, col, call, at):
    clean = call(bars).to_numpy()
    gapped = call(_gap_bars(bars, col, K)).to_numpy()
    assert np.isfinite(clean[at])
    assert np.isnan(gapped[at]), f"{name} turned a missing bar into {gapped[at]}"


def test_review_rolling_corr_and_beta_keep_a_gap_nan(ohlcv):
    r = np.diff(ohlcv["close"]) / ohlcv["close"][:-1]
    b = np.roll(r, 1)
    r_gap = r.copy()
    r_gap[K] = np.nan
    assert np.isfinite(ind.rolling_corr(r, b, 20, series=True).to_numpy()[K + 1])
    assert np.isnan(ind.rolling_corr(r_gap, b, 20, series=True).to_numpy()[K + 1])
    assert np.isnan(ind.rolling_beta(r_gap, b, 20, series=True).to_numpy()[K + 1])
    assert np.isnan(ind.rolling_beta(b, r_gap, 20, series=True).to_numpy()[K + 1])
