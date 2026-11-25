"""CPZ Indicators — local technical indicator computation with optional Rust acceleration.

Usage::

    from cpz_quant.indicators import sma, rsi, atr, bollinger, macd

    bars = data["NVDA"]  # DataFrame with OHLCV columns

    # Scalar (latest value) — for signal logic
    current_rsi = rsi(bars, period=14)

    # Full series — for crossover detection, plotting
    rsi_history = rsi(bars, period=14, series=True)

Every function accepts a polars or pandas DataFrame (with configurable
column names), a polars or pandas Series, a 1-D numpy array, or a plain list.
Parameters such as period, smoothing method, and source price are always
configurable; there are no hard-coded magic numbers.

The scalar form returns the value at the latest bar. When that value is
undefined (a missing bar inside the look-back window, or before it for a
recursive smoother such as EMA, RSI or ATR) it is NaN with a
``RuntimeWarning``; an older value is never returned in its place. An empty
input raises ``ValueError`` in the scalar form.
"""

from __future__ import annotations

import importlib
import inspect
import types
import warnings
from typing import Any, Optional, Tuple, Union

import numpy as np
import polars as pl

# Re-exported for reference. These are the defaults bound into the function
# signatures at import; they are not global settings, so reassigning them
# changes nothing. Pass ``trading_days=`` (or the column names) per call.
from ._helpers import (  # noqa: F811
    ANNUALIZATION_FACTOR,
    DEFAULT_COL_CLOSE,
    DEFAULT_COL_HIGH,
    DEFAULT_COL_LOW,
    DEFAULT_COL_OPEN,
    DEFAULT_COL_VOLUME,
    EPSILON,
    TRADING_DAYS_PER_YEAR,
    ensure_array,
    ensure_hlc,
    ensure_ohlc,
    ensure_ohlcv,
    is_frame,
    require_frame,
    source_price,
)
from ._helpers import _col_to_numpy as _column
from ._rust_accel import (
    fast_adx,
    fast_atr,
    fast_bollinger,
    fast_ema,
    fast_garman_klass,
    fast_kama,
    fast_macd,
    fast_rsi,
    fast_sma,
    fast_wma,
    has_rust,
)

# ── Internal helpers ─────────────────────────────────────────────────

def _to_output(arr: np.ndarray, bars: Any, series: bool):
    """Convert result array to scalar or ``pl.Series`` based on *series* flag.

    Polars Series have no index — the returned series is positional and
    has the same length as *arr*.

    The scalar is the value at the latest bar. It is NaN when the latest bar
    has no value, with a ``RuntimeWarning`` if earlier bars did: returning
    the last non-NaN value instead would hand back a number from before a
    missing bar as if it were current. An empty input has no latest bar and
    raises ``ValueError``.
    """
    if series:
        return pl.Series(values=np.asarray(arr, dtype=np.float64), dtype=pl.Float64)
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == 0:
        raise ValueError("no bars: the input is empty, so there is no latest value")
    latest = float(arr[-1])
    if np.isnan(latest) and not np.all(np.isnan(arr)):
        warnings.warn(
            "the latest value is NaN: a missing or undefined input inside the look-back "
            "window (or anywhere before it, for a recursive smoother such as EMA, RSI or "
            "ATR) leaves the latest bar without a value. Fill or drop the missing bars; "
            "earlier values are not returned in its place.",
            RuntimeWarning,
            stacklevel=3,
        )
    return latest


def _all_finite(*arrays: np.ndarray) -> bool:
    """The Rust kernels assume complete data; NaN input takes the NumPy path."""
    return all(bool(np.all(np.isfinite(a))) for a in arrays)


def _to_tuple_output(arrays: tuple, bars: Any, series: bool):
    """Convert a tuple of result arrays."""
    return tuple(_to_output(a, bars, series) for a in arrays)


# ═══════════════════════════════════════════════════════════════════
#  TREND INDICATORS
# ═══════════════════════════════════════════════════════════════════

def sma(
    bars: Any,
    period: int = 20,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Simple Moving Average.

    Args:
        bars: DataFrame, Series, array, or list of prices.
        period: Look-back window.
        source: Price source (``"close"``, ``"hl2"``, ``"hlc3"``, ``"ohlc4"``).
        series: Return full history as ``pl.Series`` instead of latest scalar.
        col_*: Column name overrides for non-standard DataFrames.
    """
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    rust_result = fast_sma(close.tolist(), period) if _all_finite(close) else None
    if rust_result is not None:
        result = np.array(rust_result)
    else:
        from .trend import sma_series
        result = sma_series(close, period)
    return _to_output(result, bars, series)


def ema(
    bars: Any,
    period: int = 20,
    *,
    alpha: float | None = None,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Exponential Moving Average.

    Seeded with the first valid price (``ema[0] = close[0]``), so it has no
    warm-up gap. TA-Lib and TradingView seed with the simple mean of the
    first *period* prices instead; the two agree once the seed has decayed
    (it weighs ``(1 - alpha)^k`` after ``k`` bars, about 1% after
    ``2.3 * (period + 1) / 2`` bars for the default alpha).

    Args:
        bars: DataFrame, Series, array, or list of prices.
        period: Look-back window (used to derive alpha when not given).
        alpha: Explicit smoothing factor — overrides ``2/(period+1)``.
        source: Price source.
        series: Return full history.
        col_*: Column name overrides.
    """
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    if alpha is None and _all_finite(close):
        rust_result = fast_ema(close.tolist(), period)
        if rust_result is not None:
            return _to_output(np.array(rust_result), bars, series)
    from .trend import ema_series
    result = ema_series(close, period, alpha=alpha)
    return _to_output(result, bars, series)


def wma(
    bars: Any,
    period: int = 20,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Weighted Moving Average (linearly increasing weights)."""
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    rust_result = fast_wma(close.tolist(), period) if _all_finite(close) else None
    if rust_result is not None:
        return _to_output(np.array(rust_result), bars, series)
    from .trend import wma_series
    return _to_output(wma_series(close, period), bars, series)


def dema(
    bars: Any,
    period: int = 20,
    *,
    alpha: float | None = None,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Double Exponential Moving Average."""
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    from .trend import dema_series
    return _to_output(dema_series(close, period, alpha=alpha), bars, series)


def tema(
    bars: Any,
    period: int = 20,
    *,
    alpha: float | None = None,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Triple Exponential Moving Average."""
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    from .trend import tema_series
    return _to_output(tema_series(close, period, alpha=alpha), bars, series)


def kama(
    bars: Any,
    period: int = 10,
    *,
    fast_period: int = 2,
    slow_period: int = 30,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Kaufman Adaptive Moving Average."""
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    from .trend import kama_series
    return _to_output(kama_series(close, period, fast_period=fast_period, slow_period=slow_period), bars, series)


def supertrend(
    bars: Any,
    period: int = 10,
    *,
    multiplier: float = 3.0,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Union[float, "pl.Series"]:
    """Supertrend indicator."""
    high, low, close = ensure_hlc(bars, col_high=col_high, col_low=col_low, col_close=col_close)
    from .trend import supertrend_series
    return _to_output(supertrend_series(high, low, close, period, multiplier=multiplier), bars, series)


# ═══════════════════════════════════════════════════════════════════
#  MOMENTUM INDICATORS
# ═══════════════════════════════════════════════════════════════════

def rsi(
    bars: Any,
    period: int = 14,
    *,
    smoothing: str = "wilder",
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Relative Strength Index (0-100).

    NaN where the price has not moved at all over the averaging window
    (0 gains over 0 losses is undefined; TA-Lib returns 0, an oversold
    reading). See ``momentum.rsi_series`` for the definition.

    Args:
        bars: Price data.
        period: RSI period.
        smoothing: ``"wilder"`` (default) or ``"ema"``.
        source: Price source.
    """
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    if smoothing == "wilder" and _all_finite(close):
        rust_result = fast_rsi(close.tolist(), period)
        if rust_result is not None:
            return _to_output(np.array(rust_result), bars, series)
    from .momentum import rsi_series
    return _to_output(rsi_series(close, period, smoothing=smoothing), bars, series)


class MACDResult:
    """Container for MACD output values."""
    __slots__ = ("macd", "signal", "histogram")

    def __init__(self, macd: Any, signal: Any, histogram: Any):
        self.macd = macd
        self.signal = signal
        self.histogram = histogram

    def __iter__(self):
        return iter((self.macd, self.signal, self.histogram))

    def __repr__(self) -> str:
        return f"MACDResult(macd={self.macd}, signal={self.signal}, histogram={self.histogram})"


def macd(
    bars: Any,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> MACDResult:
    """Moving Average Convergence Divergence.

    The fast, slow and signal EMAs are seeded with their first valid input
    (see ``ema``), not with TA-Lib's simple mean of the first *period*
    values. Early values, including the histogram's sign, can therefore
    differ from TA-Lib ``MACD`` and from ``apo()`` (TA-Lib seeding) until
    the seeds have decayed; use a history several times ``slow_period``
    long when that matters.

    Returns:
        ``MACDResult(macd, signal, histogram)`` — each is a scalar or
        ``pl.Series`` depending on *series*.
    """
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    rust_result = (fast_macd(close.tolist(), fast_period, slow_period, signal_period)
                   if _all_finite(close) else None)
    if rust_result is not None:
        m, s, h = rust_result
        arrs = (np.array(m), np.array(s), np.array(h))
    else:
        from .momentum import macd_series
        arrs = macd_series(close, fast_period, slow_period, signal_period)
    vals = _to_tuple_output(arrs, bars, series)
    return MACDResult(*vals)


def stochastic(
    bars: Any,
    k_period: int = 14,
    d_period: int = 3,
    *,
    d_ma_type: str = "sma",
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Tuple[Any, Any]:
    """Stochastic Oscillator. Returns ``(percent_k, percent_d)``."""
    high, low, close = ensure_hlc(bars, col_high=col_high, col_low=col_low, col_close=col_close)
    from .momentum import stochastic_series
    k, d = stochastic_series(high, low, close, k_period, d_period, d_ma_type=d_ma_type)
    return _to_output(k, bars, series), _to_output(d, bars, series)


def williams_r(
    bars: Any,
    period: int = 14,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Union[float, "pl.Series"]:
    """Williams %R oscillator."""
    high, low, close = ensure_hlc(bars, col_high=col_high, col_low=col_low, col_close=col_close)
    from .momentum import williams_r_series
    return _to_output(williams_r_series(high, low, close, period), bars, series)


def cci(
    bars: Any,
    period: int = 20,
    *,
    constant: float = 0.015,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Union[float, "pl.Series"]:
    """Commodity Channel Index."""
    high, low, close = ensure_hlc(bars, col_high=col_high, col_low=col_low, col_close=col_close)
    from .momentum import cci_series
    return _to_output(cci_series(high, low, close, period, constant=constant), bars, series)


def roc(
    bars: Any,
    period: int = 12,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Rate of Change (percentage)."""
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    from .momentum import roc_series
    return _to_output(roc_series(close, period), bars, series)


def momentum(
    bars: Any,
    period: int = 20,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Simple price momentum (percentage return over *period* bars)."""
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    from .momentum import momentum_pct_series
    return _to_output(momentum_pct_series(close, period), bars, series)


def adx(
    bars: Any,
    period: int = 14,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Union[float, "pl.Series"]:
    """Average Directional Index."""
    high, low, close = ensure_hlc(bars, col_high=col_high, col_low=col_low, col_close=col_close)
    rust_result = (fast_adx(high.tolist(), low.tolist(), close.tolist(), period)
                   if _all_finite(high, low, close) else None)
    if rust_result is not None:
        return _to_output(np.array(rust_result), bars, series)
    from .momentum import adx_series
    return _to_output(adx_series(high, low, close, period), bars, series)


def mfi(
    bars: Any,
    period: int = 14,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Union[float, "pl.Series"]:
    """Money Flow Index (volume-weighted RSI)."""
    high, low, close, vol = _cols(bars, "mfi", col_high, col_low, col_close, col_volume)
    from .momentum import mfi_series
    return _to_output(mfi_series(high, low, close, vol, period), bars, series)


# ═══════════════════════════════════════════════════════════════════
#  VOLATILITY INDICATORS
# ═══════════════════════════════════════════════════════════════════

def atr(
    bars: Any,
    period: int = 14,
    *,
    smoothing: str = "wilder",
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Union[float, "pl.Series"]:
    """Average True Range.

    Args:
        bars: DataFrame with OHLC data.
        period: ATR period.
        smoothing: ``"wilder"`` (default), ``"ema"``, or ``"sma"``.
    """
    high, low, close = ensure_hlc(bars, col_high=col_high, col_low=col_low, col_close=col_close)
    if smoothing == "wilder" and _all_finite(high, low, close):
        rust_result = fast_atr(high.tolist(), low.tolist(), close.tolist(), period)
        if rust_result is not None:
            return _to_output(np.array(rust_result), bars, series)
    from .volatility import atr_series
    return _to_output(atr_series(high, low, close, period, smoothing=smoothing), bars, series)


class BollingerResult:
    """Container for Bollinger Bands output."""
    __slots__ = ("upper", "middle", "lower")

    def __init__(self, upper: Any, middle: Any, lower: Any):
        self.upper = upper
        self.middle = middle
        self.lower = lower

    def __iter__(self):
        return iter((self.upper, self.middle, self.lower))

    def __repr__(self) -> str:
        return f"BollingerResult(upper={self.upper}, middle={self.middle}, lower={self.lower})"


def bollinger(
    bars: Any,
    period: int = 20,
    *,
    num_std: float = 2.0,
    ma_type: str = "sma",
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> BollingerResult:
    """Bollinger Bands.

    Returns:
        ``BollingerResult(upper, middle, lower)``.
    """
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    if ma_type == "sma" and _all_finite(close):
        rust_result = fast_bollinger(close.tolist(), period, num_std)
        if rust_result is not None:
            u, m, l = rust_result
            arrs = (np.array(u), np.array(m), np.array(l))
            vals = _to_tuple_output(arrs, bars, series)
            return BollingerResult(*vals)
    from .volatility import bollinger_series
    arrs = bollinger_series(close, period, num_std=num_std, ma_type=ma_type)
    vals = _to_tuple_output(arrs, bars, series)
    return BollingerResult(*vals)


def keltner(
    bars: Any,
    period: int = 20,
    *,
    atr_period: int = 10,
    multiplier: float = 1.5,
    atr_smoothing: str = "wilder",
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Tuple[Any, Any, Any]:
    """Keltner Channels. Returns ``(upper, middle, lower)``."""
    high, low, close = ensure_hlc(bars, col_high=col_high, col_low=col_low, col_close=col_close)
    from .volatility import keltner_series
    arrs = keltner_series(high, low, close, period, atr_period=atr_period,
                          multiplier=multiplier, atr_smoothing=atr_smoothing)
    return _to_tuple_output(arrs, bars, series)


def donchian(
    bars: Any,
    period: int = 20,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
) -> Tuple[Any, Any]:
    """Donchian Channels. Returns ``(upper, lower)``.

    On a DataFrame the channels use the high and low columns; on a single
    series both channels use that series.
    """
    if is_frame(bars):
        high, low = _cols(bars, "donchian", col_high, col_low)
    else:
        high = ensure_array(bars)
        low = ensure_array(bars)
    from .volatility import donchian_series
    u, l = donchian_series(high, low, period)
    return _to_output(u, bars, series), _to_output(l, bars, series)


def realized_vol(
    bars: Any,
    period: int = 20,
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Realized / historical volatility."""
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    from .volatility import realized_vol_series
    return _to_output(realized_vol_series(close, period, annualize=annualize, trading_days=trading_days), bars, series)


def garman_klass(
    bars: Any,
    period: int = 20,
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    series: bool = False,
    col_open: str = DEFAULT_COL_OPEN,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Union[float, "pl.Series"]:
    """Garman-Klass volatility estimator (OHLC-based, more efficient than close-to-close)."""
    o, h, l, c = ensure_ohlc(bars, col_open=col_open, col_high=col_high, col_low=col_low, col_close=col_close)
    # The Rust kernel always annualises with 252 days: use it only then.
    rust_result = (fast_garman_klass(o.tolist(), h.tolist(), l.tolist(), c.tolist(), period)
                   if annualize and trading_days == 252 and _all_finite(o, h, l, c) else None)
    if rust_result is not None:
        return _to_output(np.array(rust_result), bars, series)
    from .volatility import garman_klass_series
    return _to_output(garman_klass_series(o, h, l, c, period, annualize=annualize, trading_days=trading_days), bars, series)


def natr(
    bars: Any,
    period: int = 14,
    *,
    smoothing: str = "wilder",
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Union[float, "pl.Series"]:
    """Normalised ATR (ATR / close * 100)."""
    high, low, close = ensure_hlc(bars, col_high=col_high, col_low=col_low, col_close=col_close)
    from .volatility import natr_series
    return _to_output(natr_series(high, low, close, period, smoothing=smoothing), bars, series)


# ═══════════════════════════════════════════════════════════════════
#  VOLUME INDICATORS
# ═══════════════════════════════════════════════════════════════════

def vwap(
    bars: Any,
    *,
    session_col: Optional[str] = None,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Union[float, "pl.Series"]:
    """Volume-Weighted Average Price.

    Args:
        session_col: Optional column that anchors the VWAP. When given, the
            cumulative sums restart whenever the session changes. A
            ``Datetime`` column is anchored per calendar date (convert
            timestamps to the exchange time zone first); a ``Date`` or any
            other column is used as the session label directly. Without it
            the VWAP is cumulative over the whole frame.
    """
    if session_col is None:
        h, l, c, v = ensure_ohlcv(bars, col_high=col_high, col_low=col_low,
                                   col_close=col_close, col_volume=col_volume)[1:]
        from .volume import vwap_series
        return _to_output(vwap_series(h, l, c, v), bars, series)
    h, l, c, v = _cols(bars, "vwap", col_high, col_low, col_close, col_volume)
    session = bars[session_col]
    if isinstance(session, pl.Series):
        if session.dtype == pl.Datetime:
            session = session.dt.date()
        labels = session.to_numpy()
    elif str(session.dtype).startswith("datetime64"):
        labels = session.dt.date.to_numpy()  # pandas timestamps: anchor per calendar date
    else:
        labels = session.to_numpy()
    from .volume import anchored_vwap_series
    return _to_output(anchored_vwap_series(h, l, c, v, labels), bars, series)


def obv(
    bars: Any,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_volume: str = DEFAULT_COL_VOLUME,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """On-Balance Volume. Needs a DataFrame with price and volume columns."""
    require_frame(bars, "obv", (col_close, col_volume))
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    vol = _column(bars, col_volume)
    from .volume import obv_series
    return _to_output(obv_series(close, vol), bars, series)


def cmf(
    bars: Any,
    period: int = 20,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Union[float, "pl.Series"]:
    """Chaikin Money Flow."""
    h, l, c, v = _cols(bars, "cmf", col_high, col_low, col_close, col_volume)
    from .volume import cmf_series
    return _to_output(cmf_series(h, l, c, v, period), bars, series)


def volume_zscore(
    bars: Any,
    period: int = 20,
    *,
    series: bool = False,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Union[float, "pl.Series"]:
    """Volume z-score relative to rolling mean."""
    vol = _column(bars, col_volume) if is_frame(bars) else ensure_array(bars)
    from .volume import volume_zscore_series
    return _to_output(volume_zscore_series(vol, period), bars, series)


# ═══════════════════════════════════════════════════════════════════
#  STATISTICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════

def zscore(
    data: Any,
    period: int = 60,
    *,
    ddof: int = 1,
    series: bool = False,
    col: Optional[str] = None,
) -> Union[float, "pl.Series"]:
    """Rolling z-score of any series."""
    values = ensure_array(data, col)
    from .statistical import zscore_series
    return _to_output(zscore_series(values, period, ddof=ddof), data, series)


def rolling_corr(
    series_a: Any,
    series_b: Any,
    period: int = 60,
    *,
    series: bool = False,
    col_a: Optional[str] = None,
    col_b: Optional[str] = None,
) -> Union[float, "pl.Series"]:
    """Rolling Pearson correlation between two series."""
    a = ensure_array(series_a, col_a)
    b = ensure_array(series_b, col_b)
    from .statistical import rolling_corr_series
    return _to_output(rolling_corr_series(a, b, period), series_a, series)


def rolling_beta(
    asset_returns: Any,
    benchmark_returns: Any,
    period: int = 60,
    *,
    series: bool = False,
    col_asset: Optional[str] = None,
    col_bench: Optional[str] = None,
) -> Union[float, "pl.Series"]:
    """Rolling OLS beta."""
    a = ensure_array(asset_returns, col_asset)
    b = ensure_array(benchmark_returns, col_bench)
    from .statistical import rolling_beta_series
    return _to_output(rolling_beta_series(a, b, period), asset_returns, series)


def hurst(
    bars: Any,
    max_lag: int = 100,
    *,
    min_lag: int = 10,
    source: str = "close",
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> float:
    """Hurst exponent (H<0.5 mean-reverting, H=0.5 random, H>0.5 trending).

    Bias-corrected rescaled range (Anis-Lloyd-Peters), so a random walk
    reads about 0.5; see ``statistical.hurst_series``. NaN when the series
    is too short for two block sizes between *min_lag* and *max_lag*.
    """
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    from .statistical import hurst_series
    return hurst_series(close, max_lag, min_lag=min_lag)


class LinRegResult:
    """Container for linear regression output."""
    __slots__ = ("slope", "r_squared", "intercept", "forecast")

    def __init__(self, slope: Any, r_squared: Any, intercept: Any, forecast: Any):
        self.slope = slope
        self.r_squared = r_squared
        self.intercept = intercept
        self.forecast = forecast

    def __iter__(self):
        return iter((self.slope, self.r_squared, self.intercept, self.forecast))

    def __repr__(self) -> str:
        return (f"LinRegResult(slope={self.slope}, r_squared={self.r_squared}, "
                f"intercept={self.intercept}, forecast={self.forecast})")


def linear_reg(
    bars: Any,
    period: int = 20,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> LinRegResult:
    """Rolling linear regression. Returns ``LinRegResult(slope, r_squared, intercept, forecast)``."""
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    from .statistical import linear_reg_series
    arrs = linear_reg_series(close, period)
    vals = _to_tuple_output(arrs, bars, series)
    return LinRegResult(*vals)


# ═══════════════════════════════════════════════════════════════════
#  EXTENDED INDICATORS
#
#  Shared input helpers keep the wrappers below short. Every wrapper
#  follows the same contract as the functions above: ``series=False``
#  returns the value at the latest bar (NaN, with a warning, if that bar
#  has none), ``series=True`` returns the full positional ``pl.Series``
#  with a leading NaN warm-up.
# ═══════════════════════════════════════════════════════════════════

def _price(
    bars: Any,
    source: str,
    col_open: str,
    col_high: str,
    col_low: str,
    col_close: str,
) -> np.ndarray:
    if source != "close":
        return source_price(bars, source, col_open=col_open, col_high=col_high,
                            col_low=col_low, col_close=col_close)
    return ensure_array(bars, col_close)


def _require_frame(bars: Any, fn: str) -> Any:
    return require_frame(bars, fn, ("open", "high", "low", "close", "volume"))


def _cols(bars: Any, fn: str, *cols: str) -> Tuple[np.ndarray, ...]:
    df = require_frame(bars, fn, cols)
    return tuple(_column(df, c) for c in cols)


# ── Extended trend ───────────────────────────────────────────────────

def trima(
    bars: Any,
    period: int = 30,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Triangular Moving Average (matches TA-Lib ``TRIMA``)."""
    from .trend import trima_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(trima_series(x, period), bars, series)


def t3(
    bars: Any,
    period: int = 5,
    *,
    vfactor: float = 0.7,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Tillson T3 moving average (matches TA-Lib ``T3``)."""
    from .trend import t3_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(t3_series(x, period, vfactor=vfactor), bars, series)


def zlema(
    bars: Any,
    period: int = 20,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Zero-Lag EMA (matches TA-Lib ``ZLEMA`` and pandas-ta ``zlma``)."""
    from .trend import zlema_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(zlema_series(x, period), bars, series)


def hma(
    bars: Any,
    period: int = 20,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Hull Moving Average (matches TA-Lib ``HMA`` and pandas-ta ``hma``)."""
    from .trend import hma_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(hma_series(x, period), bars, series)


def alma(
    bars: Any,
    period: int = 9,
    *,
    offset: float = 0.85,
    sigma: float = 6.0,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Arnaud Legoux Moving Average (matches pandas-ta ``alma``)."""
    from .trend import alma_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(alma_series(x, period, offset=offset, sigma=sigma), bars, series)


def vidya(
    bars: Any,
    period: int = 14,
    *,
    cmo_period: int = 9,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Chande's Variable Index Dynamic Average (CMO-driven EMA)."""
    from .trend import vidya_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(vidya_series(x, period, cmo_period=cmo_period), bars, series)


def mcginley(
    bars: Any,
    period: int = 10,
    *,
    k: float = 0.6,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """McGinley Dynamic."""
    from .trend import mcginley_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(mcginley_series(x, period, k=k), bars, series)


def psar(
    bars: Any,
    *,
    acceleration: float = 0.02,
    maximum: float = 0.2,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
) -> Union[float, "pl.Series"]:
    """Parabolic SAR (matches TA-Lib ``SAR``)."""
    from .trend import psar_series
    h, l = _cols(bars, "psar", col_high, col_low)
    return _to_output(psar_series(h, l, acceleration=acceleration, maximum=maximum), bars, series)


class IchimokuResult:
    """Container for Ichimoku output (no Chikou span, see ``ichimoku_series``)."""
    __slots__ = ("tenkan", "kijun", "senkou_a", "senkou_b")

    def __init__(self, tenkan: Any, kijun: Any, senkou_a: Any, senkou_b: Any):
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_a = senkou_a
        self.senkou_b = senkou_b

    def __iter__(self):
        return iter((self.tenkan, self.kijun, self.senkou_a, self.senkou_b))

    def __repr__(self) -> str:
        return (f"IchimokuResult(tenkan={self.tenkan}, kijun={self.kijun}, "
                f"senkou_a={self.senkou_a}, senkou_b={self.senkou_b})")


def ichimoku(
    bars: Any,
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
    *,
    displacement: int = 26,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
) -> IchimokuResult:
    """Ichimoku Kinko Hyo (matches pandas-ta ``ichimoku``, no look-ahead)."""
    from .trend import ichimoku_series
    h, l = _cols(bars, "ichimoku", col_high, col_low)
    arrs = ichimoku_series(h, l, tenkan, kijun, senkou, displacement=displacement)
    return IchimokuResult(*_to_tuple_output(arrs, bars, series))


def aroon(
    bars: Any,
    period: int = 14,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
) -> Tuple[Any, Any]:
    """Aroon indicator. Returns ``(aroon_down, aroon_up)`` (matches TA-Lib ``AROON``)."""
    from .trend import aroon_series
    h, l = _cols(bars, "aroon", col_high, col_low)
    return _to_tuple_output(aroon_series(h, l, period), bars, series)


def aroon_osc(
    bars: Any,
    period: int = 14,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
) -> Union[float, "pl.Series"]:
    """Aroon Oscillator (matches TA-Lib ``AROONOSC``)."""
    from .trend import aroon_osc_series
    h, l = _cols(bars, "aroon_osc", col_high, col_low)
    return _to_output(aroon_osc_series(h, l, period), bars, series)


def vortex(
    bars: Any,
    period: int = 14,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Tuple[Any, Any]:
    """Vortex Indicator. Returns ``(vi_plus, vi_minus)`` (matches TA-Lib ``VORTEX``)."""
    from .trend import vortex_series
    high, low, close = ensure_hlc(bars, col_high=col_high, col_low=col_low, col_close=col_close)
    return _to_tuple_output(vortex_series(high, low, close, period), bars, series)


def trix(
    bars: Any,
    period: int = 30,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """TRIX, percent rate of change of a triple EMA (matches TA-Lib ``TRIX``)."""
    from .trend import trix_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(trix_series(x, period), bars, series)


def mass_index(
    bars: Any,
    fast_period: int = 9,
    slow_period: int = 25,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
) -> Union[float, "pl.Series"]:
    """Mass Index (matches TA-Lib ``MASSI`` and pandas-ta ``massi``)."""
    from .trend import mass_index_series
    h, l = _cols(bars, "mass_index", col_high, col_low)
    return _to_output(mass_index_series(h, l, fast_period, slow_period), bars, series)


def midpoint(
    bars: Any,
    period: int = 14,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Midpoint of the rolling price range (matches TA-Lib ``MIDPOINT``)."""
    from .trend import midpoint_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(midpoint_series(x, period), bars, series)


def midprice(
    bars: Any,
    period: int = 14,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
) -> Union[float, "pl.Series"]:
    """Midpoint of rolling highest high and lowest low (matches TA-Lib ``MIDPRICE``)."""
    from .trend import midprice_series
    h, l = _cols(bars, "midprice", col_high, col_low)
    return _to_output(midprice_series(h, l, period), bars, series)


def stc(
    bars: Any,
    cycle: int = 10,
    fast_period: int = 23,
    slow_period: int = 50,
    *,
    factor: float = 0.5,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Schaff Trend Cycle (0..100)."""
    from .trend import stc_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(stc_series(x, cycle, fast_period, slow_period, factor=factor), bars, series)


# ── Extended momentum ────────────────────────────────────────────────

def stochastic_slow(
    bars: Any,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowd_period: int = 3,
    *,
    ma_type: str = "sma",
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Tuple[Any, Any]:
    """Slow Stochastic. Returns ``(slow_k, slow_d)`` (matches TA-Lib ``STOCH``)."""
    from .momentum import stochastic_slow_series
    high, low, close = ensure_hlc(bars, col_high=col_high, col_low=col_low, col_close=col_close)
    arrs = stochastic_slow_series(high, low, close, fastk_period, slowk_period, slowd_period,
                                  ma_type=ma_type)
    return _to_tuple_output(arrs, bars, series)


def stoch_rsi(
    bars: Any,
    period: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
    *,
    ma_type: str = "sma",
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Tuple[Any, Any]:
    """Stochastic RSI. Returns ``(fast_k, fast_d)`` (matches TA-Lib ``STOCHRSI``)."""
    from .momentum import stochrsi_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    arrs = stochrsi_series(x, period, fastk_period, fastd_period, ma_type=ma_type)
    return _to_tuple_output(arrs, bars, series)


def cmo(
    bars: Any,
    period: int = 14,
    *,
    smoothing: str = "sum",
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Chande Momentum Oscillator.

    ``smoothing="sum"`` is Chande's definition (TA-Lib ``CMOU``);
    ``smoothing="wilder"`` matches TA-Lib ``CMO``.
    """
    from .momentum import cmo_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(cmo_series(x, period, smoothing=smoothing), bars, series)


def mom(
    bars: Any,
    period: int = 10,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Momentum as a price difference (matches TA-Lib ``MOM``)."""
    from .momentum import mom_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(mom_series(x, period), bars, series)


def rocp(
    bars: Any,
    period: int = 10,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Rate of change as a fraction (matches TA-Lib ``ROCP``)."""
    from .momentum import rocp_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(rocp_series(x, period), bars, series)


def rocr(
    bars: Any,
    period: int = 10,
    *,
    scale: float = 1.0,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Rate of change ratio (TA-Lib ``ROCR``; ``scale=100`` gives ``ROCR100``)."""
    from .momentum import rocr_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(rocr_series(x, period, scale=scale), bars, series)


def ultimate_oscillator(
    bars: Any,
    period1: int = 7,
    period2: int = 14,
    period3: int = 28,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Union[float, "pl.Series"]:
    """Ultimate Oscillator (matches TA-Lib ``ULTOSC``)."""
    from .momentum import ultosc_series
    high, low, close = ensure_hlc(bars, col_high=col_high, col_low=col_low, col_close=col_close)
    return _to_output(ultosc_series(high, low, close, period1, period2, period3), bars, series)


def awesome_oscillator(
    bars: Any,
    fast_period: int = 5,
    slow_period: int = 34,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
) -> Union[float, "pl.Series"]:
    """Awesome Oscillator (matches TA-Lib ``AO`` and pandas-ta ``ao``)."""
    from .momentum import awesome_oscillator_series
    h, l = _cols(bars, "awesome_oscillator", col_high, col_low)
    return _to_output(awesome_oscillator_series(h, l, fast_period, slow_period), bars, series)


def apo(
    bars: Any,
    fast_period: int = 12,
    slow_period: int = 26,
    *,
    ma_type: str = "ema",
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Absolute Price Oscillator (matches TA-Lib ``APO`` for the same MA type)."""
    from .momentum import apo_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(apo_series(x, fast_period, slow_period, ma_type=ma_type), bars, series)


def ppo(
    bars: Any,
    fast_period: int = 12,
    slow_period: int = 26,
    *,
    ma_type: str = "ema",
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Percentage Price Oscillator (matches TA-Lib ``PPO`` for the same MA type)."""
    from .momentum import ppo_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(ppo_series(x, fast_period, slow_period, ma_type=ma_type), bars, series)


def kst(
    bars: Any,
    roc_periods: Tuple[int, int, int, int] = (10, 15, 20, 30),
    sma_periods: Tuple[int, int, int, int] = (10, 10, 10, 15),
    signal_period: int = 9,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Tuple[Any, Any]:
    """Know Sure Thing. Returns ``(kst, signal)``."""
    from .momentum import kst_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_tuple_output(kst_series(x, roc_periods, sma_periods, signal_period), bars, series)


def tsi(
    bars: Any,
    long_period: int = 25,
    short_period: int = 13,
    signal_period: int = 13,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Tuple[Any, Any]:
    """True Strength Index. Returns ``(tsi, signal)`` (TSI matches TA-Lib ``TSI``)."""
    from .momentum import tsi_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_tuple_output(tsi_series(x, long_period, short_period, signal_period), bars, series)


def connors_rsi(
    bars: Any,
    rsi_period: int = 3,
    streak_period: int = 2,
    rank_period: int = 100,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Connors RSI (0..100)."""
    from .momentum import connors_rsi_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(connors_rsi_series(x, rsi_period, streak_period, rank_period), bars, series)


def fisher_transform(
    bars: Any,
    period: int = 9,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
) -> Tuple[Any, Any]:
    """Ehlers Fisher Transform. Returns ``(fisher, trigger)``."""
    from .momentum import fisher_transform_series
    h, l = _cols(bars, "fisher_transform", col_high, col_low)
    return _to_tuple_output(fisher_transform_series(h, l, period), bars, series)


def coppock(
    bars: Any,
    wma_period: int = 10,
    long_roc: int = 14,
    short_roc: int = 11,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Coppock Curve (matches TA-Lib ``COPPOCK`` and pandas-ta ``coppock``)."""
    from .momentum import coppock_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(coppock_series(x, wma_period, long_roc, short_roc), bars, series)


def dpo(
    bars: Any,
    period: int = 20,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Detrended Price Oscillator, no look-ahead (matches TA-Lib ``DPO``)."""
    from .momentum import dpo_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(dpo_series(x, period), bars, series)


def rvgi(
    bars: Any,
    period: int = 10,
    *,
    series: bool = False,
    col_open: str = DEFAULT_COL_OPEN,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Tuple[Any, Any]:
    """Relative Vigor Index. Returns ``(rvgi, signal)``."""
    from .momentum import rvgi_series
    o, h, l, c = ensure_ohlc(bars, col_open=col_open, col_high=col_high, col_low=col_low,
                             col_close=col_close)
    return _to_tuple_output(rvgi_series(o, h, l, c, period), bars, series)


def elder_ray(
    bars: Any,
    period: int = 13,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Tuple[Any, Any]:
    """Elder Ray Index. Returns ``(bull_power, bear_power)`` (matches TA-Lib ``ERI``)."""
    from .momentum import elder_ray_series
    high, low, close = ensure_hlc(bars, col_high=col_high, col_low=col_low, col_close=col_close)
    return _to_tuple_output(elder_ray_series(high, low, close, period), bars, series)


def wavetrend(
    bars: Any,
    channel_period: int = 10,
    average_period: int = 21,
    signal_period: int = 4,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Tuple[Any, Any]:
    """WaveTrend Oscillator. Returns ``(wt1, wt2)``."""
    from .momentum import wavetrend_series
    high, low, close = ensure_hlc(bars, col_high=col_high, col_low=col_low, col_close=col_close)
    arrs = wavetrend_series(high, low, close, channel_period, average_period, signal_period)
    return _to_tuple_output(arrs, bars, series)


def bop(
    bars: Any,
    *,
    series: bool = False,
    col_open: str = DEFAULT_COL_OPEN,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Union[float, "pl.Series"]:
    """Balance of Power (matches TA-Lib ``BOP``)."""
    from .momentum import bop_series
    o, h, l, c = ensure_ohlc(bars, col_open=col_open, col_high=col_high, col_low=col_low,
                             col_close=col_close)
    return _to_output(bop_series(o, h, l, c), bars, series)


class DMIResult:
    """Container for the Directional Movement system."""
    __slots__ = ("plus_di", "minus_di", "dx", "adx", "adxr")

    def __init__(self, plus_di: Any, minus_di: Any, dx: Any, adx: Any, adxr: Any):
        self.plus_di = plus_di
        self.minus_di = minus_di
        self.dx = dx
        self.adx = adx
        self.adxr = adxr

    def __iter__(self):
        return iter((self.plus_di, self.minus_di, self.dx, self.adx, self.adxr))

    def __repr__(self) -> str:
        return (f"DMIResult(plus_di={self.plus_di}, minus_di={self.minus_di}, dx={self.dx}, "
                f"adx={self.adx}, adxr={self.adxr})")


def dmi(
    bars: Any,
    period: int = 14,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> DMIResult:
    """Directional Movement Index: +DI, -DI, DX, ADX, ADXR.

    Every component matches the TA-Lib function of the same name. The
    standalone ``adx()`` predates this and seeds its smoothing differently,
    so its first values differ slightly from ``dmi().adx``.
    """
    from .momentum import adx_talib_series, dmi_components_series
    high, low, close = ensure_hlc(bars, col_high=col_high, col_low=col_low, col_close=col_close)
    _, _, plus_di, minus_di, dx = dmi_components_series(high, low, close, period)
    adx_arr, adxr_arr = adx_talib_series(high, low, close, period)
    vals = _to_tuple_output((plus_di, minus_di, dx, adx_arr, adxr_arr), bars, series)
    return DMIResult(*vals)


# ── Extended volatility ──────────────────────────────────────────────

def chaikin_volatility(
    bars: Any,
    period: int = 10,
    roc_period: int = 10,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
) -> Union[float, "pl.Series"]:
    """Chaikin Volatility (matches TA-Lib ``CVI``)."""
    from .volatility import chaikin_volatility_series
    h, l = _cols(bars, "chaikin_volatility", col_high, col_low)
    return _to_output(chaikin_volatility_series(h, l, period, roc_period), bars, series)


def ulcer_index(
    bars: Any,
    period: int = 14,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Ulcer Index (matches pandas-ta ``ui``)."""
    from .volatility import ulcer_index_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(ulcer_index_series(x, period), bars, series)


def choppiness(
    bars: Any,
    period: int = 14,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Union[float, "pl.Series"]:
    """Choppiness Index (matches pandas-ta ``chop``)."""
    from .volatility import choppiness_series
    high, low, close = ensure_hlc(bars, col_high=col_high, col_low=col_low, col_close=col_close)
    return _to_output(choppiness_series(high, low, close, period), bars, series)


def parkinson(
    bars: Any,
    period: int = 20,
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
) -> Union[float, "pl.Series"]:
    """Parkinson high-low volatility estimator."""
    from .volatility import parkinson_series
    h, l = _cols(bars, "parkinson", col_high, col_low)
    arr = parkinson_series(h, l, period, annualize=annualize, trading_days=trading_days)
    return _to_output(arr, bars, series)


def rogers_satchell(
    bars: Any,
    period: int = 20,
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    series: bool = False,
    col_open: str = DEFAULT_COL_OPEN,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Union[float, "pl.Series"]:
    """Rogers-Satchell volatility estimator."""
    from .volatility import rogers_satchell_series
    o, h, l, c = ensure_ohlc(bars, col_open=col_open, col_high=col_high, col_low=col_low,
                             col_close=col_close)
    arr = rogers_satchell_series(o, h, l, c, period, annualize=annualize,
                                 trading_days=trading_days)
    return _to_output(arr, bars, series)


def yang_zhang(
    bars: Any,
    period: int = 20,
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    series: bool = False,
    col_open: str = DEFAULT_COL_OPEN,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Union[float, "pl.Series"]:
    """Yang-Zhang volatility estimator."""
    from .volatility import yang_zhang_series
    o, h, l, c = ensure_ohlc(bars, col_open=col_open, col_high=col_high, col_low=col_low,
                             col_close=col_close)
    arr = yang_zhang_series(o, h, l, c, period, annualize=annualize, trading_days=trading_days)
    return _to_output(arr, bars, series)


# ── Extended volume ──────────────────────────────────────────────────

def vwma(
    bars: Any,
    period: int = 20,
    *,
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Union[float, "pl.Series"]:
    """Volume-Weighted Moving Average (matches TA-Lib ``VWMA``)."""
    from .volume import vwma_series
    c, v = _cols(bars, "vwma", col_close, col_volume)
    return _to_output(vwma_series(c, v, period), bars, series)


def ad_line(
    bars: Any,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Union[float, "pl.Series"]:
    """Chaikin Accumulation/Distribution Line (matches TA-Lib ``AD``)."""
    from .volume import ad_series
    h, l, c, v = _cols(bars, "ad_line", col_high, col_low, col_close, col_volume)
    return _to_output(ad_series(h, l, c, v), bars, series)


def adosc(
    bars: Any,
    fast_period: int = 3,
    slow_period: int = 10,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Union[float, "pl.Series"]:
    """Chaikin A/D Oscillator (matches TA-Lib ``ADOSC``)."""
    from .volume import adosc_series
    h, l, c, v = _cols(bars, "adosc", col_high, col_low, col_close, col_volume)
    return _to_output(adosc_series(h, l, c, v, fast_period, slow_period), bars, series)


def force_index(
    bars: Any,
    period: int = 13,
    *,
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Union[float, "pl.Series"]:
    """Elder's Force Index (matches TA-Lib ``EFI`` and pandas-ta ``efi``)."""
    from .volume import force_index_series
    c, v = _cols(bars, "force_index", col_close, col_volume)
    return _to_output(force_index_series(c, v, period), bars, series)


def eom(
    bars: Any,
    period: int = 14,
    *,
    divisor: float = 100_000_000.0,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Union[float, "pl.Series"]:
    """Ease of Movement (matches pandas-ta ``eom``)."""
    from .volume import eom_series
    h, l, v = _cols(bars, "eom", col_high, col_low, col_volume)
    return _to_output(eom_series(h, l, v, period, divisor=divisor), bars, series)


def klinger(
    bars: Any,
    fast_period: int = 34,
    slow_period: int = 55,
    signal_period: int = 13,
    *,
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Tuple[Any, Any]:
    """Klinger Volume Oscillator, simplified form. Returns ``(kvo, signal)``."""
    from .volume import klinger_series
    h, l, c, v = _cols(bars, "klinger", col_high, col_low, col_close, col_volume)
    arrs = klinger_series(h, l, c, v, fast_period, slow_period, signal_period)
    return _to_tuple_output(arrs, bars, series)


def nvi(
    bars: Any,
    *,
    initial: float = 1000.0,
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Union[float, "pl.Series"]:
    """Negative Volume Index (matches TA-Lib ``NVI``)."""
    from .volume import nvi_series
    c, v = _cols(bars, "nvi", col_close, col_volume)
    return _to_output(nvi_series(c, v, initial=initial), bars, series)


def pvi(
    bars: Any,
    *,
    initial: float = 1000.0,
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Union[float, "pl.Series"]:
    """Positive Volume Index (matches TA-Lib ``PVI``)."""
    from .volume import pvi_series
    c, v = _cols(bars, "pvi", col_close, col_volume)
    return _to_output(pvi_series(c, v, initial=initial), bars, series)


def pvt(
    bars: Any,
    *,
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Union[float, "pl.Series"]:
    """Price Volume Trend (matches TA-Lib ``PVT``)."""
    from .volume import pvt_series
    c, v = _cols(bars, "pvt", col_close, col_volume)
    return _to_output(pvt_series(c, v), bars, series)


def pvo(
    bars: Any,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    *,
    series: bool = False,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Tuple[Any, Any, Any]:
    """Percentage Volume Oscillator. Returns ``(pvo, signal, histogram)``.

    The PVO line matches TA-Lib ``PVO``.
    """
    from .volume import pvo_series
    vol = _column(bars, col_volume) if is_frame(bars) else ensure_array(bars)
    return _to_tuple_output(pvo_series(vol, fast_period, slow_period, signal_period), bars, series)


# ── Extended statistical ─────────────────────────────────────────────

def linear_reg_value(
    bars: Any,
    period: int = 14,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """End value of the rolling regression line (matches TA-Lib ``LINEARREG``)."""
    from .statistical import linear_reg_value_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(linear_reg_value_series(x, period), bars, series)


def linear_reg_angle(
    bars: Any,
    period: int = 14,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Regression slope angle in degrees (matches TA-Lib ``LINEARREG_ANGLE``)."""
    from .statistical import linear_reg_angle_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(linear_reg_angle_series(x, period), bars, series)


def efficiency_ratio(
    bars: Any,
    period: int = 10,
    *,
    source: str = "close",
    series: bool = False,
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> Union[float, "pl.Series"]:
    """Kaufman Efficiency Ratio (matches TA-Lib ``ER`` and pandas-ta ``er``)."""
    from .statistical import efficiency_ratio_series
    x = _price(bars, source, col_open, col_high, col_low, col_close)
    return _to_output(efficiency_ratio_series(x, period), bars, series)


def rolling_sharpe(
    returns: Any,
    period: int = 63,
    *,
    risk_free: float = 0.0,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    series: bool = False,
    col: Optional[str] = None,
) -> Union[float, "pl.Series"]:
    """Rolling Sharpe ratio of a periodic return series."""
    from .statistical import rolling_sharpe_series
    r = ensure_array(returns, col)
    arr = rolling_sharpe_series(r, period, risk_free=risk_free, annualize=annualize,
                                trading_days=trading_days)
    return _to_output(arr, returns, series)


def rolling_sortino(
    returns: Any,
    period: int = 63,
    *,
    target: float = 0.0,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    series: bool = False,
    col: Optional[str] = None,
) -> Union[float, "pl.Series"]:
    """Rolling Sortino ratio of a periodic return series."""
    from .statistical import rolling_sortino_series
    r = ensure_array(returns, col)
    arr = rolling_sortino_series(r, period, target=target, annualize=annualize,
                                 trading_days=trading_days)
    return _to_output(arr, returns, series)


# ═══════════════════════════════════════════════════════════════════
#  CANDLESTICK PATTERNS
# ═══════════════════════════════════════════════════════════════════

from .candles import CANDLE_PATTERNS as _CANDLE_PATTERNS  # noqa: E402
from .candles import TALIB_NAMES as _CANDLE_TALIB_NAMES  # noqa: E402

#: Names accepted by :func:`candle_pattern` and :func:`candle_patterns`.
CANDLE_PATTERN_NAMES: Tuple[str, ...] = tuple(sorted(_CANDLE_PATTERNS))


def _candle_kernel(name: str):
    try:
        return _CANDLE_PATTERNS[name]
    except KeyError:
        raise ValueError(
            f"Unknown candlestick pattern '{name}'. See CANDLE_PATTERN_NAMES."
        ) from None


def candle_pattern(
    bars: Any,
    name: str,
    *,
    penetration: Optional[float] = None,
    series: bool = False,
    col_open: str = DEFAULT_COL_OPEN,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Union[int, "pl.Series"]:
    """Recognise one candlestick pattern, bit-compatible with TA-Lib ``CDL*``.

    Args:
        bars: polars DataFrame with OHLC columns.
        name: Pattern name, e.g. ``"engulfing"`` or ``"morning_star"``; see
            ``CANDLE_PATTERN_NAMES``. The TA-Lib function each name mirrors is
            in ``cpz_quant.indicators.candles.TALIB_NAMES``.
        penetration: Override the TA-Lib penetration for the patterns that take
            one (abandoned baby, dark cloud cover, evening / morning star and
            doji star, mat hold). Passing it to any other pattern raises.
        series: Return the full ``pl.Series`` (Int32) instead of the latest bar.

    Returns:
        ``+100`` bullish, ``-100`` bearish, ``0`` none. Engulfing, harami and
        harami cross use ``+/-80`` when a body edge is equal, and hikkake uses
        ``+/-200`` on confirmation, exactly as TA-Lib does.
    """
    kernel = _candle_kernel(name)
    o, h, l, c = ensure_ohlc(_require_frame(bars, "candle_pattern"), col_open=col_open,
                             col_high=col_high, col_low=col_low, col_close=col_close)
    kwargs = {} if penetration is None else {"penetration": penetration}
    out = kernel(o, h, l, c, **kwargs)
    if series:
        return pl.Series(name=name, values=out, dtype=pl.Int32)
    return int(out[-1]) if len(out) else 0


def candle_patterns(
    bars: Any,
    names: Optional[Any] = None,
    *,
    col_open: str = DEFAULT_COL_OPEN,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> pl.DataFrame:
    """Run several candlestick patterns at once.

    Args:
        bars: polars DataFrame with OHLC columns.
        names: Iterable of pattern names; defaults to every pattern.

    Returns:
        A DataFrame with one Int32 column per pattern, aligned to *bars*.
    """
    selected = CANDLE_PATTERN_NAMES if names is None else tuple(names)
    kernels = {n: _candle_kernel(n) for n in selected}
    o, h, l, c = ensure_ohlc(_require_frame(bars, "candle_patterns"), col_open=col_open,
                             col_high=col_high, col_low=col_low, col_close=col_close)
    return pl.DataFrame(
        [pl.Series(name=n, values=fn(o, h, l, c), dtype=pl.Int32) for n, fn in kernels.items()]
    )


# ═══════════════════════════════════════════════════════════════════
#  MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════

__all__ = [
    # Trend
    "sma", "ema", "wma", "dema", "tema", "kama", "supertrend",
    # Momentum
    "rsi", "macd", "stochastic", "williams_r", "cci", "roc", "momentum", "adx", "mfi",
    # Volatility
    "atr", "bollinger", "keltner", "donchian", "realized_vol", "garman_klass", "natr",
    # Volume
    "vwap", "obv", "cmf", "volume_zscore",
    # Statistical
    "zscore", "rolling_corr", "rolling_beta", "hurst", "linear_reg",
    # Extended trend
    "trima", "t3", "zlema", "hma", "alma", "vidya", "mcginley", "psar", "ichimoku",
    "aroon", "aroon_osc", "vortex", "trix", "mass_index", "midpoint", "midprice", "stc",
    # Extended momentum
    "stochastic_slow", "stoch_rsi", "cmo", "mom", "rocp", "rocr", "ultimate_oscillator",
    "awesome_oscillator", "apo", "ppo", "kst", "tsi", "connors_rsi", "fisher_transform",
    "coppock", "dpo", "rvgi", "elder_ray", "wavetrend", "bop", "dmi",
    # Extended volatility
    "chaikin_volatility", "ulcer_index", "choppiness", "parkinson", "rogers_satchell",
    "yang_zhang",
    # Extended volume
    "vwma", "ad_line", "adosc", "force_index", "eom", "klinger", "nvi", "pvi", "pvt", "pvo",
    # Extended statistical
    "linear_reg_value", "linear_reg_angle", "efficiency_ratio", "rolling_sharpe",
    "rolling_sortino",
    # Candlestick patterns
    "candle_pattern", "candle_patterns", "CANDLE_PATTERN_NAMES",
    # Result types
    "MACDResult", "BollingerResult", "LinRegResult", "IchimokuResult", "DMIResult",
    # Constants (defaults for reference, not global settings)
    "EPSILON", "TRADING_DAYS_PER_YEAR", "ANNUALIZATION_FACTOR",
    # Rust status
    "has_rust",
]


# ═══════════════════════════════════════════════════════════════════
#  ``momentum`` IS BOTH AN INDICATOR AND A SUBMODULE
#
#  Importing a package's submodule binds it as an attribute of the
#  package, so the first lazy ``from .momentum import ...`` (rsi, macd and
#  most momentum wrappers do one) used to replace the ``momentum()``
#  function with the module, and ``momentum(...)`` then raised
#  "'module' object is not callable". The submodule is loaded here and made
#  callable as the indicator, so ``momentum(bars, 10)`` and
#  ``import cpz_quant.indicators.momentum as mo`` both work in any order.
# ═══════════════════════════════════════════════════════════════════

_momentum_indicator = momentum
if not isinstance(_momentum_indicator, types.FunctionType):  # pragma: no cover
    raise ImportError("cpz_quant.indicators: momentum() was shadowed before it could be wrapped")


class _CallableIndicatorModule(types.ModuleType):
    """The ``momentum`` submodule, also callable as the ``momentum()`` indicator."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _momentum_indicator(*args, **kwargs)


_momentum_module = importlib.import_module(".momentum", __name__)
_momentum_module.__class__ = _CallableIndicatorModule
_momentum_module.__signature__ = inspect.signature(_momentum_indicator)  # type: ignore[attr-defined]
globals()["momentum"] = _momentum_module
