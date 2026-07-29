"""CPZ Indicators — local technical indicator computation with optional Rust acceleration.

Usage::

    from cpz_quant.indicators import sma, rsi, atr, bollinger, macd

    bars = data["NVDA"]  # DataFrame with OHLCV columns

    # Scalar (latest value) — for signal logic
    current_rsi = rsi(bars, period=14)

    # Full series — for crossover detection, plotting
    rsi_history = rsi(bars, period=14, series=True)

Every function accepts a polars DataFrame (with configurable column names),
a polars Series, a numpy array, or a plain list. Parameters such as period,
smoothing method, and source price are always configurable — no hard-coded
magic numbers.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import polars as pl

# Re-export constants so users can override globally
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
    source_price,
)
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
    """
    if series:
        return pl.Series(values=arr.astype(np.float64), dtype=pl.Float64)
    if len(arr) == 0:
        return 0.0
    val = arr[~np.isnan(arr)]
    return float(val[-1]) if len(val) > 0 else float("nan")


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
    rust_result = fast_sma(close.tolist(), period)
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
    if alpha is None:
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
    rust_result = fast_wma(close.tolist(), period)
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

    Args:
        bars: Price data.
        period: RSI period.
        smoothing: ``"wilder"`` (default) or ``"ema"``.
        source: Price source.
    """
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    if smoothing == "wilder":
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

    Returns:
        ``MACDResult(macd, signal, histogram)`` — each is a scalar or
        ``pl.Series`` depending on *series*.
    """
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    rust_result = fast_macd(close.tolist(), fast_period, slow_period, signal_period)
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
    rust_result = fast_adx(high.tolist(), low.tolist(), close.tolist(), period)
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
    if not isinstance(bars, pl.DataFrame):
        raise TypeError("mfi() requires a polars DataFrame with OHLCV columns")
    high = bars[col_high].to_numpy().astype(np.float64)
    low = bars[col_low].to_numpy().astype(np.float64)
    close = bars[col_close].to_numpy().astype(np.float64)
    vol = bars[col_volume].to_numpy().astype(np.float64)
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
    if smoothing == "wilder":
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
    if ma_type == "sma":
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
    """Donchian Channels. Returns ``(upper, lower)``."""
    if isinstance(bars, pl.DataFrame):
        high = bars[col_high].to_numpy().astype(np.float64)
        low = bars[col_low].to_numpy().astype(np.float64)
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
    rust_result = fast_garman_klass(o.tolist(), h.tolist(), l.tolist(), c.tolist(), period)
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
    series: bool = False,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Union[float, "pl.Series"]:
    """Volume-Weighted Average Price."""
    h, l, c, v = ensure_ohlcv(bars, col_high=col_high, col_low=col_low,
                               col_close=col_close, col_volume=col_volume)[1:]
    from .volume import vwap_series
    return _to_output(vwap_series(h, l, c, v), bars, series)


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
    """On-Balance Volume."""
    close = source_price(bars, source, col_open=col_open, col_high=col_high,
                         col_low=col_low, col_close=col_close) if source != "close" else ensure_array(bars, col_close)
    vol = bars[col_volume].to_numpy().astype(np.float64) if isinstance(bars, pl.DataFrame) else np.zeros_like(close)
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
    if not isinstance(bars, pl.DataFrame):
        raise TypeError("cmf() requires a polars DataFrame with OHLCV columns")
    h = bars[col_high].to_numpy().astype(np.float64)
    l = bars[col_low].to_numpy().astype(np.float64)
    c = bars[col_close].to_numpy().astype(np.float64)
    v = bars[col_volume].to_numpy().astype(np.float64)
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
    vol = bars[col_volume].to_numpy().astype(np.float64) if isinstance(bars, pl.DataFrame) else ensure_array(bars)
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
    min_lag: int = 2,
    source: str = "close",
    col_close: str = DEFAULT_COL_CLOSE,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_open: str = DEFAULT_COL_OPEN,
) -> float:
    """Hurst exponent (H<0.5 mean-reverting, H=0.5 random, H>0.5 trending)."""
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
    # Result types
    "MACDResult", "BollingerResult", "LinRegResult",
    # Constants (overridable)
    "EPSILON", "TRADING_DAYS_PER_YEAR", "ANNUALIZATION_FACTOR",
    # Rust status
    "has_rust",
]
