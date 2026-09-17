"""Volatility indicators: ATR, Bollinger, Keltner, Donchian, realized vol, Garman-Klass, NATR,
Chaikin Volatility, Ulcer Index, Choppiness Index, Parkinson, Rogers-Satchell, Yang-Zhang.

All functions operate on raw numpy arrays and return numpy arrays.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._helpers import (
    EPSILON,
    TRADING_DAYS_PER_YEAR,
    _rolling_view,
    ema_sma_seed,
    rolling_apply,
    safe_divide,
    shift,
    wilder_smooth,
)
from .trend import ema_series, sma_series

# ── ATR ──────────────────────────────────────────────────────────────

def true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """True Range: max(H-L, |H-prev_C|, |L-prev_C|)."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return tr


def atr_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
    *,
    smoothing: str = "wilder",
) -> np.ndarray:
    """Average True Range.

    Args:
        high, low, close: Price arrays.
        period: Smoothing window.
        smoothing: ``"wilder"`` (Wilder's smoothing, default),
                   ``"ema"`` (standard EMA), or ``"sma"`` (simple average).
    """
    tr = true_range(high, low, close)
    if smoothing == "wilder":
        return wilder_smooth(tr, period)
    if smoothing == "ema":
        return ema_series(tr, period)
    if smoothing == "sma":
        return sma_series(tr, period)
    raise ValueError(f"Unknown smoothing '{smoothing}'. Use: wilder, ema, sma")


# ── Bollinger Bands ──────────────────────────────────────────────────

def bollinger_series(
    close: np.ndarray,
    period: int = 20,
    *,
    num_std: float = 2.0,
    ma_type: str = "sma",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands.

    Args:
        close: Price array.
        period: Moving average period.
        num_std: Number of standard deviations for bands.
        ma_type: ``"sma"`` (default) or ``"ema"`` for the centre line.

    Returns:
        ``(upper, middle, lower)`` arrays.
    """
    if ma_type == "sma":
        mid = sma_series(close, period)
    elif ma_type == "ema":
        mid = ema_series(close, period)
    else:
        raise ValueError(f"Unknown ma_type '{ma_type}'. Use: sma, ema")

    n = len(close)
    std = np.full(n, np.nan)
    for i in range(period - 1, n):
        std[i] = np.std(close[i - period + 1 : i + 1], ddof=0)

    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


# ── Keltner Channels ─────────────────────────────────────────────────

def keltner_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20,
    *,
    atr_period: int = 10,
    multiplier: float = 1.5,
    atr_smoothing: str = "wilder",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keltner Channels: EMA +/- multiplier * ATR.

    Args:
        high, low, close: Price arrays.
        period: EMA period for centre line.
        atr_period: ATR calculation period.
        multiplier: ATR multiplier for channel width.
        atr_smoothing: Smoothing for ATR (``"wilder"``, ``"ema"``, ``"sma"``).

    Returns:
        ``(upper, middle, lower)`` arrays.
    """
    mid = ema_series(close, period)
    atr_vals = atr_series(high, low, close, atr_period, smoothing=atr_smoothing)
    upper = mid + multiplier * atr_vals
    lower = mid - multiplier * atr_vals
    return upper, mid, lower


# ── Donchian Channels ────────────────────────────────────────────────

def donchian_series(
    high: np.ndarray,
    low: np.ndarray,
    period: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Donchian Channels: rolling highest high / lowest low.

    Returns:
        ``(upper, lower)`` arrays.
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1 : i + 1])
        lower[i] = np.min(low[i - period + 1 : i + 1])
    return upper, lower


# ── Realized Volatility ─────────────────────────────────────────────

def realized_vol_series(
    close: np.ndarray,
    period: int = 20,
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Historical / realised volatility (std of log returns).

    Args:
        close: Price array.
        period: Rolling window.
        annualize: Multiply by ``sqrt(trading_days)`` (default ``True``).
        trading_days: Number of trading days per year (default 252).
    """
    n = len(close)
    out = np.full(n, np.nan)
    log_ret = np.diff(np.log(np.maximum(close, EPSILON)))

    for i in range(period, n):
        window = log_ret[i - period : i]
        out[i] = np.std(window, ddof=1)

    if annualize:
        out *= np.sqrt(trading_days)
    return out


# ── Garman-Klass Volatility ─────────────────────────────────────────

def garman_klass_series(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20,
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Garman-Klass volatility estimator (uses OHLC, more efficient than close-to-close).

    GK = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2

    Args:
        open_, high, low, close: OHLC arrays.
        period: Rolling window.
        annualize: Multiply by ``sqrt(trading_days)`` (default ``True``).
        trading_days: Trading days per year (default 252).
    """
    ln2_const = 2.0 * np.log(2.0) - 1.0
    hl = np.log(np.maximum(high / np.maximum(low, EPSILON), EPSILON))
    co = np.log(np.maximum(close / np.maximum(open_, EPSILON), EPSILON))
    gk_daily = 0.5 * hl ** 2 - ln2_const * co ** 2

    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = np.sqrt(np.mean(gk_daily[i - period + 1 : i + 1]))

    if annualize:
        out *= np.sqrt(trading_days)
    return out


# ── Normalised ATR ───────────────────────────────────────────────────

def natr_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
    *,
    smoothing: str = "wilder",
) -> np.ndarray:
    """Normalised ATR: ``ATR / close * 100`` (percentage of price).

    Args:
        high, low, close: Price arrays.
        period: ATR period.
        smoothing: ATR smoothing method (``"wilder"``, ``"ema"``, ``"sma"``).
    """
    atr_vals = atr_series(high, low, close, period, smoothing=smoothing)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(close) > EPSILON, atr_vals / close * 100.0, np.nan)


# ═══════════════════════════════════════════════════════════════════
#  Extended volatility family
# ═══════════════════════════════════════════════════════════════════


def _annualize(values: np.ndarray, annualize: bool, trading_days: int) -> np.ndarray:
    return values * np.sqrt(trading_days) if annualize else values


# ── Chaikin Volatility ───────────────────────────────────────────────

def chaikin_volatility_series(
    high: np.ndarray,
    low: np.ndarray,
    period: int = 10,
    roc_period: int = 10,
) -> np.ndarray:
    """Chaikin Volatility: percent change over *roc_period* of ``EMA(high - low, period)``.

    The EMA is SMA-seeded.
    """
    e = ema_sma_seed(high - low, period)
    prev = shift(e, roc_period)
    return safe_divide(e - prev, prev) * 100.0


# ── Ulcer Index ──────────────────────────────────────────────────────

def ulcer_index_series(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Peter Martin's Ulcer Index (StockCharts definition).

    ``pct_dd = 100 * (close - max(close, period)) / max(close, period)`` and
    ``UI = sqrt(mean(pct_dd^2, period))``. First value at ``2 * period - 2``.
    """
    peak = rolling_apply(close, period, "max")
    pct_dd = safe_divide(close - peak, peak) * 100.0
    return np.sqrt(rolling_apply(pct_dd**2, period, "mean"))


# ── Choppiness Index ─────────────────────────────────────────────────

def choppiness_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Dreiss' Choppiness Index (0..100).

    ``100 * log10(sum(TR, period) / (max(high, period) - min(low, period))) / log10(period)``
    where TR needs a previous close, so the first value is at index *period*
    (TA-Lib ``TRANGE`` and pandas-ta convention).
    """
    tr = true_range(high, low, close)
    tr[:1] = np.nan
    ratio = safe_divide(
        rolling_apply(tr, period, "sum"),
        rolling_apply(high, period, "max") - rolling_apply(low, period, "min"),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * np.log10(ratio) / np.log10(period)


# ── Range-based volatility estimators ────────────────────────────────

def parkinson_series(
    high: np.ndarray,
    low: np.ndarray,
    period: int = 20,
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Parkinson (1980) high-low volatility.

    ``sigma^2 = mean(ln(high / low)^2) / (4 ln 2)`` over *period* bars.
    """
    hl = np.log(high / low)
    var = rolling_apply(hl**2, period, "mean") / (4.0 * np.log(2.0))
    return _annualize(np.sqrt(var), annualize, trading_days)


def rogers_satchell_series(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20,
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Rogers-Satchell (1991) drift-independent volatility.

    ``sigma^2 = mean(ln(H/C) ln(H/O) + ln(L/C) ln(L/O))`` over *period* bars.
    """
    rs = np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(low / open_)
    var = rolling_apply(rs, period, "mean")
    return _annualize(np.sqrt(np.maximum(var, 0.0)), annualize, trading_days)


def yang_zhang_series(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20,
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Yang-Zhang (2000) volatility.

    ``sigma^2 = var(overnight) + k * var(open_to_close) + (1 - k) * RS`` with
    ``overnight = ln(O_t / C_{t-1})``, ``open_to_close = ln(C_t / O_t)``,
    sample variances (ddof=1) and the Rogers-Satchell mean ``RS`` over
    *period* bars, and ``k = 0.34 / (1.34 + (period + 1) / (period - 1))``.
    The first value needs a previous close, so it appears at index *period*.
    """
    if period < 2:
        raise ValueError("yang_zhang period must be >= 2")
    n = len(close)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    overnight = np.log(open_[1:] / close[:-1])
    oc = np.log(close[1:] / open_[1:])
    rs = (
        np.log(high[1:] / close[1:]) * np.log(high[1:] / open_[1:])
        + np.log(low[1:] / close[1:]) * np.log(low[1:] / open_[1:])
    )
    var_o = _rolling_view(overnight, period).var(axis=1, ddof=1)
    var_c = _rolling_view(oc, period).var(axis=1, ddof=1)
    var_rs = _rolling_view(rs, period).mean(axis=1)
    k = 0.34 / (1.34 + (period + 1.0) / (period - 1.0))
    out[period:] = np.sqrt(np.maximum(var_o + k * var_c + (1.0 - k) * var_rs, 0.0))
    return _annualize(out, annualize, trading_days)
