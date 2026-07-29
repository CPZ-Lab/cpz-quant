"""Trend indicators: SMA, EMA, WMA, DEMA, TEMA, KAMA, Supertrend.

All functions operate on raw numpy arrays and return numpy arrays.
The public ``__init__`` API handles DataFrame extraction, Rust dispatch,
and scalar / Series conversion.
"""

from __future__ import annotations

import numpy as np

from ._helpers import EPSILON

# ── SMA ──────────────────────────────────────────────────────────────

def sma_series(close: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average via cumulative-sum trick (O(n))."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period or period < 1:
        return out
    cs = np.cumsum(close)
    out[period - 1] = cs[period - 1] / period
    if period < n:
        out[period:] = (cs[period:] - cs[:-period]) / period
    return out


# ── EMA ──────────────────────────────────────────────────────────────

def ema_series(
    close: np.ndarray,
    period: int,
    *,
    alpha: float | None = None,
) -> np.ndarray:
    """Exponential Moving Average.

    Args:
        close: Price array.
        period: EMA look-back window (used to derive *alpha* when not given).
        alpha: Explicit smoothing factor. Overrides the standard
               ``2 / (period + 1)`` formula when provided, allowing full
               user control over decay speed.
    """
    if alpha is None:
        alpha = 2.0 / (period + 1)
    n = len(close)
    out = np.empty(n)
    out[0] = close[0]
    one_minus_alpha = 1.0 - alpha
    for i in range(1, n):
        out[i] = alpha * close[i] + one_minus_alpha * out[i - 1]
    return out


# ── WMA ──────────────────────────────────────────────────────────────

def wma_series(close: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average (linearly increasing weights)."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period or period < 1:
        return out
    weights = np.arange(1, period + 1, dtype=np.float64)
    denom = weights.sum()
    for i in range(period - 1, n):
        out[i] = np.dot(close[i - period + 1 : i + 1], weights) / denom
    return out


# ── DEMA ─────────────────────────────────────────────────────────────

def dema_series(
    close: np.ndarray,
    period: int,
    *,
    alpha: float | None = None,
) -> np.ndarray:
    """Double Exponential Moving Average: ``2 * EMA - EMA(EMA)``."""
    e1 = ema_series(close, period, alpha=alpha)
    e2 = ema_series(e1, period, alpha=alpha)
    return 2.0 * e1 - e2


# ── TEMA ─────────────────────────────────────────────────────────────

def tema_series(
    close: np.ndarray,
    period: int,
    *,
    alpha: float | None = None,
) -> np.ndarray:
    """Triple Exponential Moving Average: ``3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))``."""
    e1 = ema_series(close, period, alpha=alpha)
    e2 = ema_series(e1, period, alpha=alpha)
    e3 = ema_series(e2, period, alpha=alpha)
    return 3.0 * e1 - 3.0 * e2 + e3


# ── KAMA ─────────────────────────────────────────────────────────────

def kama_series(
    close: np.ndarray,
    period: int = 10,
    *,
    fast_period: int = 2,
    slow_period: int = 30,
) -> np.ndarray:
    """Kaufman Adaptive Moving Average.

    Args:
        close: Price array.
        period: Efficiency ratio look-back.
        fast_period: Fast smoothing constant period.
        slow_period: Slow smoothing constant period.
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n <= period:
        return out

    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)

    out[period - 1] = close[period - 1]
    for i in range(period, n):
        direction = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period : i + 1])))
        er = direction / volatility if volatility > EPSILON else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out[i] = out[i - 1] + sc * (close[i] - out[i - 1])
    return out


# ── Supertrend ───────────────────────────────────────────────────────

def supertrend_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 10,
    *,
    multiplier: float = 3.0,
) -> np.ndarray:
    """Supertrend indicator.

    Args:
        high, low, close: OHLC arrays.
        period: ATR period for band calculation.
        multiplier: ATR multiplier for upper / lower bands.

    Returns:
        Supertrend line values (same length as input).
    """
    from .volatility import atr_series

    n = len(close)
    atr_vals = atr_series(high, low, close, period)
    hl2 = (high + low) / 2.0

    upper_band = hl2 + multiplier * atr_vals
    lower_band = hl2 - multiplier * atr_vals

    supertrend = np.full(n, np.nan)
    trend_dir = np.ones(n)

    for i in range(period, n):
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            continue
        if i == period:
            supertrend[i] = upper_band[i]
            trend_dir[i] = -1.0
            continue

        prev = supertrend[i - 1] if not np.isnan(supertrend[i - 1]) else upper_band[i]
        trend_dir[i] = 1.0 if close[i - 1] > prev else -1.0

        if trend_dir[i] == 1.0:
            if not np.isnan(lower_band[i - 1]):
                lower_band[i] = max(lower_band[i], lower_band[i - 1])
            supertrend[i] = lower_band[i]
        else:
            if not np.isnan(upper_band[i - 1]):
                upper_band[i] = min(upper_band[i], upper_band[i - 1])
            supertrend[i] = upper_band[i]

    return supertrend
