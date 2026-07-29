"""Volatility indicators: ATR, Bollinger, Keltner, Donchian, realized vol, Garman-Klass, NATR.

All functions operate on raw numpy arrays and return numpy arrays.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._helpers import EPSILON, TRADING_DAYS_PER_YEAR, wilder_smooth
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
