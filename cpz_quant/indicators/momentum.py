"""Momentum indicators: RSI, MACD, Stochastic, Williams %R, CCI, ROC, ADX, MFI, momentum.

All functions operate on raw numpy arrays and return numpy arrays.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._helpers import EPSILON
from .trend import ema_series, sma_series

# ── RSI ──────────────────────────────────────────────────────────────

def rsi_series(
    close: np.ndarray,
    period: int = 14,
    *,
    smoothing: str = "wilder",
) -> np.ndarray:
    """Relative Strength Index.

    Args:
        close: Price array.
        period: Look-back period.
        smoothing: ``"wilder"`` (Wilder's smoothing, default) or ``"ema"``
                   (standard exponential smoothing).
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    if smoothing == "wilder":
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        out[period] = 100.0 - 100.0 / (1.0 + avg_gain / max(avg_loss, EPSILON))
        alpha = 1.0 / period
    elif smoothing == "ema":
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        out[period] = 100.0 - 100.0 / (1.0 + avg_gain / max(avg_loss, EPSILON))
        alpha = 2.0 / (period + 1)
    else:
        raise ValueError(f"Unknown smoothing '{smoothing}'. Use: wilder, ema")

    for i in range(period, len(deltas)):
        avg_gain = alpha * gains[i] + (1.0 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1.0 - alpha) * avg_loss
        rs = avg_gain / max(avg_loss, EPSILON)
        out[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return out


# ── MACD ─────────────────────────────────────────────────────────────

def macd_series(
    close: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving Average Convergence Divergence.

    Args:
        close: Price array.
        fast_period: Fast EMA period.
        slow_period: Slow EMA period.
        signal_period: Signal line EMA period.

    Returns:
        ``(macd_line, signal_line, histogram)`` arrays.
    """
    fast_ema = ema_series(close, fast_period)
    slow_ema = ema_series(close, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = ema_series(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ── Stochastic ───────────────────────────────────────────────────────

def stochastic_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
    *,
    d_ma_type: str = "sma",
) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic Oscillator (%K and %D).

    Args:
        high, low, close: Price arrays.
        k_period: %K look-back.
        d_period: %D smoothing period.
        d_ma_type: Smoothing type for %D — ``"sma"`` (default) or ``"ema"``.

    Returns:
        ``(percent_k, percent_d)`` arrays.
    """
    n = len(close)
    k = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        highest = np.max(high[i - k_period + 1 : i + 1])
        lowest = np.min(low[i - k_period + 1 : i + 1])
        rng = highest - lowest
        k[i] = ((close[i] - lowest) / rng * 100.0) if rng > EPSILON else 50.0

    k_filled = np.nan_to_num(k, nan=50.0)
    if d_ma_type == "sma":
        d = sma_series(k_filled, d_period)
    elif d_ma_type == "ema":
        d = ema_series(k_filled, d_period)
    else:
        raise ValueError(f"Unknown d_ma_type '{d_ma_type}'. Use: sma, ema")
    d[: k_period - 1] = np.nan
    return k, d


# ── Williams %R ──────────────────────────────────────────────────────

def williams_r_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Williams %R oscillator (range: -100 to 0)."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1 : i + 1])
        lowest = np.min(low[i - period + 1 : i + 1])
        rng = highest - lowest
        out[i] = ((highest - close[i]) / rng * -100.0) if rng > EPSILON else -50.0
    return out


# ── CCI ──────────────────────────────────────────────────────────────

def cci_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20,
    *,
    constant: float = 0.015,
) -> np.ndarray:
    """Commodity Channel Index.

    Args:
        high, low, close: Price arrays.
        period: Look-back window.
        constant: Lambert's constant (default 0.015).
    """
    tp = (high + low + close) / 3.0
    n = len(tp)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = tp[i - period + 1 : i + 1]
        mean_tp = np.mean(window)
        mean_dev = np.mean(np.abs(window - mean_tp))
        denom = constant * mean_dev
        out[i] = (tp[i] - mean_tp) / denom if denom > EPSILON else 0.0
    return out


# ── ROC ──────────────────────────────────────────────────────────────

def roc_series(close: np.ndarray, period: int = 12) -> np.ndarray:
    """Rate of Change (percentage)."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period, n):
        prev = close[i - period]
        out[i] = ((close[i] - prev) / prev * 100.0) if abs(prev) > EPSILON else 0.0
    return out


# ── Momentum ─────────────────────────────────────────────────────────

def momentum_pct_series(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Simple price momentum as percentage return over *period* bars."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period, n):
        prev = close[i - period]
        out[i] = ((close[i] / prev - 1.0) * 100.0) if abs(prev) > EPSILON else 0.0
    return out


# ── ADX ──────────────────────────────────────────────────────────────

def adx_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average Directional Index."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        h_diff = high[i] - high[i - 1]
        l_diff = low[i - 1] - low[i]
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        plus_dm[i] = h_diff if (h_diff > l_diff and h_diff > 0) else 0.0
        minus_dm[i] = l_diff if (l_diff > h_diff and l_diff > 0) else 0.0

    smoothed_tr = np.mean(tr[1 : period + 1])
    smoothed_plus = np.mean(plus_dm[1 : period + 1])
    smoothed_minus = np.mean(minus_dm[1 : period + 1])

    dx_values: list[float] = []

    for i in range(period, n):
        if i > period:
            smoothed_tr = (smoothed_tr * (period - 1) + tr[i]) / period
            smoothed_plus = (smoothed_plus * (period - 1) + plus_dm[i]) / period
            smoothed_minus = (smoothed_minus * (period - 1) + minus_dm[i]) / period

        plus_di = (smoothed_plus / smoothed_tr * 100.0) if smoothed_tr > EPSILON else 0.0
        minus_di = (smoothed_minus / smoothed_tr * 100.0) if smoothed_tr > EPSILON else 0.0
        di_sum = plus_di + minus_di
        dx = (abs(plus_di - minus_di) / di_sum * 100.0) if di_sum > EPSILON else 0.0
        dx_values.append(dx)

        if len(dx_values) == period:
            out[i] = np.mean(dx_values)
        elif len(dx_values) > period:
            out[i] = (out[i - 1] * (period - 1) + dx) / period

    return out


# ── MFI ──────────────────────────────────────────────────────────────

def mfi_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Money Flow Index (volume-weighted RSI)."""
    tp = (high + low + close) / 3.0
    raw_mf = tp * volume
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    for i in range(period, n):
        pos_flow = 0.0
        neg_flow = 0.0
        for j in range(i - period + 1, i + 1):
            if tp[j] > tp[j - 1]:
                pos_flow += raw_mf[j]
            elif tp[j] < tp[j - 1]:
                neg_flow += raw_mf[j]
        mfr = pos_flow / max(neg_flow, EPSILON)
        out[i] = 100.0 - 100.0 / (1.0 + mfr)

    return out
