"""Volume indicators: VWAP, OBV, CMF, volume z-score.

All functions operate on raw numpy arrays and return numpy arrays.
"""

from __future__ import annotations

import numpy as np

from ._helpers import EPSILON

# ── VWAP ─────────────────────────────────────────────────────────────

def vwap_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    *,
    anchor_session: bool = True,
) -> np.ndarray:
    """Volume-Weighted Average Price.

    Args:
        high, low, close: Price arrays.
        volume: Volume array.
        anchor_session: If ``True`` (default), computes a cumulative VWAP
            across the full array.  Set ``False`` for a rolling variant
            (not standard but sometimes useful).
    """
    tp = (high + low + close) / 3.0
    cum_tp_vol = np.cumsum(tp * volume)
    cum_vol = np.cumsum(volume)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cum_vol > EPSILON, cum_tp_vol / cum_vol, np.nan)


# ── OBV ──────────────────────────────────────────────────────────────

def obv_series(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume."""
    n = len(close)
    out = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - volume[i]
        else:
            out[i] = out[i - 1]
    return out


# ── CMF ──────────────────────────────────────────────────────────────

def cmf_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Chaikin Money Flow.

    CMF = sum(MFV, period) / sum(volume, period)
    where MFV = ((close - low) - (high - close)) / (high - low) * volume
    """
    n = len(close)
    hl_range = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        mf_multiplier = np.where(
            hl_range > EPSILON,
            ((close - low) - (high - close)) / hl_range,
            0.0,
        )
    mfv = mf_multiplier * volume

    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_sum = np.sum(volume[i - period + 1 : i + 1])
        if vol_sum > EPSILON:
            out[i] = np.sum(mfv[i - period + 1 : i + 1]) / vol_sum
        else:
            out[i] = 0.0
    return out


# ── Volume Z-Score ───────────────────────────────────────────────────

def volume_zscore_series(
    volume: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Volume z-score relative to its rolling mean and standard deviation.

    Args:
        volume: Volume array.
        period: Rolling look-back window for mean and std.
    """
    n = len(volume)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = volume[i - period + 1 : i + 1]
        mean = np.mean(window)
        std = np.std(window, ddof=1)
        out[i] = (volume[i] - mean) / std if std > EPSILON else 0.0
    return out
