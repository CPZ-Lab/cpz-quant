"""Volume indicators: VWAP, OBV, CMF, volume z-score, anchored VWAP, VWMA,
Accumulation/Distribution, Chaikin A/D Oscillator, Force Index, Ease of Movement,
Klinger Volume Oscillator, NVI, PVI, Price Volume Trend, Percentage Volume Oscillator.

All functions operate on raw numpy arrays and return numpy arrays.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._helpers import EPSILON, ema_sma_seed, rolling_apply, safe_divide, shift

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


# ═══════════════════════════════════════════════════════════════════
#  Extended volume family
# ═══════════════════════════════════════════════════════════════════


# ── Anchored (session) VWAP ─────────────────────────────────────────

def anchored_vwap_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    sessions: np.ndarray,
) -> np.ndarray:
    """Session-anchored VWAP on the typical price ``(H + L + C) / 3``.

    Cumulative price-volume and volume restart whenever *sessions* changes
    value between consecutive bars (for example a trading date, or a week
    label). Bars with zero cumulative volume in their session are NaN.
    """
    sessions = np.asarray(sessions)
    n = len(close)
    if len(sessions) != n:
        raise ValueError("sessions must have the same length as the price arrays")
    tp_vol = (high + low + close) / 3.0 * volume
    out = np.full(n, np.nan)
    if n == 0:
        return out
    starts = np.flatnonzero(np.r_[True, sessions[1:] != sessions[:-1]])
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        cum_pv = np.cumsum(tp_vol[s:e])
        cum_v = np.cumsum(volume[s:e])
        out[s:e] = safe_divide(cum_pv, cum_v)
    return out


# ── VWMA ─────────────────────────────────────────────────────────────

def vwma_series(close: np.ndarray, volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Volume-Weighted Moving Average: ``sum(close * volume) / sum(volume)``."""
    return safe_divide(
        rolling_apply(close * volume, period, "sum"),
        rolling_apply(volume, period, "sum"),
    )


# ── Accumulation / Distribution ──────────────────────────────────────

def ad_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Chaikin Accumulation/Distribution Line (TA-Lib ``AD``).

    Cumulative ``((C - L) - (H - C)) / (H - L) * V``; bars with a zero range
    add nothing.
    """
    clv = safe_divide((close - low) - (high - close), high - low, fill=0.0)
    return np.cumsum(clv * volume)


def adosc_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    fast_period: int = 3,
    slow_period: int = 10,
) -> np.ndarray:
    """Chaikin A/D Oscillator (TA-Lib ``ADOSC``): ``EMA(AD, fast) - EMA(AD, slow)``.

    Following TA-Lib, both EMAs are seeded with the first A/D value (not an
    SMA) and the output starts at index ``slow_period - 1``.
    """
    ad = ad_series(high, low, close, volume)
    n = len(ad)
    out = np.full(n, np.nan)
    if n == 0:
        return out

    def ema_first(p: int) -> np.ndarray:
        a = 2.0 / (p + 1)
        e = np.empty(n)
        e[0] = ad[0]
        for i in range(1, n):
            e[i] = a * ad[i] + (1.0 - a) * e[i - 1]
        return e

    diff = ema_first(fast_period) - ema_first(slow_period)
    lookback = max(fast_period, slow_period) - 1
    out[lookback:] = diff[lookback:]
    return out


# ── Force Index ──────────────────────────────────────────────────────

def force_index_series(close: np.ndarray, volume: np.ndarray, period: int = 13) -> np.ndarray:
    """Elder's Force Index: ``EMA((close - close[t-1]) * volume, period)``, SMA-seeded."""
    raw = (close - shift(close, 1)) * volume
    return ema_sma_seed(raw, period)


# ── Ease of Movement ─────────────────────────────────────────────────

def eom_series(
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    period: int = 14,
    *,
    divisor: float = 100_000_000.0,
) -> np.ndarray:
    """Arms' Ease of Movement (StockCharts definition).

    ``distance = hl2 - hl2[t-1]``, ``box_ratio = (volume / divisor) / (high - low)``,
    ``EMV = distance / box_ratio`` and ``EOM = SMA(EMV, period)``. Computed
    as ``distance * (high - low) * divisor / volume`` so a zero range gives 0;
    zero volume gives NaN.
    """
    hl2 = (high + low) / 2.0
    distance = hl2 - shift(hl2, 1)
    emv = safe_divide(distance * (high - low) * divisor, volume)
    n = len(emv)
    out = np.full(n, np.nan)
    if n > 1:
        from .trend import sma_series

        out[1:] = sma_series(emv[1:], period)
    return out


# ── Klinger Volume Oscillator ────────────────────────────────────────

def klinger_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    fast_period: int = 34,
    slow_period: int = 55,
    signal_period: int = 13,
) -> Tuple[np.ndarray, np.ndarray]:
    """Klinger Volume Oscillator, simplified form.

    ``sv = +volume`` when ``hlc3`` rises versus the previous bar, ``-volume``
    when it falls (0 when unchanged); ``KVO = EMA(sv, 34) - EMA(sv, 55)`` and
    the signal is ``EMA(KVO, 13)``, all SMA-seeded. This is the form used by
    TradingView's built-in and pandas-ta. Klinger's original "volume force"
    additionally scales volume by a cumulative-range term that published
    sources describe inconsistently, so it is not implemented.

    Returns:
        ``(kvo, signal)``.
    """
    hlc3 = (high + low + close) / 3.0
    sv = np.sign(hlc3 - shift(hlc3, 1)) * volume
    kvo = ema_sma_seed(sv, fast_period) - ema_sma_seed(sv, slow_period)
    return kvo, ema_sma_seed(kvo, signal_period)


# ── NVI / PVI ────────────────────────────────────────────────────────

def _volume_index(
    close: np.ndarray,
    volume: np.ndarray,
    initial: float,
    positive: bool,
) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    out[0] = initial
    for i in range(1, n):
        moved = volume[i] > volume[i - 1] if positive else volume[i] < volume[i - 1]
        if moved and abs(close[i - 1]) > EPSILON:
            out[i] = out[i - 1] * close[i] / close[i - 1]
        else:
            out[i] = out[i - 1]
    return out


def nvi_series(close: np.ndarray, volume: np.ndarray, *, initial: float = 1000.0) -> np.ndarray:
    """Negative Volume Index: compounds the close-to-close return only on
    bars where volume falls, starting from *initial*."""
    return _volume_index(close, volume, initial, positive=False)


def pvi_series(close: np.ndarray, volume: np.ndarray, *, initial: float = 1000.0) -> np.ndarray:
    """Positive Volume Index: compounds the close-to-close return only on
    bars where volume rises, starting from *initial*."""
    return _volume_index(close, volume, initial, positive=True)


# ── Price Volume Trend ───────────────────────────────────────────────

def pvt_series(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Price Volume Trend: cumulative ``volume * (close - prev) / prev``, starting at 0."""
    n = len(close)
    out = np.zeros(n)
    if n > 1:
        step = safe_divide(close[1:] - close[:-1], close[:-1], fill=0.0) * volume[1:]
        out[1:] = np.cumsum(step)
    return out


# ── Percentage Volume Oscillator ─────────────────────────────────────

def pvo_series(
    volume: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentage Volume Oscillator (volume oscillator).

    ``PVO = 100 * (EMA(volume, fast) - EMA(volume, slow)) / EMA(volume, slow)``
    with SMA-seeded EMAs over the full series; ``signal = EMA(PVO, signal)``
    and ``histogram = PVO - signal``.

    Returns:
        ``(pvo, signal, histogram)``.
    """
    fast = ema_sma_seed(volume, fast_period)
    slow = ema_sma_seed(volume, slow_period)
    pvo = safe_divide(fast - slow, slow) * 100.0
    signal = ema_sma_seed(pvo, signal_period)
    return pvo, signal, pvo - signal
