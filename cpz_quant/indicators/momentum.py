"""Momentum indicators: RSI, MACD, Stochastic, Williams %R, CCI, ROC, ADX, MFI, momentum,
slow Stochastic, Stochastic RSI, CMO, MOM / ROCP / ROCR, Ultimate Oscillator,
Awesome Oscillator, APO / PPO, KST, TSI, Connors RSI, Fisher Transform, Coppock,
DPO, Relative Vigor Index, Elder Ray, WaveTrend, Balance of Power, and the
Directional Movement family (+DM, -DM, +DI, -DI, DX, ADXR).

All functions operate on raw numpy arrays and return numpy arrays.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._helpers import (
    EPSILON,
    _rolling_view,
    ema_sma_seed,
    first_valid_index,
    rolling_apply,
    safe_divide,
    shift,
)
from .trend import ema_series, sma_series, wma_series

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


# ═══════════════════════════════════════════════════════════════════
#  Extended momentum family
#
#  Exponential smoothers below are SMA-seeded (TA-Lib convention) via
#  ``ema_sma_seed`` unless stated otherwise; outputs carry a leading NaN
#  warm-up aligned to the input.
# ═══════════════════════════════════════════════════════════════════


def _stoch_k(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """Raw %K with TA-Lib's zero-range rule (0 when the range is zero)."""
    hh = rolling_apply(high, period, "max")
    ll = rolling_apply(low, period, "min")
    return safe_divide(close - ll, hh - ll, fill=0.0) * 100.0


def _nan_sma(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan)
    start = first_valid_index(values)
    if start < n:
        out[start:] = sma_series(values[start:], period)
    return out


def _ma(values: np.ndarray, period: int, ma_type: str) -> np.ndarray:
    if ma_type == "sma":
        return _nan_sma(values, period)
    if ma_type == "ema":
        return ema_sma_seed(values, period)
    raise ValueError(f"Unknown ma_type '{ma_type}'. Use: sma, ema")


# ── Slow Stochastic ──────────────────────────────────────────────────

def stochastic_slow_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowd_period: int = 3,
    *,
    ma_type: str = "sma",
) -> Tuple[np.ndarray, np.ndarray]:
    """Slow Stochastic (TA-Lib ``STOCH``).

    ``slow_k = MA(raw_k, slowk_period)``, ``slow_d = MA(slow_k, slowd_period)``.
    A zero high-low range gives a raw %K of 0 (TA-Lib), unlike the fast
    ``stochastic_series`` above which uses 50.

    Returns:
        ``(slow_k, slow_d)``.
    """
    raw_k = _stoch_k(high, low, close, fastk_period)
    slow_k = _ma(raw_k, slowk_period, ma_type)
    slow_d = _ma(slow_k, slowd_period, ma_type)
    slow_k[np.isnan(slow_d)] = np.nan
    return slow_k, slow_d


# ── Stochastic RSI ───────────────────────────────────────────────────

def stochrsi_series(
    close: np.ndarray,
    period: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
    *,
    ma_type: str = "sma",
) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic RSI (TA-Lib ``STOCHRSI``): a fast stochastic of Wilder RSI.

    Returns:
        ``(fast_k, fast_d)`` on a 0..100 scale.
    """
    r = rsi_series(close, period)
    k = _stoch_k(r, r, r, fastk_period)
    d = _ma(k, fastd_period, ma_type)
    k[np.isnan(d)] = np.nan
    return k, d


# ── CMO ──────────────────────────────────────────────────────────────

def cmo_series(close: np.ndarray, period: int = 14, *, smoothing: str = "sum") -> np.ndarray:
    """Chande Momentum Oscillator (-100..100).

    ``100 * (up - down) / (up + down)`` where, with ``smoothing="sum"``
    (Chande's definition, TA-Lib ``CMOU``), ``up`` / ``down`` are the sums of
    gains / losses over *period* bars; with ``smoothing="wilder"`` they are
    Wilder-smoothed averages seeded like RSI (TA-Lib ``CMO``).
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    if smoothing == "sum":
        up = rolling_apply(gains, period, "sum")
        down = rolling_apply(losses, period, "sum")
    elif smoothing == "wilder":
        up = ema_sma_seed(gains, period, alpha=1.0 / period)
        down = ema_sma_seed(losses, period, alpha=1.0 / period)
    else:
        raise ValueError(f"Unknown smoothing '{smoothing}'. Use: sum, wilder")
    out[1:] = safe_divide(up - down, up + down, fill=0.0) * 100.0
    return out


# ── MOM / ROCP / ROCR ────────────────────────────────────────────────

def mom_series(close: np.ndarray, period: int = 10) -> np.ndarray:
    """Momentum as a price difference: ``close - close[t - period]`` (TA-Lib ``MOM``)."""
    return close - shift(close, period)


def rocp_series(close: np.ndarray, period: int = 10) -> np.ndarray:
    """Rate of change as a fraction: ``(close - prev) / prev`` (TA-Lib ``ROCP``).

    Returns 0 where ``prev`` is zero, matching TA-Lib.
    """
    prev = shift(close, period)
    return safe_divide(close - prev, prev, fill=0.0)


def rocr_series(close: np.ndarray, period: int = 10, *, scale: float = 1.0) -> np.ndarray:
    """Rate of change ratio: ``scale * close / prev``.

    ``scale=1`` is TA-Lib ``ROCR``; ``scale=100`` is ``ROCR100``. Returns 0
    where ``prev`` is zero, matching TA-Lib.
    """
    return safe_divide(close, shift(close, period), fill=0.0) * scale


# ── Ultimate Oscillator ──────────────────────────────────────────────

def ultosc_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period1: int = 7,
    period2: int = 14,
    period3: int = 28,
) -> np.ndarray:
    """Williams' Ultimate Oscillator (TA-Lib ``ULTOSC``).

    ``BP = close - min(low, prev_close)``, ``TR = max(high, prev_close) - min(low, prev_close)``;
    ``UO = 100 * (4*A1 + 2*A2 + A3) / 7`` with ``Ai = sum(BP)/sum(TR)`` over
    each period (a zero TR sum contributes 0).
    """
    n = len(close)
    bp = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    if n >= 2:
        true_low = np.minimum(low[1:], close[:-1])
        true_high = np.maximum(high[1:], close[:-1])
        bp[1:] = close[1:] - true_low
        tr[1:] = true_high - true_low

    def avg(p: int) -> np.ndarray:
        return safe_divide(rolling_apply(bp, p, "sum"), rolling_apply(tr, p, "sum"), fill=0.0)

    uo = 100.0 * (4.0 * avg(period1) + 2.0 * avg(period2) + avg(period3)) / 7.0
    uo[: max(period1, period2, period3)] = np.nan
    return uo


# ── Awesome Oscillator ───────────────────────────────────────────────

def awesome_oscillator_series(
    high: np.ndarray,
    low: np.ndarray,
    fast_period: int = 5,
    slow_period: int = 34,
) -> np.ndarray:
    """Bill Williams' Awesome Oscillator: ``SMA(hl2, 5) - SMA(hl2, 34)``."""
    median = (high + low) / 2.0
    return sma_series(median, fast_period) - sma_series(median, slow_period)


# ── APO / PPO ────────────────────────────────────────────────────────

def _po_mas(
    close: np.ndarray,
    fast_period: int,
    slow_period: int,
    ma_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast / slow MAs over the full series, masked to the slow warm-up.

    Periods are swapped if given in the wrong order, as TA-Lib does.
    """
    if fast_period > slow_period:
        fast_period, slow_period = slow_period, fast_period
    slow = _ma(close, slow_period, ma_type)
    fast = _ma(close, fast_period, ma_type)
    fast[np.isnan(slow)] = np.nan
    return fast, slow


def apo_series(
    close: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    *,
    ma_type: str = "ema",
) -> np.ndarray:
    """Absolute Price Oscillator: ``MA(fast) - MA(slow)`` (TA-Lib ``APO``)."""
    fast, slow = _po_mas(close, fast_period, slow_period, ma_type)
    return fast - slow


def ppo_series(
    close: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    *,
    ma_type: str = "ema",
) -> np.ndarray:
    """Percentage Price Oscillator: ``100 * (fast - slow) / slow`` (TA-Lib ``PPO``).

    TA-Lib's own default ``matype`` is SMA; the conventional (Appel /
    StockCharts) PPO uses EMAs, which is the default here. Both match TA-Lib
    when the same MA type is requested.
    """
    fast, slow = _po_mas(close, fast_period, slow_period, ma_type)
    return safe_divide(fast - slow, slow, fill=0.0) * 100.0


# ── KST ──────────────────────────────────────────────────────────────

def kst_series(
    close: np.ndarray,
    roc_periods: Tuple[int, int, int, int] = (10, 15, 20, 30),
    sma_periods: Tuple[int, int, int, int] = (10, 10, 10, 15),
    signal_period: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pring's Know Sure Thing.

    ``KST = 1*SMA(ROC1) + 2*SMA(ROC2) + 3*SMA(ROC3) + 4*SMA(ROC4)`` with
    percentage ROCs; the signal is an SMA of KST. pandas-ta ``kst`` returns
    the same series multiplied by an extra factor of 100.

    Returns:
        ``(kst, signal)``.
    """
    total = np.zeros(len(close))
    for weight, (rp, sp) in enumerate(zip(roc_periods, sma_periods), start=1):
        roc_pct = rocp_series(close, rp) * 100.0
        roc_pct[:rp] = np.nan
        total = total + weight * _nan_sma(roc_pct, sp)
    return total, _nan_sma(total, signal_period)


# ── TSI ──────────────────────────────────────────────────────────────

def tsi_series(
    close: np.ndarray,
    long_period: int = 25,
    short_period: int = 13,
    signal_period: int = 13,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blau's True Strength Index.

    ``100 * EMA(EMA(m, long), short) / EMA(EMA(|m|, long), short)`` where
    ``m = close - close[t-1]``; the signal is an EMA of TSI. EMAs are
    SMA-seeded.

    Returns:
        ``(tsi, signal)``.
    """
    m = shift(close, 0) - shift(close, 1)
    num = ema_sma_seed(ema_sma_seed(m, long_period), short_period)
    den = ema_sma_seed(ema_sma_seed(np.abs(m), long_period), short_period)
    tsi = safe_divide(num, den, fill=0.0) * 100.0
    return tsi, ema_sma_seed(tsi, signal_period)


# ── Connors RSI ──────────────────────────────────────────────────────

def _streak(close: np.ndarray) -> np.ndarray:
    n = len(close)
    out = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + 1.0 if out[i - 1] > 0 else 1.0
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - 1.0 if out[i - 1] < 0 else -1.0
    return out


def connors_rsi_series(
    close: np.ndarray,
    rsi_period: int = 3,
    streak_period: int = 2,
    rank_period: int = 100,
) -> np.ndarray:
    """Connors RSI (Connors Research).

    ``CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(ROC1, 100)) / 3``.
    The streak counts consecutive up (+1, +2, ...) or down (-1, -2, ...)
    closes and resets to 0 on an unchanged close. PercentRank is the
    percentage of the previous *rank_period* one-bar returns strictly less
    than the current one (Connors' definition; TradingView's
    ``ta.percentrank`` counts less-or-equal). RSIs are Wilder RSIs.
    """
    n = len(close)
    out = np.full(n, np.nan)
    r1 = rsi_series(close, rsi_period)
    r2 = rsi_series(_streak(close), streak_period)
    roc1 = rocp_series(close, 1)
    rank = np.full(n, np.nan)
    if n > rank_period + 1:
        prior = _rolling_view(roc1[1:], rank_period)[:-1]
        current = roc1[rank_period + 1 :]
        rank[rank_period + 1 :] = (prior < current[:, None]).sum(axis=1) * 100.0 / rank_period
    out[:] = (r1 + r2 + rank) / 3.0
    return out


# ── Fisher Transform ─────────────────────────────────────────────────

def fisher_transform_series(
    high: np.ndarray,
    low: np.ndarray,
    period: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ehlers' Fisher Transform (Stocks & Commodities, Nov 2002).

    With ``p = hl2`` normalised over *period* bars,
    ``v = 0.66 * (p_norm - 0.5) + 0.67 * v[t-1]`` clamped to +/-0.999 when
    beyond +/-0.99, and ``fish = 0.5 * ln((1 + v) / (1 - v)) + 0.5 * fish[t-1]``.
    The trigger is ``fish[t-1]``. Both recursions start from 0 on the first
    bar with a full window (TradingView convention); pandas-ta starts one
    bar later, a difference that decays geometrically (0.67^k).

    Returns:
        ``(fisher, trigger)``.
    """
    n = len(high)
    fish = np.full(n, np.nan)
    hl2 = (high + low) / 2.0
    hh = rolling_apply(hl2, period, "max")
    ll = rolling_apply(hl2, period, "min")
    v = 0.0
    f = 0.0
    for i in range(period - 1, n):
        rng = hh[i] - ll[i]
        norm = (hl2[i] - ll[i]) / rng if rng > EPSILON else 0.5
        v = 0.66 * (norm - 0.5) + 0.67 * v
        if v > 0.99:
            v = 0.999
        elif v < -0.99:
            v = -0.999
        f = 0.5 * np.log((1.0 + v) / (1.0 - v)) + 0.5 * f
        fish[i] = f
    return fish, shift(fish, 1)


# ── Coppock Curve ────────────────────────────────────────────────────

def coppock_series(
    close: np.ndarray,
    wma_period: int = 10,
    long_roc: int = 14,
    short_roc: int = 11,
) -> np.ndarray:
    """Coppock Curve: ``WMA(ROC(long) + ROC(short), wma_period)`` with percent ROCs."""
    total = (rocp_series(close, long_roc) + rocp_series(close, short_roc)) * 100.0
    total[: max(long_roc, short_roc)] = np.nan
    n = len(close)
    out = np.full(n, np.nan)
    start = first_valid_index(total)
    if start < n:
        out[start:] = wma_series(total[start:], wma_period)
    return out


# ── DPO ──────────────────────────────────────────────────────────────

def dpo_series(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Detrended Price Oscillator (StockCharts definition, no look-ahead).

    ``close[t - (period // 2 + 1)] - SMA(close, period)[t]``. Some charting
    packages plot this value shifted back by ``period // 2 + 1`` bars
    (pandas-ta ``centered=True``), which uses future data; that shift is not
    applied here.
    """
    return shift(close, period // 2 + 1) - sma_series(close, period)


# ── Relative Vigor Index ─────────────────────────────────────────────

def _swma4(values: np.ndarray) -> np.ndarray:
    """Symmetric 4-bar weighted MA with weights (1, 2, 2, 1) / 6."""
    return (values + 2.0 * shift(values, 1) + 2.0 * shift(values, 2) + shift(values, 3)) / 6.0


def rvgi_series(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ehlers' Relative Vigor Index.

    ``RVGI = sum(SWMA(close - open), period) / sum(SWMA(high - low), period)``
    with ``SWMA`` the (1, 2, 2, 1)/6 symmetric weighting; the signal is the
    SWMA of RVGI. Not to be confused with Dorsey's Relative Volatility Index.

    Returns:
        ``(rvgi, signal)``.
    """
    num = rolling_apply(_swma4(close - open_), period, "sum")
    den = rolling_apply(_swma4(high - low), period, "sum")
    rvgi = safe_divide(num, den)
    return rvgi, _swma4(rvgi)


# ── Elder Ray ────────────────────────────────────────────────────────

def elder_ray_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 13,
) -> Tuple[np.ndarray, np.ndarray]:
    """Elder Ray Index: ``bull = high - EMA(close)``, ``bear = low - EMA(close)``.

    Returns:
        ``(bull_power, bear_power)``.
    """
    e = ema_sma_seed(close, period)
    return high - e, low - e


# ── WaveTrend ────────────────────────────────────────────────────────

def wavetrend_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    channel_period: int = 10,
    average_period: int = 21,
    signal_period: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """WaveTrend Oscillator (LazyBear's published TradingView script).

    ``ap = hlc3``, ``esa = EMA(ap, n1)``, ``d = EMA(|ap - esa|, n1)``,
    ``ci = (ap - esa) / (0.015 * d)``, ``wt1 = EMA(ci, n2)``,
    ``wt2 = SMA(wt1, signal_period)``. EMAs are SMA-seeded, matching Pine
    Script's ``ta.ema``.

    Returns:
        ``(wt1, wt2)``.
    """
    ap = (high + low + close) / 3.0
    esa = ema_sma_seed(ap, channel_period)
    d = ema_sma_seed(np.abs(ap - esa), channel_period)
    ci = safe_divide(ap - esa, 0.015 * d)
    wt1 = ema_sma_seed(ci, average_period)
    return wt1, _nan_sma(wt1, signal_period)


# ── Balance of Power ─────────────────────────────────────────────────

def bop_series(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Balance of Power: ``(close - open) / (high - low)``, 0 on a zero range (TA-Lib ``BOP``)."""
    return safe_divide(close - open_, high - low, fill=0.0)


# ── Directional Movement family ──────────────────────────────────────

def _dm_tr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    if n >= 2:
        diff_p = high[1:] - high[:-1]
        diff_m = low[:-1] - low[1:]
        plus_dm[1:] = np.where((diff_p > 0) & (diff_p > diff_m), diff_p, 0.0)
        minus_dm[1:] = np.where((diff_m > 0) & (diff_m > diff_p), diff_m, 0.0)
        tr[1:] = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ])
    return plus_dm, minus_dm, tr


def _wilder_sum(values: np.ndarray, period: int, first_out: int) -> np.ndarray:
    """Wilder running sum ``S = S - S/period + x`` as TA-Lib's DM functions do.

    The sum is seeded with ``values[1 : period]`` (``period - 1`` terms) and
    the recursion is applied from index ``period`` onward, except for
    ``PLUS_DM`` / ``MINUS_DM`` style outputs (``first_out = period - 1``),
    which emit the seed sum itself.
    """
    n = len(values)
    out = np.full(n, np.nan)
    if n <= first_out or period < 2:
        return out
    s = float(np.sum(values[1:period]))
    if first_out == period - 1:
        out[period - 1] = s
    for i in range(period, n):
        s = s - s / period + values[i]
        out[i] = s
    return out


def dmi_components_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wilder's Directional Movement system, TA-Lib compatible.

    Returns:
        ``(plus_dm, minus_dm, plus_di, minus_di, dx)`` matching TA-Lib
        ``PLUS_DM``, ``MINUS_DM``, ``PLUS_DI``, ``MINUS_DI`` and ``DX``.
        DM sums start at index ``period - 1``; DI and DX at ``period``.
    """
    pdm, mdm, tr = _dm_tr(high, low, close)
    plus_dm = _wilder_sum(pdm, period, period - 1)
    minus_dm = _wilder_sum(mdm, period, period - 1)
    str_ = _wilder_sum(tr, period, period)
    spdm = plus_dm.copy()
    smdm = minus_dm.copy()
    spdm[: period] = np.nan
    smdm[: period] = np.nan
    plus_di = safe_divide(100.0 * spdm, str_, fill=0.0)
    minus_di = safe_divide(100.0 * smdm, str_, fill=0.0)
    dx = safe_divide(100.0 * np.abs(plus_di - minus_di), plus_di + minus_di, fill=0.0)
    return plus_dm, minus_dm, plus_di, minus_di, dx


def adx_talib_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> Tuple[np.ndarray, np.ndarray]:
    """ADX and ADXR exactly as TA-Lib computes them.

    ``ADX`` averages the first *period* DX values then Wilder-smooths
    (first value at ``2 * period - 1``); ``ADXR = (ADX + ADX[t - period + 1]) / 2``.
    The older ``adx_series`` above seeds its smoothing slightly differently,
    so its early values differ from TA-Lib by a decaying amount.

    Returns:
        ``(adx, adxr)``.
    """
    dx = dmi_components_series(high, low, close, period)[4]
    adx = ema_sma_seed(dx, period, alpha=1.0 / period)
    adxr = (adx + shift(adx, period - 1)) / 2.0
    return adx, adxr
