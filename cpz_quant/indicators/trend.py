"""Trend indicators: SMA, EMA, WMA, DEMA, TEMA, KAMA, Supertrend, TRIMA, T3,
ZLEMA, HMA, ALMA, VIDYA, McGinley Dynamic, Parabolic SAR, Ichimoku, Aroon,
Vortex, TRIX, Mass Index, Midpoint, Midprice, Schaff Trend Cycle.

All functions operate on raw numpy arrays and return numpy arrays.
The public ``__init__`` API handles DataFrame extraction, Rust dispatch,
and scalar / Series conversion.
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


# ═══════════════════════════════════════════════════════════════════
#  Extended trend family
#
#  Conventions for everything below: inputs are float64 arrays, outputs
#  are aligned to the input with a leading NaN warm-up. Exponential
#  smoothers are seeded with the SMA of their first ``period`` valid
#  inputs (TA-Lib / TradingView convention) unless a docstring says
#  otherwise. Note the older ``ema_series`` above seeds with the first
#  price instead and emits no warm-up NaNs.
# ═══════════════════════════════════════════════════════════════════


def _wma_nan(values: np.ndarray, period: int) -> np.ndarray:
    """WMA that skips a leading NaN warm-up."""
    n = len(values)
    out = np.full(n, np.nan)
    start = first_valid_index(values)
    if start < n:
        out[start:] = wma_series(values[start:], period)
    return out


# ── TRIMA ────────────────────────────────────────────────────────────

def trima_series(close: np.ndarray, period: int = 30) -> np.ndarray:
    """Triangular Moving Average (TA-Lib ``TRIMA``).

    A weighted average with triangular weights ``1, 2, ..., peak, ..., 2, 1``.
    For odd *period* the peak weight ``(period + 1) / 2`` appears once; for
    even *period* the peak ``period / 2`` appears twice. This is identical to
    an SMA of an SMA with the two lengths TA-Lib uses.
    """
    n = len(close)
    out = np.full(n, np.nan)
    if period < 1 or n < period:
        return out
    if period % 2 == 1:
        half = (period + 1) // 2
        weights = np.r_[np.arange(1, half + 1), np.arange(half - 1, 0, -1)]
    else:
        half = period // 2
        weights = np.r_[np.arange(1, half + 1), np.arange(half, 0, -1)]
    weights = weights.astype(np.float64) / weights.sum()
    out[period - 1 :] = _rolling_view(close, period) @ weights
    return out


# ── T3 ───────────────────────────────────────────────────────────────

def t3_series(close: np.ndarray, period: int = 5, *, vfactor: float = 0.7) -> np.ndarray:
    """Tillson T3 moving average (TA-Lib ``T3``).

    Six chained EMAs ``e1..e6`` (each SMA-seeded on the previous stage's
    first valid values) combined as
    ``c1*e6 + c2*e5 + c3*e4 + c4*e3`` with ``a = vfactor`` and
    ``c1 = -a^3``, ``c2 = 3a^2 + 3a^3``, ``c3 = -6a^2 - 3a - 3a^3``,
    ``c4 = 1 + 3a + a^3 + 3a^2``. Warm-up is ``6 * (period - 1)`` bars.
    """
    e1 = ema_sma_seed(close, period)
    e2 = ema_sma_seed(e1, period)
    e3 = ema_sma_seed(e2, period)
    e4 = ema_sma_seed(e3, period)
    e5 = ema_sma_seed(e4, period)
    e6 = ema_sma_seed(e5, period)
    a = vfactor
    c1 = -(a**3)
    c2 = 3.0 * a**2 + 3.0 * a**3
    c3 = -6.0 * a**2 - 3.0 * a - 3.0 * a**3
    c4 = 1.0 + 3.0 * a + a**3 + 3.0 * a**2
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


# ── ZLEMA ────────────────────────────────────────────────────────────

def zlema_series(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Zero-Lag EMA (Ehlers and Way).

    ``EMA(2 * close - close[t - lag], period)`` with
    ``lag = (period - 1) // 2``; the EMA is SMA-seeded on the first
    *period* de-lagged values.
    """
    lag = (period - 1) // 2
    de_lagged = 2.0 * close - shift(close, lag)
    return ema_sma_seed(de_lagged, period)


# ── HMA ──────────────────────────────────────────────────────────────

def hma_series(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Hull Moving Average.

    ``WMA(2 * WMA(close, period // 2) - WMA(close, period), floor(sqrt(period)))``.
    """
    if period < 2:
        raise ValueError("hma period must be >= 2")
    half = _wma_nan(close, period // 2)
    full = _wma_nan(close, period)
    return _wma_nan(2.0 * half - full, int(np.sqrt(period)))


# ── ALMA ─────────────────────────────────────────────────────────────

def alma_series(
    close: np.ndarray,
    period: int = 9,
    *,
    offset: float = 0.85,
    sigma: float = 6.0,
) -> np.ndarray:
    """Arnaud Legoux Moving Average.

    Gaussian-weighted window with ``w_i = exp(-(i - m)^2 / (2 s^2))`` for
    ``i = 0`` (oldest) .. ``period - 1`` (newest), where
    ``m = floor(offset * (period - 1))`` and ``s = period / sigma``.
    The ``floor`` follows TradingView ``ta.alma`` and pandas-ta; Legoux's
    original note uses the unfloored ``m``, which differs only when
    ``offset * (period - 1)`` is not an integer.
    """
    n = len(close)
    out = np.full(n, np.nan)
    if period < 1 or n < period:
        return out
    m = np.floor(offset * (period - 1))
    s = period / sigma
    i = np.arange(period, dtype=np.float64)
    weights = np.exp(-((i - m) ** 2) / (2.0 * s * s))
    weights /= weights.sum()
    out[period - 1 :] = _rolling_view(close, period) @ weights
    return out


# ── VIDYA ────────────────────────────────────────────────────────────

def vidya_series(
    close: np.ndarray,
    period: int = 14,
    *,
    cmo_period: int = 9,
) -> np.ndarray:
    """Chande's Variable Index Dynamic Average.

    ``VIDYA[t] = k*alpha*close[t] + (1 - k*alpha) * VIDYA[t-1]`` with
    ``alpha = 2 / (period + 1)`` and ``k = |CMO(cmo_period)| / 100`` where
    CMO is Chande's original sum-based oscillator (not Wilder-smoothed).
    Seeded with the close on the first bar where the CMO exists
    (index ``cmo_period``). pandas-ta seeds with zero instead, which makes
    its early values collapse toward 0 before converging.
    """
    from .momentum import cmo_series

    n = len(close)
    out = np.full(n, np.nan)
    k = np.abs(cmo_series(close, cmo_period, smoothing="sum")) / 100.0
    start = cmo_period
    if start >= n:
        return out
    alpha = 2.0 / (period + 1)
    prev = close[start]
    out[start] = prev
    for i in range(start + 1, n):
        ak = alpha * k[i]
        prev = ak * close[i] + (1.0 - ak) * prev
        out[i] = prev
    return out


# ── McGinley Dynamic ─────────────────────────────────────────────────

def mcginley_series(close: np.ndarray, period: int = 10, *, k: float = 0.6) -> np.ndarray:
    """McGinley Dynamic.

    ``MD[t] = MD[t-1] + (close[t] - MD[t-1]) / (k * period * (close[t] / MD[t-1])^4)``,
    seeded with ``MD[0] = close[0]``. McGinley suggested ``k = 0.6`` (N at
    60% of the equivalent moving-average length). Note pandas-ta ``mcgd``
    uses the previous *close* rather than the previous MD, so it is not a
    usable reference.
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    prev = close[0]
    out[0] = prev
    denom_base = k * period
    for i in range(1, n):
        ratio = close[i] / prev if abs(prev) > EPSILON else np.nan
        prev = prev + (close[i] - prev) / (denom_base * ratio**4)
        out[i] = prev
    return out


# ── Parabolic SAR ────────────────────────────────────────────────────

def psar_series(
    high: np.ndarray,
    low: np.ndarray,
    *,
    acceleration: float = 0.02,
    maximum: float = 0.2,
) -> np.ndarray:
    """Wilder's Parabolic SAR, port of TA-Lib ``SAR``.

    The initial direction is short when the first bar's -DM is positive
    (``low[0] - low[1] > max(high[1] - high[0], 0)``), long otherwise. The
    first SAR (index 1) is the prior extreme of the opposite side. On a
    reversal the SAR jumps to the extreme point and the acceleration factor
    resets. The SAR never penetrates the current or prior bar's range.
    """
    n = len(high)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    af = acceleration
    diff_m = low[0] - low[1]
    diff_p = high[1] - high[0]
    minus_dm = diff_m if (diff_m > 0 and diff_p < diff_m) else 0.0
    is_long = not (minus_dm > 0)

    new_high = high[0]
    new_low = low[0]
    if is_long:
        ep = high[1]
        sar = new_low
    else:
        ep = low[1]
        sar = new_high
    new_low = low[1]
    new_high = high[1]

    for today in range(1, n):
        prev_low = new_low
        prev_high = new_high
        new_low = low[today]
        new_high = high[today]
        if is_long:
            if new_low <= sar:
                is_long = False
                sar = ep
                sar = max(sar, prev_high, new_high)
                out[today] = sar
                af = acceleration
                ep = new_low
                sar = sar + af * (ep - sar)
                sar = max(sar, prev_high, new_high)
            else:
                out[today] = sar
                if new_high > ep:
                    ep = new_high
                    af = min(af + acceleration, maximum)
                sar = sar + af * (ep - sar)
                sar = min(sar, prev_low, new_low)
        else:
            if new_high >= sar:
                is_long = True
                sar = ep
                sar = min(sar, prev_low, new_low)
                out[today] = sar
                af = acceleration
                ep = new_high
                sar = sar + af * (ep - sar)
                sar = min(sar, prev_low, new_low)
            else:
                out[today] = sar
                if new_low < ep:
                    ep = new_low
                    af = min(af + acceleration, maximum)
                sar = sar + af * (ep - sar)
                sar = max(sar, prev_high, new_high)
    return out


# ── Ichimoku ─────────────────────────────────────────────────────────

def ichimoku_series(
    high: np.ndarray,
    low: np.ndarray,
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
    *,
    displacement: int = 26,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ichimoku Kinko Hyo.

    - Tenkan-sen: midpoint of the *tenkan*-bar high/low range.
    - Kijun-sen: midpoint of the *kijun*-bar range.
    - Senkou span A: ``(tenkan + kijun) / 2`` plotted ahead.
    - Senkou span B: midpoint of the *senkou*-bar range plotted ahead.

    The spans are shifted forward by ``displacement - 1`` bars (the
    TradingView and pandas-ta charting convention, which counts the current
    bar as the first of the 26), so the value at index ``t`` only uses data
    up to ``t - displacement + 1``: no look-ahead. The Chikou span is not
    returned because it is the close shifted *backward*, which would place
    future prices at past indices.

    Returns:
        ``(tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b)``.
    """
    def mid(period: int) -> np.ndarray:
        return (rolling_apply(high, period, "max") + rolling_apply(low, period, "min")) / 2.0

    tenkan_sen = mid(tenkan)
    kijun_sen = mid(kijun)
    span_a = shift((tenkan_sen + kijun_sen) / 2.0, displacement - 1)
    span_b = shift(mid(senkou), displacement - 1)
    return tenkan_sen, kijun_sen, span_a, span_b


# ── Aroon ────────────────────────────────────────────────────────────

def aroon_series(
    high: np.ndarray,
    low: np.ndarray,
    period: int = 14,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aroon Down / Up (TA-Lib ``AROON``).

    Over the trailing ``period + 1`` bars,
    ``up = 100 * (period - bars_since_highest_high) / period`` and likewise
    for the lowest low. Ties resolve to the most recent bar, as in TA-Lib.

    Returns:
        ``(aroon_down, aroon_up)`` (TA-Lib output order).
    """
    n = len(high)
    down = np.full(n, np.nan)
    up = np.full(n, np.nan)
    w = period + 1
    if n < w:
        return down, up
    hw = _rolling_view(high, w)
    lw = _rolling_view(low, w)
    # argmax returns the first occurrence; flip to prefer the latest bar.
    since_high = np.argmax(hw[:, ::-1], axis=1)
    since_low = np.argmin(lw[:, ::-1], axis=1)
    up[period:] = 100.0 * (period - since_high) / period
    down[period:] = 100.0 * (period - since_low) / period
    return down, up


def aroon_osc_series(high: np.ndarray, low: np.ndarray, period: int = 14) -> np.ndarray:
    """Aroon Oscillator: ``aroon_up - aroon_down`` (TA-Lib ``AROONOSC``)."""
    down, up = aroon_series(high, low, period)
    return up - down


# ── Vortex ───────────────────────────────────────────────────────────

def vortex_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vortex Indicator (Botes and Siepman).

    ``VI+ = sum(|high - low[t-1]|) / sum(TR)`` and
    ``VI- = sum(|low - high[t-1]|) / sum(TR)`` over *period* bars, where TR
    is the true range (first computable at index 1).

    Returns:
        ``(vi_plus, vi_minus)``.
    """
    n = len(close)
    vm_plus = np.full(n, np.nan)
    vm_minus = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    if n >= 2:
        vm_plus[1:] = np.abs(high[1:] - low[:-1])
        vm_minus[1:] = np.abs(low[1:] - high[:-1])
        tr[1:] = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ])
    tr_sum = rolling_apply(tr, period, "sum")
    vi_plus = safe_divide(rolling_apply(vm_plus, period, "sum"), tr_sum)
    vi_minus = safe_divide(rolling_apply(vm_minus, period, "sum"), tr_sum)
    return vi_plus, vi_minus


# ── TRIX ─────────────────────────────────────────────────────────────

def trix_series(close: np.ndarray, period: int = 30) -> np.ndarray:
    """TRIX (TA-Lib ``TRIX``): 1-bar percent rate of change of a triple EMA."""
    e3 = ema_sma_seed(ema_sma_seed(ema_sma_seed(close, period), period), period)
    prev = shift(e3, 1)
    return safe_divide(e3 - prev, prev, fill=0.0) * 100.0


# ── Mass Index ───────────────────────────────────────────────────────

def mass_index_series(
    high: np.ndarray,
    low: np.ndarray,
    fast_period: int = 9,
    slow_period: int = 25,
) -> np.ndarray:
    """Dorsey's Mass Index.

    ``sum(EMA(H-L, fast) / EMA(EMA(H-L, fast), fast), slow)`` with
    SMA-seeded EMAs.
    """
    rng = high - low
    e1 = ema_sma_seed(rng, fast_period)
    e2 = ema_sma_seed(e1, fast_period)
    return rolling_apply(safe_divide(e1, e2), slow_period, "sum")


# ── Midpoint / Midprice ──────────────────────────────────────────────

def midpoint_series(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Midpoint of the rolling close range (TA-Lib ``MIDPOINT``)."""
    return (rolling_apply(close, period, "max") + rolling_apply(close, period, "min")) / 2.0


def midprice_series(high: np.ndarray, low: np.ndarray, period: int = 14) -> np.ndarray:
    """Midpoint of rolling highest high and lowest low (TA-Lib ``MIDPRICE``)."""
    return (rolling_apply(high, period, "max") + rolling_apply(low, period, "min")) / 2.0


# ── Schaff Trend Cycle ───────────────────────────────────────────────

def stc_series(
    close: np.ndarray,
    cycle: int = 10,
    fast_period: int = 23,
    slow_period: int = 50,
    *,
    factor: float = 0.5,
) -> np.ndarray:
    """Schaff Trend Cycle, as published in the widely used TradingView
    script by LazyBear.

    ``macd = EMA(fast) - EMA(slow)``; a *cycle*-bar stochastic of the MACD
    is smoothed with ``pf += factor * (stoch - pf)``; a second stochastic of
    ``pf`` is smoothed the same way to give STC (0..100). When a stochastic
    range is zero the previous stochastic value is carried forward. EMAs
    are SMA-seeded. Both smoothers start from their first input value.

    pandas-ta ``stc`` is not used as a reference: it only updates the first
    stochastic when the rolling *minimum* MACD is positive, which departs
    from the published definition.
    """
    n = len(close)
    out = np.full(n, np.nan)
    macd = ema_sma_seed(close, fast_period) - ema_sma_seed(close, slow_period)
    lo = rolling_apply(macd, cycle, "min")
    hi = rolling_apply(macd, cycle, "max")
    start = first_valid_index(lo)
    if start >= n:
        return out

    pf = np.full(n, np.nan)
    f1_prev = 0.0
    for i in range(start, n):
        rng = hi[i] - lo[i]
        f1 = (macd[i] - lo[i]) / rng * 100.0 if rng > 0 else f1_prev
        pf[i] = f1 if i == start else pf[i - 1] + factor * (f1 - pf[i - 1])
        f1_prev = f1

    lo2 = rolling_apply(pf, cycle, "min")
    hi2 = rolling_apply(pf, cycle, "max")
    start2 = first_valid_index(lo2)
    f2_prev = 0.0
    for i in range(start2, n):
        rng = hi2[i] - lo2[i]
        f2 = (pf[i] - lo2[i]) / rng * 100.0 if rng > 0 else f2_prev
        out[i] = f2 if i == start2 else out[i - 1] + factor * (f2 - out[i - 1])
        f2_prev = f2
    return out
