"""Candlestick pattern recognition, bit-compatible with TA-Lib's CDL* family.

Every kernel takes raw ``open_, high, low, close`` float64 arrays and returns an
int32 array of the same length using TA-Lib's encoding: ``+100`` bullish,
``-100`` bearish, ``0`` no pattern. A few patterns use extra levels exactly as
TA-Lib does (``+/-80`` for the equal-boundary engulfing and harami variants,
``+/-200`` for confirmed hikkake setups).

The reference is the TA-Lib C implementation. The kernels reproduce it
bit-for-bit, including the details that decide edge cases:

* Candle settings (TA-Lib ``TA_CandleDefaultSettings``). Each setting measures a
  candle against the average of a range (real body, high-low range, or the sum
  of both shadows) over the ``avg_period`` candles *before* the candle being
  judged, scaled by ``factor``. Shadow-type ranges are halved. A setting with
  ``avg_period == 0`` uses the judged candle's own range instead of an average.
* Running totals. TA-Lib keeps each rolling sum as a running total (add the
  newest range, subtract the one that left the window), so its averages differ
  from a freshly summed window in the last bits. The kernels replay the same
  sequence of float operations, which matters for price grids where values sit
  exactly on a threshold.
* Output alignment. Rows before the pattern's lookback are 0. Leading rows where
  any of the four inputs is NaN are skipped, and the lookback counts from the
  first complete row, as in the ``talib`` Python wrapper.
* ``fma``. The morning star, morning doji star, abandoned baby (bullish leg),
  piercing and thrusting tests use a fused multiply-add in TA-Lib; the kernels
  evaluate that product exactly.

Patterns with state (hikkake, modified hikkake) run a loop; all others are
vectorised fixed-window boolean expressions.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import numpy as np

# ── Candle settings ──────────────────────────────────────────────────

_RANGE_REAL_BODY = 0
_RANGE_HIGH_LOW = 1
_RANGE_SHADOWS = 2

# TA-Lib's upper bound for real-valued optional inputs (TA_REAL_MAX).
_TA_REAL_MAX = 3e37


class _CandleSetting(NamedTuple):
    """One entry of TA-Lib's candle settings table."""

    range_type: int
    avg_period: int
    factor: float


# TA_CandleDefaultSettings, in TA-Lib's order.
_BODY_LONG = _CandleSetting(_RANGE_REAL_BODY, 10, 1.0)
_BODY_VERY_LONG = _CandleSetting(_RANGE_REAL_BODY, 10, 3.0)
_BODY_SHORT = _CandleSetting(_RANGE_REAL_BODY, 10, 1.0)
_BODY_DOJI = _CandleSetting(_RANGE_HIGH_LOW, 10, 0.1)
_SHADOW_LONG = _CandleSetting(_RANGE_REAL_BODY, 0, 1.0)
_SHADOW_VERY_LONG = _CandleSetting(_RANGE_REAL_BODY, 0, 2.0)
_SHADOW_SHORT = _CandleSetting(_RANGE_SHADOWS, 10, 1.0)
_SHADOW_VERY_SHORT = _CandleSetting(_RANGE_HIGH_LOW, 10, 0.1)
_NEAR = _CandleSetting(_RANGE_HIGH_LOW, 5, 0.2)
_FAR = _CandleSetting(_RANGE_HIGH_LOW, 5, 0.6)
_EQUAL = _CandleSetting(_RANGE_HIGH_LOW, 5, 0.05)


def _period(*settings: _CandleSetting) -> int:
    """Largest averaging period among *settings* (TA-Lib's lookback building block)."""
    return max(s.avg_period for s in settings)


def _cmin(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """C ``min(a, b)`` macro: ``a < b ? a : b`` (argument order matters for NaN)."""
    return np.where(a < b, a, b)


def _cmax(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """C ``max(a, b)`` macro: ``a > b ? a : b``."""
    return np.where(a > b, a, b)


def _fma_scalar(x: float, y: float, z: float) -> float:
    """Correctly rounded ``x * y + z`` (IEEE fused multiply-add)."""
    fma = getattr(math, "fma", None)
    if fma is not None:
        return float(fma(x, y, z))
    return float(Fraction(x) * Fraction(y) + Fraction(z))


def _fma_near(x: np.ndarray, y: float, z: np.ndarray, near: np.ndarray) -> np.ndarray:
    """Fused ``x * y + z``, exact wherever the result is compared against *near*.

    The unfused ``x * y + z`` differs from the fused value by a few ulps at most,
    so it already decides every comparison with *near* except where the two
    operands are within that distance. Only those rows are recomputed exactly,
    which keeps the kernel vectorised.
    """
    prod = x * y
    naive = prod + z
    tol = 8.0 * (np.spacing(np.abs(naive)) + np.spacing(np.abs(prod)))
    with np.errstate(invalid="ignore"):
        close_call = np.abs(near - naive) <= tol
    idx = np.flatnonzero(close_call)
    if idx.size:
        naive = naive.copy()
        for j in idx.tolist():
            naive[j] = _fma_scalar(float(x[j]), float(y), float(z[j]))
    return naive


def _check_penetration(penetration: float) -> float:
    pen = float(penetration)
    if not (0.0 <= pen <= _TA_REAL_MAX):
        raise ValueError(f"penetration must be in [0, {_TA_REAL_MAX:g}], got {penetration!r}")
    return pen


# ── Bar context ──────────────────────────────────────────────────────


class _Bars:
    """Aligned OHLC views and TA-Lib candle averages for one kernel call.

    Accessors take ``k`` (how many bars back from the bar being evaluated) and
    return arrays covering every evaluated bar, i.e. absolute rows
    ``start - k .. n - 1 - k``.
    """

    def __init__(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        lookback: int,
    ) -> None:
        o = np.asarray(open_, dtype=np.float64)
        h = np.asarray(high, dtype=np.float64)
        lo = np.asarray(low, dtype=np.float64)
        c = np.asarray(close, dtype=np.float64)
        if o.ndim != 1 or h.ndim != 1 or lo.ndim != 1 or c.ndim != 1:
            raise ValueError("open_, high, low and close must be 1-D arrays")
        n = o.shape[0]
        if not (h.shape[0] == n and lo.shape[0] == n and c.shape[0] == n):
            raise ValueError("open_, high, low and close must have the same length")
        self.n = n
        self.out = np.zeros(n, dtype=np.int32)
        complete = ~(np.isnan(o) | np.isnan(h) | np.isnan(lo) | np.isnan(c))
        first = int(np.argmax(complete)) if n and bool(complete.any()) else max(n - 1, 0)
        self.start = first + lookback
        self.empty = self.start >= n
        self._o, self._h, self._l, self._c = o, h, lo, c
        white = c >= o
        self._color = np.where(white, 1, -1).astype(np.int64)
        self._body = np.abs(c - o)
        self._upper = h - np.where(white, c, o)
        self._lower = np.where(white, o, c) - lo
        self._avg_cache: Dict[Tuple[_CandleSetting, int, int], np.ndarray] = {}

    # Per-bar series, shifted k bars back.
    def _at(self, arr: np.ndarray, k: int) -> np.ndarray:
        return arr[self.start - k : self.n - k]

    def o(self, k: int) -> np.ndarray:
        return self._at(self._o, k)

    def h(self, k: int) -> np.ndarray:
        return self._at(self._h, k)

    def lo(self, k: int) -> np.ndarray:
        return self._at(self._l, k)

    def c(self, k: int) -> np.ndarray:
        return self._at(self._c, k)

    def color(self, k: int) -> np.ndarray:
        """TA_CANDLECOLOR: +1 when close >= open, else -1."""
        return self._at(self._color, k)

    def body(self, k: int) -> np.ndarray:
        """TA_REALBODY: ``|close - open|``."""
        return self._at(self._body, k)

    def upper(self, k: int) -> np.ndarray:
        """TA_UPPERSHADOW."""
        return self._at(self._upper, k)

    def lower(self, k: int) -> np.ndarray:
        """TA_LOWERSHADOW."""
        return self._at(self._lower, k)

    def gap_up(self, k2: int, k1: int) -> np.ndarray:
        """TA_REALBODYGAPUP(k2, k1): body of bar k2 entirely above body of bar k1."""
        return _cmin(self.o(k2), self.c(k2)) > _cmax(self.o(k1), self.c(k1))

    def gap_down(self, k2: int, k1: int) -> np.ndarray:
        """TA_REALBODYGAPDOWN(k2, k1): body of bar k2 entirely below body of bar k1."""
        return _cmax(self.o(k2), self.c(k2)) < _cmin(self.o(k1), self.c(k1))

    def _range(self, range_type: int) -> np.ndarray:
        if range_type == _RANGE_REAL_BODY:
            return self._body
        if range_type == _RANGE_HIGH_LOW:
            return self._h - self._l
        return self._upper + self._lower

    def avg(self, setting: _CandleSetting, k: int, init_back: int = 0) -> np.ndarray:
        """TA_CANDLEAVERAGE for bar ``i - k`` at every evaluated bar ``i``.

        Replays TA-Lib's running total: the window for bar ``j`` is the
        ``avg_period`` ranges before ``j``; the total is seeded at loop index
        ``start - init_back`` and then updated with ``total += new - old``.
        With ``init_back > 0`` the returned array also covers the
        ``init_back`` warm-up bars before ``start``.
        """
        key = (setting, k, init_back)
        cached = self._avg_cache.get(key)
        if cached is not None:
            return cached
        rng = self._range(setting.range_type)
        div = 2.0 if setting.range_type == _RANGE_SHADOWS else 1.0
        t0 = self.start - init_back
        p = setting.avg_period
        if p == 0:
            val = setting.factor * rng[t0 - k : self.n - k] / div
        else:
            total0 = 0.0
            for j in range(t0 - k - p, t0 - k):
                total0 += float(rng[j])
            diffs = rng[t0 - k : self.n - 1 - k] - rng[t0 - k - p : self.n - 1 - k - p]
            totals = np.add.accumulate(np.concatenate((np.array([total0]), diffs)))
            val = setting.factor * (totals / float(p)) / div
        self._avg_cache[key] = val
        return val

    def emit(self, values: np.ndarray) -> np.ndarray:
        self.out[self.start :] = values
        return self.out


def _signal(cond: np.ndarray, value: Union[int, np.ndarray]) -> np.ndarray:
    return np.where(cond, value, 0)


# ── Single-candle patterns ───────────────────────────────────────────


def cdl_doji(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Doji (TA-Lib CDLDOJI).

    The real body is no larger than 10% of the average high-low range of the
    previous 10 candles. Emits +100 on a doji.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_DOJI))
    if b.empty:
        return b.out
    return b.emit(_signal(b.body(0) <= b.avg(_BODY_DOJI, 0), 100))


def cdl_dragonfly_doji(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Dragonfly doji (TA-Lib CDLDRAGONFLYDOJI).

    A doji body with a very short upper shadow and a lower shadow that is not
    very short (both judged against 10% of the 10-candle average high-low
    range). Emits +100.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_DOJI, _SHADOW_VERY_SHORT))
    if b.empty:
        return b.out
    svs = b.avg(_SHADOW_VERY_SHORT, 0)
    cond = (b.body(0) <= b.avg(_BODY_DOJI, 0)) & (b.upper(0) < svs) & (b.lower(0) > svs)
    return b.emit(_signal(cond, 100))


def cdl_gravestone_doji(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Gravestone doji (TA-Lib CDLGRAVESTONEDOJI).

    A doji body with a very short lower shadow and an upper shadow that is not
    very short. Emits +100 (TA-Lib does not assign direction).
    """
    b = _Bars(open_, high, low, close, _period(_BODY_DOJI, _SHADOW_VERY_SHORT))
    if b.empty:
        return b.out
    svs = b.avg(_SHADOW_VERY_SHORT, 0)
    cond = (b.body(0) <= b.avg(_BODY_DOJI, 0)) & (b.lower(0) < svs) & (b.upper(0) > svs)
    return b.emit(_signal(cond, 100))


def cdl_long_legged_doji(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Long-legged doji (TA-Lib CDLLONGLEGGEDDOJI).

    A doji body with at least one shadow longer than the real body. Emits +100.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_DOJI, _SHADOW_LONG))
    if b.empty:
        return b.out
    sl = b.avg(_SHADOW_LONG, 0)
    cond = (b.body(0) <= b.avg(_BODY_DOJI, 0)) & ((b.lower(0) > sl) | (b.upper(0) > sl))
    return b.emit(_signal(cond, 100))


def cdl_rickshaw_man(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Rickshaw man (TA-Lib CDLRICKSHAWMAN).

    A doji with two long shadows whose body sits near the midpoint of the
    high-low range ("near" is 20% of the 5-candle average range). Emits +100.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_DOJI, _SHADOW_LONG, _NEAR))
    if b.empty:
        return b.out
    o0, h0, l0, c0 = b.o(0), b.h(0), b.lo(0), b.c(0)
    sl = b.avg(_SHADOW_LONG, 0)
    near = b.avg(_NEAR, 0)
    mid = l0 + (h0 - l0) / 2
    cond = (
        (b.body(0) <= b.avg(_BODY_DOJI, 0))
        & (b.lower(0) > sl)
        & (b.upper(0) > sl)
        & (_cmin(o0, c0) <= mid + near)
        & (_cmax(o0, c0) >= mid - near)
    )
    return b.emit(_signal(cond, 100))


def cdl_takuri(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Takuri, a dragonfly doji with a very long lower shadow (TA-Lib CDLTAKURI).

    Doji body, very short upper shadow, lower shadow longer than twice the real
    body. Emits +100.
    """
    b = _Bars(
        open_, high, low, close, _period(_BODY_DOJI, _SHADOW_VERY_SHORT, _SHADOW_VERY_LONG)
    )
    if b.empty:
        return b.out
    cond = (
        (b.body(0) <= b.avg(_BODY_DOJI, 0))
        & (b.upper(0) < b.avg(_SHADOW_VERY_SHORT, 0))
        & (b.lower(0) > b.avg(_SHADOW_VERY_LONG, 0))
    )
    return b.emit(_signal(cond, 100))


def cdl_marubozu(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Marubozu (TA-Lib CDLMARUBOZU).

    Long real body (above the 10-candle average body) with very short upper and
    lower shadows. Emits +100 for a white candle, -100 for a black one.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_LONG, _SHADOW_VERY_SHORT))
    if b.empty:
        return b.out
    svs = b.avg(_SHADOW_VERY_SHORT, 0)
    cond = (b.body(0) > b.avg(_BODY_LONG, 0)) & (b.upper(0) < svs) & (b.lower(0) < svs)
    return b.emit(_signal(cond, b.color(0) * 100))


def cdl_closing_marubozu(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Closing marubozu (TA-Lib CDLCLOSINGMARUBOZU).

    Long real body with a very short shadow on the closing side (upper shadow
    for white, lower shadow for black). Emits +100 white, -100 black.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_LONG, _SHADOW_VERY_SHORT))
    if b.empty:
        return b.out
    col = b.color(0)
    svs = b.avg(_SHADOW_VERY_SHORT, 0)
    cond = (b.body(0) > b.avg(_BODY_LONG, 0)) & (
        ((col == 1) & (b.upper(0) < svs)) | ((col == -1) & (b.lower(0) < svs))
    )
    return b.emit(_signal(cond, col * 100))


def cdl_belt_hold(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Belt hold (TA-Lib CDLBELTHOLD).

    Long real body with a very short shadow on the opening side (lower shadow
    for white, upper shadow for black). Emits +100 white, -100 black.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_LONG, _SHADOW_VERY_SHORT))
    if b.empty:
        return b.out
    col = b.color(0)
    svs = b.avg(_SHADOW_VERY_SHORT, 0)
    cond = (b.body(0) > b.avg(_BODY_LONG, 0)) & (
        ((col == 1) & (b.lower(0) < svs)) | ((col == -1) & (b.upper(0) < svs))
    )
    return b.emit(_signal(cond, col * 100))


def cdl_long_line(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Long line candle (TA-Lib CDLLONGLINE).

    Long real body with short upper and lower shadows (each below half the
    10-candle average of the shadow sum). Emits +100 white, -100 black.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_LONG, _SHADOW_SHORT))
    if b.empty:
        return b.out
    ss = b.avg(_SHADOW_SHORT, 0)
    cond = (b.body(0) > b.avg(_BODY_LONG, 0)) & (b.upper(0) < ss) & (b.lower(0) < ss)
    return b.emit(_signal(cond, b.color(0) * 100))


def cdl_short_line(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Short line candle (TA-Lib CDLSHORTLINE).

    Short real body (below the 10-candle average body) with short shadows.
    Emits +100 white, -100 black.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_SHORT, _SHADOW_SHORT))
    if b.empty:
        return b.out
    ss = b.avg(_SHADOW_SHORT, 0)
    cond = (b.body(0) < b.avg(_BODY_SHORT, 0)) & (b.upper(0) < ss) & (b.lower(0) < ss)
    return b.emit(_signal(cond, b.color(0) * 100))


def cdl_spinning_top(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Spinning top (TA-Lib CDLSPINNINGTOP).

    Short real body with both shadows longer than the body. Emits +100 white,
    -100 black.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_SHORT))
    if b.empty:
        return b.out
    body = b.body(0)
    cond = (b.upper(0) > body) & (b.lower(0) > body) & (body < b.avg(_BODY_SHORT, 0))
    return b.emit(_signal(cond, b.color(0) * 100))


def cdl_high_wave(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """High-wave candle (TA-Lib CDLHIGHWAVE).

    Short real body with both shadows longer than twice the body. Emits +100
    white, -100 black.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_SHORT, _SHADOW_VERY_LONG))
    if b.empty:
        return b.out
    svl = b.avg(_SHADOW_VERY_LONG, 0)
    cond = (b.body(0) < b.avg(_BODY_SHORT, 0)) & (b.upper(0) > svl) & (b.lower(0) > svl)
    return b.emit(_signal(cond, b.color(0) * 100))


# ── Hammer family ────────────────────────────────────────────────────


def cdl_hammer(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Hammer (TA-Lib CDLHAMMER).

    Short real body, lower shadow longer than the body, very short upper
    shadow, and the body at or near the prior candle's low. Emits +100.
    """
    lookback = _period(_BODY_SHORT, _SHADOW_LONG, _SHADOW_VERY_SHORT, _NEAR) + 1
    b = _Bars(open_, high, low, close, lookback)
    if b.empty:
        return b.out
    cond = (
        (b.body(0) < b.avg(_BODY_SHORT, 0))
        & (b.lower(0) > b.avg(_SHADOW_LONG, 0))
        & (b.upper(0) < b.avg(_SHADOW_VERY_SHORT, 0))
        & (_cmin(b.c(0), b.o(0)) <= b.lo(1) + b.avg(_NEAR, 1))
    )
    return b.emit(_signal(cond, 100))


def cdl_hanging_man(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Hanging man (TA-Lib CDLHANGINGMAN).

    Hammer-shaped candle (short body, long lower shadow, very short upper
    shadow) whose body is at or near the prior candle's high. Emits -100.
    """
    lookback = _period(_BODY_SHORT, _SHADOW_LONG, _SHADOW_VERY_SHORT, _NEAR) + 1
    b = _Bars(open_, high, low, close, lookback)
    if b.empty:
        return b.out
    cond = (
        (b.body(0) < b.avg(_BODY_SHORT, 0))
        & (b.lower(0) > b.avg(_SHADOW_LONG, 0))
        & (b.upper(0) < b.avg(_SHADOW_VERY_SHORT, 0))
        & (_cmin(b.c(0), b.o(0)) >= b.h(1) - b.avg(_NEAR, 1))
    )
    return b.emit(_signal(cond, -100))


def cdl_inverted_hammer(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Inverted hammer (TA-Lib CDLINVERTEDHAMMER).

    Short real body gapping down from the prior body, upper shadow longer than
    the body, very short lower shadow. Emits +100.
    """
    lookback = _period(_BODY_SHORT, _SHADOW_LONG, _SHADOW_VERY_SHORT) + 1
    b = _Bars(open_, high, low, close, lookback)
    if b.empty:
        return b.out
    cond = (
        b.gap_down(0, 1)
        & (b.body(0) < b.avg(_BODY_SHORT, 0))
        & (b.upper(0) > b.avg(_SHADOW_LONG, 0))
        & (b.lower(0) < b.avg(_SHADOW_VERY_SHORT, 0))
    )
    return b.emit(_signal(cond, 100))


def cdl_shooting_star(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Shooting star (TA-Lib CDLSHOOTINGSTAR).

    Short real body gapping up from the prior body, upper shadow longer than
    the body, very short lower shadow. Emits -100.
    """
    lookback = _period(_BODY_SHORT, _SHADOW_LONG, _SHADOW_VERY_SHORT) + 1
    b = _Bars(open_, high, low, close, lookback)
    if b.empty:
        return b.out
    cond = (
        b.gap_up(0, 1)
        & (b.body(0) < b.avg(_BODY_SHORT, 0))
        & (b.upper(0) > b.avg(_SHADOW_LONG, 0))
        & (b.lower(0) < b.avg(_SHADOW_VERY_SHORT, 0))
    )
    return b.emit(_signal(cond, -100))


# ── Two-candle patterns ──────────────────────────────────────────────


def cdl_engulfing(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Engulfing pattern (TA-Lib CDLENGULFING).

    A candle of the opposite colour whose real body engulfs the prior real
    body (one boundary may be equal). Emits +/-100 when both boundaries are
    strictly outside the prior body and +/-80 when one boundary is equal; the
    sign follows the engulfing candle's colour.
    """
    b = _Bars(open_, high, low, close, 2)
    if b.empty:
        return b.out
    o0, c0, o1, c1 = b.o(0), b.c(0), b.o(1), b.c(1)
    col0, col1 = b.color(0), b.color(1)
    bull = (col0 == 1) & (col1 == -1) & (((c0 >= o1) & (o0 < c1)) | ((c0 > o1) & (o0 <= c1)))
    bear = (col0 == -1) & (col1 == 1) & (((o0 >= c1) & (c0 < o1)) | ((o0 > c1) & (c0 <= o1)))
    strict = (o0 != c1) & (c0 != o1)
    value = np.where(strict, col0 * 100, col0 * 80)
    return b.emit(_signal(bull | bear, value))


def _harami_impl(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    second: _CandleSetting,
) -> np.ndarray:
    b = _Bars(open_, high, low, close, _period(second, _BODY_LONG) + 1)
    if b.empty:
        return b.out
    o0, c0, o1, c1 = b.o(0), b.c(0), b.o(1), b.c(1)
    base = (b.body(1) > b.avg(_BODY_LONG, 1)) & (b.body(0) <= b.avg(second, 0))
    hi0, lo0 = _cmax(c0, o0), _cmin(c0, o0)
    hi1, lo1 = _cmax(c1, o1), _cmin(c1, o1)
    strict = (hi0 < hi1) & (lo0 > lo1)
    loose = (hi0 <= hi1) & (lo0 >= lo1)
    sign = -b.color(1)
    value = np.where(strict, sign * 100, np.where(loose, sign * 80, 0))
    return b.emit(np.where(base, value, 0))


def cdl_harami(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Harami (TA-Lib CDLHARAMI).

    A long candle followed by a short candle whose real body lies inside the
    first body. Emits +/-100 when strictly inside and +/-80 when a boundary
    touches; the sign is opposite to the first candle's colour.
    """
    return _harami_impl(open_, high, low, close, _BODY_SHORT)


def cdl_harami_cross(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Harami cross (TA-Lib CDLHARAMICROSS).

    Harami whose second candle is a doji. Emits +/-100 strictly inside and
    +/-80 when a boundary touches; sign opposite to the first candle's colour.
    """
    return _harami_impl(open_, high, low, close, _BODY_DOJI)


def cdl_doji_star(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Doji star (TA-Lib CDLDOJISTAR).

    A long candle followed by a doji whose body gaps away in the direction of
    the first candle. Emits -100 after a white candle, +100 after a black one.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_DOJI, _BODY_LONG) + 1)
    if b.empty:
        return b.out
    col1 = b.color(1)
    cond = (
        (b.body(1) > b.avg(_BODY_LONG, 1))
        & (b.body(0) <= b.avg(_BODY_DOJI, 0))
        & (((col1 == 1) & b.gap_up(0, 1)) | ((col1 == -1) & b.gap_down(0, 1)))
    )
    return b.emit(_signal(cond, -col1 * 100))


def cdl_piercing(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Piercing line (TA-Lib CDLPIERCING).

    Long black candle, then a long white candle opening below the prior low and
    closing above the prior body's midpoint but inside the body. Emits +100.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_LONG) + 1)
    if b.empty:
        return b.out
    c0, c1 = b.c(0), b.c(1)
    body1 = b.body(1)
    cond = (
        (b.color(1) == -1)
        & (body1 > b.avg(_BODY_LONG, 1))
        & (b.color(0) == 1)
        & (b.body(0) > b.avg(_BODY_LONG, 0))
        & (b.o(0) < b.lo(1))
        & (c0 < b.o(1))
        & (c0 > _fma_near(body1, 0.5, c1, c0))
    )
    return b.emit(_signal(cond, 100))


def cdl_dark_cloud_cover(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    penetration: float = 0.5,
) -> np.ndarray:
    """Dark cloud cover (TA-Lib CDLDARKCLOUDCOVER).

    Long white candle, then a black candle opening above the prior high and
    closing inside the prior body, below ``close - body * penetration``.

    Args:
        penetration: Fraction of the first body the second close must reach
            into (TA-Lib default 0.5).
    """
    pen = _check_penetration(penetration)
    b = _Bars(open_, high, low, close, _period(_BODY_LONG) + 1)
    if b.empty:
        return b.out
    c0 = b.c(0)
    body1 = b.body(1)
    cond = (
        (b.color(1) == 1)
        & (body1 > b.avg(_BODY_LONG, 1))
        & (b.color(0) == -1)
        & (b.o(0) > b.h(1))
        & (c0 > b.o(1))
        & (c0 < b.c(1) - body1 * pen)
    )
    return b.emit(_signal(cond, -100))


def cdl_counterattack(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Counterattack lines (TA-Lib CDLCOUNTERATTACK).

    Two long candles of opposite colour with (nearly) equal closes ("equal" is
    5% of the 5-candle average range). Emits the second candle's colour * 100.
    """
    b = _Bars(open_, high, low, close, _period(_EQUAL, _BODY_LONG) + 1)
    if b.empty:
        return b.out
    c0, c1 = b.c(0), b.c(1)
    eq = b.avg(_EQUAL, 1)
    col0 = b.color(0)
    cond = (
        (b.color(1) == -col0)
        & (b.body(1) > b.avg(_BODY_LONG, 1))
        & (b.body(0) > b.avg(_BODY_LONG, 0))
        & (c0 <= c1 + eq)
        & (c0 >= c1 - eq)
    )
    return b.emit(_signal(cond, col0 * 100))


def cdl_homing_pigeon(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Homing pigeon (TA-Lib CDLHOMINGPIGEON).

    Long black candle followed by a short black candle whose body is strictly
    inside the first body. Emits +100.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_SHORT, _BODY_LONG) + 1)
    if b.empty:
        return b.out
    cond = (
        (b.color(1) == -1)
        & (b.color(0) == -1)
        & (b.body(1) > b.avg(_BODY_LONG, 1))
        & (b.body(0) <= b.avg(_BODY_SHORT, 0))
        & (b.o(0) < b.o(1))
        & (b.c(0) > b.c(1))
    )
    return b.emit(_signal(cond, 100))


def cdl_matching_low(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Matching low (TA-Lib CDLMATCHINGLOW).

    Two black candles with (nearly) equal closes. Emits +100.
    """
    b = _Bars(open_, high, low, close, _period(_EQUAL) + 1)
    if b.empty:
        return b.out
    c0, c1 = b.c(0), b.c(1)
    eq = b.avg(_EQUAL, 1)
    cond = (b.color(1) == -1) & (b.color(0) == -1) & (c0 <= c1 + eq) & (c0 >= c1 - eq)
    return b.emit(_signal(cond, 100))


def cdl_separating_lines(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Separating lines (TA-Lib CDLSEPARATINGLINES).

    Opposite-coloured candle opening at (nearly) the prior open, with a long
    body and a very short shadow on the opening side. Emits the second
    candle's colour * 100.
    """
    lookback = _period(_SHADOW_VERY_SHORT, _BODY_LONG, _EQUAL) + 1
    b = _Bars(open_, high, low, close, lookback)
    if b.empty:
        return b.out
    o0, o1 = b.o(0), b.o(1)
    col0 = b.color(0)
    eq = b.avg(_EQUAL, 1)
    svs = b.avg(_SHADOW_VERY_SHORT, 0)
    cond = (
        (b.color(1) == -col0)
        & (o0 <= o1 + eq)
        & (o0 >= o1 - eq)
        & (b.body(0) > b.avg(_BODY_LONG, 0))
        & (((col0 == 1) & (b.lower(0) < svs)) | ((col0 == -1) & (b.upper(0) < svs)))
    )
    return b.emit(_signal(cond, col0 * 100))


def cdl_in_neck(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """In-neck (TA-Lib CDLINNECK).

    Long black candle, then a white candle opening below the prior low and
    closing at or slightly above the prior close. Emits -100.
    """
    b = _Bars(open_, high, low, close, _period(_EQUAL, _BODY_LONG) + 1)
    if b.empty:
        return b.out
    c0, c1 = b.c(0), b.c(1)
    cond = (
        (b.color(1) == -1)
        & (b.body(1) > b.avg(_BODY_LONG, 1))
        & (b.color(0) == 1)
        & (b.o(0) < b.lo(1))
        & (c0 <= c1 + b.avg(_EQUAL, 1))
        & (c0 >= c1)
    )
    return b.emit(_signal(cond, -100))


def cdl_on_neck(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """On-neck (TA-Lib CDLONNECK).

    Long black candle, then a white candle opening below the prior low and
    closing (nearly) at the prior low. Emits -100.
    """
    b = _Bars(open_, high, low, close, _period(_EQUAL, _BODY_LONG) + 1)
    if b.empty:
        return b.out
    c0, l1 = b.c(0), b.lo(1)
    eq = b.avg(_EQUAL, 1)
    cond = (
        (b.color(1) == -1)
        & (b.body(1) > b.avg(_BODY_LONG, 1))
        & (b.color(0) == 1)
        & (b.o(0) < l1)
        & (c0 <= l1 + eq)
        & (c0 >= l1 - eq)
    )
    return b.emit(_signal(cond, -100))


def cdl_thrusting(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Thrusting (TA-Lib CDLTHRUSTING).

    Long black candle, then a white candle opening below the prior low and
    closing above the prior close (by more than "equal") but at or below the
    prior body's midpoint. Emits -100.
    """
    b = _Bars(open_, high, low, close, _period(_EQUAL, _BODY_LONG) + 1)
    if b.empty:
        return b.out
    c0, c1 = b.c(0), b.c(1)
    body1 = b.body(1)
    cond = (
        (b.color(1) == -1)
        & (body1 > b.avg(_BODY_LONG, 1))
        & (b.color(0) == 1)
        & (b.o(0) < b.lo(1))
        & (c0 > c1 + b.avg(_EQUAL, 1))
        & (c0 <= _fma_near(body1, 0.5, c1, c0))
    )
    return b.emit(_signal(cond, -100))


def _kicking_cond(b: _Bars) -> np.ndarray:
    svs1 = b.avg(_SHADOW_VERY_SHORT, 1)
    svs0 = b.avg(_SHADOW_VERY_SHORT, 0)
    col1 = b.color(1)
    return (
        (col1 == -b.color(0))
        & (b.body(1) > b.avg(_BODY_LONG, 1))
        & (b.upper(1) < svs1)
        & (b.lower(1) < svs1)
        & (b.body(0) > b.avg(_BODY_LONG, 0))
        & (b.upper(0) < svs0)
        & (b.lower(0) < svs0)
        & (((col1 == -1) & (b.lo(0) > b.h(1))) | ((col1 == 1) & (b.h(0) < b.lo(1))))
    )


def cdl_kicking(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Kicking (TA-Lib CDLKICKING).

    Two opposite-coloured marubozu candles separated by a full high-low gap.
    Emits the second candle's colour * 100.
    """
    b = _Bars(open_, high, low, close, _period(_SHADOW_VERY_SHORT, _BODY_LONG) + 1)
    if b.empty:
        return b.out
    return b.emit(_signal(_kicking_cond(b), b.color(0) * 100))


def cdl_kicking_by_length(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Kicking, direction set by the longer marubozu (TA-Lib CDLKICKINGBYLENGTH).

    Same shape as kicking; the sign is the colour of whichever candle has the
    longer real body (the first candle on a tie).
    """
    b = _Bars(open_, high, low, close, _period(_SHADOW_VERY_SHORT, _BODY_LONG) + 1)
    if b.empty:
        return b.out
    longer = np.where(b.body(0) > b.body(1), b.color(0), b.color(1))
    return b.emit(_signal(_kicking_cond(b), longer * 100))


# ── Three-candle patterns ────────────────────────────────────────────


def _star_body_gap(b: _Bars, bullish: bool) -> np.ndarray:
    return b.gap_down(1, 2) if bullish else b.gap_up(1, 2)


def cdl_morning_star(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    penetration: float = 0.3,
) -> np.ndarray:
    """Morning star (TA-Lib CDLMORNINGSTAR).

    Long black candle, a short candle whose body gaps down, then a white candle
    (longer than short) closing above ``close1 + body1 * penetration`` of the
    first candle. Emits +100.

    Args:
        penetration: Fraction of the first body the third close must recover
            (TA-Lib default 0.3).
    """
    pen = _check_penetration(penetration)
    b = _Bars(open_, high, low, close, _period(_BODY_SHORT, _BODY_LONG) + 2)
    if b.empty:
        return b.out
    c0 = b.c(0)
    body2 = b.body(2)
    cond = (
        (b.color(2) == -1)
        & (b.color(0) == 1)
        & _star_body_gap(b, True)
        & (c0 > _fma_near(body2, pen, b.c(2), c0))
        & (body2 > b.avg(_BODY_LONG, 2))
        & (b.body(1) <= b.avg(_BODY_SHORT, 1))
        & (b.body(0) > b.avg(_BODY_SHORT, 0))
    )
    return b.emit(_signal(cond, 100))


def cdl_evening_star(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    penetration: float = 0.3,
) -> np.ndarray:
    """Evening star (TA-Lib CDLEVENINGSTAR).

    Long white candle, a short candle whose body gaps up, then a black candle
    (longer than short) closing below ``close1 - body1 * penetration``.
    Emits -100.

    Args:
        penetration: Fraction of the first body the third close must give back
            (TA-Lib default 0.3).
    """
    pen = _check_penetration(penetration)
    b = _Bars(open_, high, low, close, _period(_BODY_SHORT, _BODY_LONG) + 2)
    if b.empty:
        return b.out
    body2 = b.body(2)
    cond = (
        (b.color(2) == 1)
        & (b.color(0) == -1)
        & _star_body_gap(b, False)
        & (b.c(0) < b.c(2) - body2 * pen)
        & (body2 > b.avg(_BODY_LONG, 2))
        & (b.body(1) <= b.avg(_BODY_SHORT, 1))
        & (b.body(0) > b.avg(_BODY_SHORT, 0))
    )
    return b.emit(_signal(cond, -100))


def cdl_morning_doji_star(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    penetration: float = 0.3,
) -> np.ndarray:
    """Morning doji star (TA-Lib CDLMORNINGDOJISTAR).

    Morning star whose middle candle is a doji. Emits +100.

    Args:
        penetration: Fraction of the first body the third close must recover
            (TA-Lib default 0.3).
    """
    pen = _check_penetration(penetration)
    lookback = _period(_BODY_DOJI, _BODY_LONG, _BODY_SHORT) + 2
    b = _Bars(open_, high, low, close, lookback)
    if b.empty:
        return b.out
    c0 = b.c(0)
    body2 = b.body(2)
    cond = (
        (b.color(2) == -1)
        & (b.color(0) == 1)
        & _star_body_gap(b, True)
        & (c0 > _fma_near(body2, pen, b.c(2), c0))
        & (body2 > b.avg(_BODY_LONG, 2))
        & (b.body(1) <= b.avg(_BODY_DOJI, 1))
        & (b.body(0) > b.avg(_BODY_SHORT, 0))
    )
    return b.emit(_signal(cond, 100))


def cdl_evening_doji_star(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    penetration: float = 0.3,
) -> np.ndarray:
    """Evening doji star (TA-Lib CDLEVENINGDOJISTAR).

    Evening star whose middle candle is a doji. Emits -100.

    Args:
        penetration: Fraction of the first body the third close must give back
            (TA-Lib default 0.3).
    """
    pen = _check_penetration(penetration)
    lookback = _period(_BODY_DOJI, _BODY_LONG, _BODY_SHORT) + 2
    b = _Bars(open_, high, low, close, lookback)
    if b.empty:
        return b.out
    body2 = b.body(2)
    cond = (
        (b.color(2) == 1)
        & (b.color(0) == -1)
        & _star_body_gap(b, False)
        & (b.c(0) < b.c(2) - body2 * pen)
        & (body2 > b.avg(_BODY_LONG, 2))
        & (b.body(1) <= b.avg(_BODY_DOJI, 1))
        & (b.body(0) > b.avg(_BODY_SHORT, 0))
    )
    return b.emit(_signal(cond, -100))


def cdl_abandoned_baby(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    penetration: float = 0.3,
) -> np.ndarray:
    """Abandoned baby (TA-Lib CDLABANDONEDBABY).

    Long candle, a doji separated from both neighbours by full high-low gaps,
    then an opposite candle (longer than short) that reaches ``penetration`` of
    the first body back. Emits +100 bullish (black first), -100 bearish.

    Args:
        penetration: Fraction of the first body the third close must reach
            (TA-Lib default 0.3).
    """
    pen = _check_penetration(penetration)
    lookback = _period(_BODY_DOJI, _BODY_LONG, _BODY_SHORT) + 2
    b = _Bars(open_, high, low, close, lookback)
    if b.empty:
        return b.out
    c0, c2 = b.c(0), b.c(2)
    h0, l0, h1, l1, h2, l2 = b.h(0), b.lo(0), b.h(1), b.lo(1), b.h(2), b.lo(2)
    body2 = b.body(2)
    col0, col2 = b.color(0), b.color(2)
    bearish = (col2 == 1) & (col0 == -1) & (c0 < c2 - body2 * pen) & (l1 > h2) & (h0 < l1)
    bullish = (
        (col2 == -1)
        & (col0 == 1)
        & (c0 > _fma_near(body2, pen, c2, c0))
        & (h1 < l2)
        & (l0 > h1)
    )
    cond = (
        (body2 > b.avg(_BODY_LONG, 2))
        & (b.body(1) <= b.avg(_BODY_DOJI, 1))
        & (b.body(0) > b.avg(_BODY_SHORT, 0))
        & (bearish | bullish)
    )
    return b.emit(_signal(cond, col0 * 100))


def cdl_tristar(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Tristar (TA-Lib CDLTRISTAR).

    Three doji (all judged against the average range before the first one).
    Emits -100 when the second doji gaps up and the third's body top is below
    the second's; +100 when the second gaps down and the third's body bottom is
    above the second's.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_DOJI) + 2)
    if b.empty:
        return b.out
    doji = b.avg(_BODY_DOJI, 2)
    o0, c0, o1, c1 = b.o(0), b.c(0), b.o(1), b.c(1)
    three = (b.body(2) <= doji) & (b.body(1) <= doji) & (b.body(0) <= doji)
    value = np.zeros(len(three), dtype=np.int32)
    value[three & b.gap_up(1, 2) & (_cmax(o0, c0) < _cmax(o1, c1))] = -100
    value[three & b.gap_down(1, 2) & (_cmin(o0, c0) > _cmin(o1, c1))] = 100
    return b.emit(value)


def cdl_three_inside(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Three inside up/down (TA-Lib CDL3INSIDE).

    Long candle, a short candle inside its body (harami), then a candle of the
    opposite colour to the first closing beyond the first open. Emits +100 when
    the first candle is black, -100 when white.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_SHORT, _BODY_LONG) + 2)
    if b.empty:
        return b.out
    o1, c1, o2, c2, c0 = b.o(1), b.c(1), b.o(2), b.c(2), b.c(0)
    col0, col2 = b.color(0), b.color(2)
    cond = (
        (_cmax(c1, o1) < _cmax(c2, o2))
        & (_cmin(c1, o1) > _cmin(c2, o2))
        & (((col2 == 1) & (col0 == -1) & (c0 < o2)) | ((col2 == -1) & (col0 == 1) & (c0 > o2)))
        & (b.body(2) > b.avg(_BODY_LONG, 2))
        & (b.body(1) <= b.avg(_BODY_SHORT, 1))
    )
    return b.emit(_signal(cond, -col2 * 100))


def cdl_three_outside(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Three outside up/down (TA-Lib CDL3OUTSIDE).

    An engulfing pair followed by a candle closing beyond the engulfing
    candle's close in its direction. Emits the engulfing candle's colour * 100.
    """
    b = _Bars(open_, high, low, close, 3)
    if b.empty:
        return b.out
    o1, c1, o2, c2, c0 = b.o(1), b.c(1), b.o(2), b.c(2), b.c(0)
    col1, col2 = b.color(1), b.color(2)
    cond = ((col1 == 1) & (col2 == -1) & (c1 > o2) & (o1 < c2) & (c0 > c1)) | (
        (col1 == -1) & (col2 == 1) & (o1 > c2) & (c1 < o2) & (c0 < c1)
    )
    return b.emit(_signal(cond, col1 * 100))


def cdl_three_white_soldiers(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Three advancing white soldiers (TA-Lib CDL3WHITESOLDIERS).

    Three white candles with rising closes, very short upper shadows, each
    opening within or near the prior body, bodies not shrinking by "far", and a
    last body that is not short. Emits +100.
    """
    lookback = _period(_SHADOW_VERY_SHORT, _BODY_SHORT, _FAR, _NEAR) + 2
    b = _Bars(open_, high, low, close, lookback)
    if b.empty:
        return b.out
    o0, o1, o2, c0, c1, c2 = b.o(0), b.o(1), b.o(2), b.c(0), b.c(1), b.c(2)
    body0, body1, body2 = b.body(0), b.body(1), b.body(2)
    cond = (
        (b.color(2) == 1)
        & (b.upper(2) < b.avg(_SHADOW_VERY_SHORT, 2))
        & (b.color(1) == 1)
        & (b.upper(1) < b.avg(_SHADOW_VERY_SHORT, 1))
        & (b.color(0) == 1)
        & (b.upper(0) < b.avg(_SHADOW_VERY_SHORT, 0))
        & (c0 > c1)
        & (c1 > c2)
        & (o1 > o2)
        & (o1 <= c2 + b.avg(_NEAR, 2))
        & (o0 > o1)
        & (o0 <= c1 + b.avg(_NEAR, 1))
        & (body1 > body2 - b.avg(_FAR, 2))
        & (body0 > body1 - b.avg(_FAR, 1))
        & (body0 > b.avg(_BODY_SHORT, 0))
    )
    return b.emit(_signal(cond, 100))


def cdl_three_black_crows(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Three black crows (TA-Lib CDL3BLACKCROWS).

    A white candle followed by three black candles with falling closes, each
    opening inside the prior black body and closing near its low (very short
    lower shadow); the white candle's high is above the first crow's close.
    Emits -100.
    """
    b = _Bars(open_, high, low, close, _period(_SHADOW_VERY_SHORT) + 3)
    if b.empty:
        return b.out
    o0, o1, o2, c0, c1, c2 = b.o(0), b.o(1), b.o(2), b.c(0), b.c(1), b.c(2)
    cond = (
        (b.color(3) == 1)
        & (b.color(2) == -1)
        & (b.color(1) == -1)
        & (b.color(0) == -1)
        & (o1 < o2)
        & (o1 > c2)
        & (o0 < o1)
        & (o0 > c1)
        & (b.h(3) > c2)
        & (c2 > c1)
        & (c1 > c0)
        & (b.lower(2) < b.avg(_SHADOW_VERY_SHORT, 2))
        & (b.lower(1) < b.avg(_SHADOW_VERY_SHORT, 1))
        & (b.lower(0) < b.avg(_SHADOW_VERY_SHORT, 0))
    )
    return b.emit(_signal(cond, -100))


def cdl_identical_three_crows(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Identical three crows (TA-Lib CDLIDENTICAL3CROWS).

    Three black candles with falling closes and very short lower shadows, each
    opening (nearly) at the prior close. Emits -100.
    """
    b = _Bars(open_, high, low, close, _period(_SHADOW_VERY_SHORT, _EQUAL) + 2)
    if b.empty:
        return b.out
    o0, o1, c0, c1, c2 = b.o(0), b.o(1), b.c(0), b.c(1), b.c(2)
    eq2, eq1 = b.avg(_EQUAL, 2), b.avg(_EQUAL, 1)
    cond = (
        (b.color(2) == -1)
        & (b.lower(2) < b.avg(_SHADOW_VERY_SHORT, 2))
        & (b.color(1) == -1)
        & (b.lower(1) < b.avg(_SHADOW_VERY_SHORT, 1))
        & (b.color(0) == -1)
        & (b.lower(0) < b.avg(_SHADOW_VERY_SHORT, 0))
        & (c2 > c1)
        & (c1 > c0)
        & (o1 <= c2 + eq2)
        & (o1 >= c2 - eq2)
        & (o0 <= c1 + eq1)
        & (o0 >= c1 - eq1)
    )
    return b.emit(_signal(cond, -100))


def cdl_two_crows(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Two crows (TA-Lib CDL2CROWS).

    Long white candle, a black candle whose body gaps up, then a black candle
    opening inside the second body and closing inside the first body.
    Emits -100.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_LONG) + 2)
    if b.empty:
        return b.out
    o0, c0 = b.o(0), b.c(0)
    cond = (
        (b.color(2) == 1)
        & (b.body(2) > b.avg(_BODY_LONG, 2))
        & (b.color(1) == -1)
        & b.gap_up(1, 2)
        & (b.color(0) == -1)
        & (o0 < b.o(1))
        & (o0 > b.c(1))
        & (c0 > b.o(2))
        & (c0 < b.c(2))
    )
    return b.emit(_signal(cond, -100))


def cdl_upside_gap_two_crows(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Upside gap two crows (TA-Lib CDLUPSIDEGAP2CROWS).

    Long white candle, a short black candle gapping up, then a black candle
    engulfing the second but closing above the first close. Emits -100.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_SHORT, _BODY_LONG) + 2)
    if b.empty:
        return b.out
    c0, c1 = b.c(0), b.c(1)
    cond = (
        (b.color(2) == 1)
        & (b.body(2) > b.avg(_BODY_LONG, 2))
        & (b.color(1) == -1)
        & (b.body(1) <= b.avg(_BODY_SHORT, 1))
        & b.gap_up(1, 2)
        & (b.color(0) == -1)
        & (b.o(0) > b.o(1))
        & (c0 < c1)
        & (c0 > b.c(2))
    )
    return b.emit(_signal(cond, -100))


def cdl_three_stars_in_south(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Three stars in the south (TA-Lib CDL3STARSINSOUTH).

    Three black candles: a long one with a long lower shadow, a smaller one
    opening above the prior close and trading within its range with a lower
    low shadow, then a small marubozu inside the second candle's range.
    Emits +100.
    """
    lookback = _period(_SHADOW_VERY_SHORT, _SHADOW_LONG, _BODY_LONG, _BODY_SHORT) + 2
    b = _Bars(open_, high, low, close, lookback)
    if b.empty:
        return b.out
    o1, c2 = b.o(1), b.c(2)
    h1, l1, h2, l2 = b.h(1), b.lo(1), b.h(2), b.lo(2)
    body2 = b.body(2)
    svs0 = b.avg(_SHADOW_VERY_SHORT, 0)
    cond = (
        (b.color(2) == -1)
        & (b.color(1) == -1)
        & (b.color(0) == -1)
        & (body2 > b.avg(_BODY_LONG, 2))
        & (b.lower(2) > b.avg(_SHADOW_LONG, 2))
        & (b.body(1) < body2)
        & (o1 > c2)
        & (o1 <= h2)
        & (l1 < c2)
        & (l1 >= l2)
        & (b.lower(1) > b.avg(_SHADOW_VERY_SHORT, 1))
        & (b.body(0) < b.avg(_BODY_SHORT, 0))
        & (b.lower(0) < svs0)
        & (b.upper(0) < svs0)
        & (b.lo(0) > l1)
        & (b.h(0) < h1)
    )
    return b.emit(_signal(cond, 100))


def cdl_advance_block(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Advance block (TA-Lib CDLADVANCEBLOCK).

    Three white candles with rising closes, each opening within or near the
    prior body, a long first body with a short upper shadow, and signs of
    weakening: shrinking bodies and/or lengthening upper shadows on the later
    candles. Emits -100.
    """
    lookback = _period(_SHADOW_LONG, _SHADOW_SHORT, _FAR, _NEAR, _BODY_LONG) + 2
    b = _Bars(open_, high, low, close, lookback)
    if b.empty:
        return b.out
    o0, o1, o2, c0, c1, c2 = b.o(0), b.o(1), b.o(2), b.c(0), b.c(1), b.c(2)
    body0, body1, body2 = b.body(0), b.body(1), b.body(2)
    upper0, upper1 = b.upper(0), b.upper(1)
    weakening = (
        ((body1 < body2 - b.avg(_FAR, 2)) & (body0 < body1 + b.avg(_NEAR, 1)))
        | (body0 < body1 - b.avg(_FAR, 1))
        | (
            (body0 < body1)
            & (body1 < body2)
            & ((upper0 > b.avg(_SHADOW_SHORT, 0)) | (upper1 > b.avg(_SHADOW_SHORT, 1)))
        )
        | ((body0 < body1) & (upper0 > b.avg(_SHADOW_LONG, 0)))
    )
    cond = (
        (b.color(2) == 1)
        & (b.color(1) == 1)
        & (b.color(0) == 1)
        & (c0 > c1)
        & (c1 > c2)
        & (o1 > o2)
        & (o1 <= c2 + b.avg(_NEAR, 2))
        & (o0 > o1)
        & (o0 <= c1 + b.avg(_NEAR, 1))
        & (body2 > b.avg(_BODY_LONG, 2))
        & (b.upper(2) < b.avg(_SHADOW_SHORT, 2))
        & weakening
    )
    return b.emit(_signal(cond, -100))


def cdl_stalled_pattern(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Stalled pattern (TA-Lib CDLSTALLEDPATTERN).

    Two long white candles (the second with a very short upper shadow, opening
    within or near the first body) followed by a small white candle that opens
    around the prior close. Emits -100.
    """
    lookback = _period(_BODY_LONG, _BODY_SHORT, _SHADOW_VERY_SHORT, _NEAR) + 2
    b = _Bars(open_, high, low, close, lookback)
    if b.empty:
        return b.out
    o1, c0, c1, c2 = b.o(1), b.c(0), b.c(1), b.c(2)
    body0 = b.body(0)
    cond = (
        (b.color(2) == 1)
        & (b.color(1) == 1)
        & (b.color(0) == 1)
        & (c0 > c1)
        & (c1 > c2)
        & (b.body(2) > b.avg(_BODY_LONG, 2))
        & (b.body(1) > b.avg(_BODY_LONG, 1))
        & (b.upper(1) < b.avg(_SHADOW_VERY_SHORT, 1))
        & (o1 > b.o(2))
        & (o1 <= c2 + b.avg(_NEAR, 2))
        & (body0 < b.avg(_BODY_SHORT, 0))
        & (b.o(0) >= c1 - body0 - b.avg(_NEAR, 1))
    )
    return b.emit(_signal(cond, -100))


def cdl_stick_sandwich(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Stick sandwich (TA-Lib CDLSTICKSANDWICH).

    Black, white (trading above the first close), black candle closing
    (nearly) at the first close. Emits +100.
    """
    b = _Bars(open_, high, low, close, _period(_EQUAL) + 2)
    if b.empty:
        return b.out
    c0, c2 = b.c(0), b.c(2)
    eq = b.avg(_EQUAL, 2)
    cond = (
        (b.color(2) == -1)
        & (b.color(1) == 1)
        & (b.color(0) == -1)
        & (b.lo(1) > c2)
        & (c0 <= c2 + eq)
        & (c0 >= c2 - eq)
    )
    return b.emit(_signal(cond, 100))


def cdl_unique_three_river(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Unique three river (TA-Lib CDLUNIQUE3RIVER).

    Long black candle, a black harami-like candle with a lower low, then a
    short white candle opening above the second low. Emits +100.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_SHORT, _BODY_LONG) + 2)
    if b.empty:
        return b.out
    l1 = b.lo(1)
    cond = (
        (b.color(2) == -1)
        & (b.color(1) == -1)
        & (b.color(0) == 1)
        & (b.c(1) > b.c(2))
        & (b.o(1) <= b.o(2))
        & (l1 < b.lo(2))
        & (b.o(0) > l1)
        & (b.body(2) > b.avg(_BODY_LONG, 2))
        & (b.body(0) < b.avg(_BODY_SHORT, 0))
    )
    return b.emit(_signal(cond, 100))


def cdl_tasuki_gap(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Tasuki gap (TA-Lib CDLTASUKIGAP).

    A body gap in the trend direction, a candle continuing it, then an
    opposite candle of (nearly) the same body size that opens inside the second
    body and closes inside the gap without filling it. Emits the second
    candle's colour * 100.
    """
    b = _Bars(open_, high, low, close, _period(_NEAR) + 2)
    if b.empty:
        return b.out
    o0, o1, o2, c0, c1, c2 = b.o(0), b.o(1), b.o(2), b.c(0), b.c(1), b.c(2)
    col0, col1 = b.color(0), b.color(1)
    same_size = np.abs(b.body(1) - b.body(0)) < b.avg(_NEAR, 1)
    up = (
        b.gap_up(1, 2)
        & (col1 == 1)
        & (col0 == -1)
        & (o0 < c1)
        & (o0 > o1)
        & (c0 < o1)
        & (c0 > _cmax(c2, o2))
        & same_size
    )
    down = (
        b.gap_down(1, 2)
        & (col1 == -1)
        & (col0 == 1)
        & (o0 < o1)
        & (o0 > c1)
        & (c0 > o1)
        & (c0 < _cmin(c2, o2))
        & same_size
    )
    return b.emit(_signal(up | down, col1 * 100))


def cdl_gap_side_side_white(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Up/down-gap side-by-side white lines (TA-Lib CDLGAPSIDESIDEWHITE).

    Two white candles of similar size opening at (nearly) the same level, both
    with bodies gapped away from the candle before them. Emits +100 for an
    upside gap, -100 for a downside gap.
    """
    b = _Bars(open_, high, low, close, _period(_NEAR, _EQUAL) + 2)
    if b.empty:
        return b.out
    o0, o1 = b.o(0), b.o(1)
    body0, body1 = b.body(0), b.body(1)
    near = b.avg(_NEAR, 1)
    eq = b.avg(_EQUAL, 1)
    gap_up_1 = b.gap_up(1, 2)
    cond = (
        ((gap_up_1 & b.gap_up(0, 2)) | (b.gap_down(1, 2) & b.gap_down(0, 2)))
        & (b.color(1) == 1)
        & (b.color(0) == 1)
        & (body0 >= body1 - near)
        & (body0 <= body1 + near)
        & (o0 >= o1 - eq)
        & (o0 <= o1 + eq)
    )
    return b.emit(_signal(cond, np.where(gap_up_1, 100, -100)))


def cdl_xside_gap_three_methods(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Upside/downside gap three methods (TA-Lib CDLXSIDEGAP3METHODS).

    Two same-coloured candles with a body gap between them, then an opposite
    candle opening inside the second body and closing inside the first body
    (filling the gap). Emits the first candle's colour * 100.
    """
    b = _Bars(open_, high, low, close, 2)
    if b.empty:
        return b.out
    o0, c0, o1, c1, o2, c2 = b.o(0), b.c(0), b.o(1), b.c(1), b.o(2), b.c(2)
    col0, col1, col2 = b.color(0), b.color(1), b.color(2)
    cond = (
        (col2 == col1)
        & (col1 == -col0)
        & (o0 < _cmax(c1, o1))
        & (o0 > _cmin(c1, o1))
        & (c0 < _cmax(c2, o2))
        & (c0 > _cmin(c2, o2))
        & (((col2 == 1) & b.gap_up(1, 2)) | ((col2 == -1) & b.gap_down(1, 2)))
    )
    return b.emit(_signal(cond, col2 * 100))


def cdl_three_line_strike(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Three-line strike (TA-Lib CDL3LINESTRIKE).

    Three same-coloured candles with closes stepping in their direction, each
    opening within or near the prior body, then an opposite candle opening
    beyond the third close and closing beyond the first open. Emits the colour
    of the three-candle run * 100.
    """
    b = _Bars(open_, high, low, close, _period(_NEAR) + 3)
    if b.empty:
        return b.out
    o0, o1, o2, o3 = b.o(0), b.o(1), b.o(2), b.o(3)
    c0, c1, c2, c3 = b.c(0), b.c(1), b.c(2), b.c(3)
    col1 = b.color(1)
    near3, near2 = b.avg(_NEAR, 3), b.avg(_NEAR, 2)
    cond = (
        (b.color(3) == b.color(2))
        & (b.color(2) == col1)
        & (b.color(0) == -col1)
        & (o2 >= _cmin(o3, c3) - near3)
        & (o2 <= _cmax(o3, c3) + near3)
        & (o1 >= _cmin(o2, c2) - near2)
        & (o1 <= _cmax(o2, c2) + near2)
        & (
            ((col1 == 1) & (c1 > c2) & (c2 > c3) & (o0 > c1) & (c0 < o3))
            | ((col1 == -1) & (c1 < c2) & (c2 < c3) & (o0 < c1) & (c0 > o3))
        )
    )
    return b.emit(_signal(cond, col1 * 100))


def cdl_concealing_baby_swallow(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Concealing baby swallow (TA-Lib CDLCONCEALBABYSWALL).

    Two black marubozu, a black candle gapping down with an upper shadow
    reaching into the prior body, then a black candle engulfing the third
    candle's full range. Emits +100.
    """
    b = _Bars(open_, high, low, close, _period(_SHADOW_VERY_SHORT) + 3)
    if b.empty:
        return b.out
    h1 = b.h(1)
    svs3 = b.avg(_SHADOW_VERY_SHORT, 3)
    svs2 = b.avg(_SHADOW_VERY_SHORT, 2)
    cond = (
        (b.color(3) == -1)
        & (b.color(2) == -1)
        & (b.color(1) == -1)
        & (b.color(0) == -1)
        & (b.lower(3) < svs3)
        & (b.upper(3) < svs3)
        & (b.lower(2) < svs2)
        & (b.upper(2) < svs2)
        & b.gap_down(1, 2)
        & (b.upper(1) > b.avg(_SHADOW_VERY_SHORT, 1))
        & (h1 > b.c(2))
        & (b.h(0) > h1)
        & (b.lo(0) < b.lo(1))
    )
    return b.emit(_signal(cond, 100))


def cdl_ladder_bottom(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Ladder bottom (TA-Lib CDLLADDERBOTTOM).

    Three black candles with falling opens and closes, a black candle with an
    upper shadow, then a white candle opening above the prior open and closing
    above the prior high. Emits +100.
    """
    b = _Bars(open_, high, low, close, _period(_SHADOW_VERY_SHORT) + 4)
    if b.empty:
        return b.out
    o1, o2, o3, o4 = b.o(1), b.o(2), b.o(3), b.o(4)
    c2, c3, c4 = b.c(2), b.c(3), b.c(4)
    cond = (
        (b.color(4) == -1)
        & (b.color(3) == -1)
        & (b.color(2) == -1)
        & (o4 > o3)
        & (o3 > o2)
        & (c4 > c3)
        & (c3 > c2)
        & (b.color(1) == -1)
        & (b.upper(1) > b.avg(_SHADOW_VERY_SHORT, 1))
        & (b.color(0) == 1)
        & (b.o(0) > o1)
        & (b.c(0) > b.h(1))
    )
    return b.emit(_signal(cond, 100))


# ── Five-candle patterns ─────────────────────────────────────────────


def cdl_breakaway(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Breakaway (TA-Lib CDLBREAKAWAY).

    Long candle, a same-coloured candle gapping away, two candles extending the
    move (the fourth the same colour), then an opposite candle closing inside
    the gap. Emits the last candle's colour * 100.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_LONG) + 4)
    if b.empty:
        return b.out
    h1, h2, h3 = b.h(1), b.h(2), b.h(3)
    l1, l2, l3 = b.lo(1), b.lo(2), b.lo(3)
    c0, o3, c4 = b.c(0), b.o(3), b.c(4)
    col0, col1, col3, col4 = b.color(0), b.color(1), b.color(3), b.color(4)
    down = (
        (col4 == -1)
        & b.gap_down(3, 4)
        & (h2 < h3)
        & (l2 < l3)
        & (h1 < h2)
        & (l1 < l2)
        & (c0 > o3)
        & (c0 < c4)
    )
    up = (
        (col4 == 1)
        & b.gap_up(3, 4)
        & (h2 > h3)
        & (l2 > l3)
        & (h1 > h2)
        & (l1 > l2)
        & (c0 < o3)
        & (c0 > c4)
    )
    cond = (
        (col4 == col3)
        & (col3 == col1)
        & (col1 == -col0)
        & (b.body(4) > b.avg(_BODY_LONG, 4))
        & (down | up)
    )
    return b.emit(_signal(cond, col0 * 100))


def cdl_rise_fall_three_methods(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Rising/falling three methods (TA-Lib CDLRISEFALL3METHODS).

    Long candle, three short counter-trend candles held within its high-low
    range with closes stepping against it, then a long candle of the original
    colour opening beyond the fourth close and closing beyond the first close.
    Emits the first candle's colour * 100.
    """
    b = _Bars(open_, high, low, close, _period(_BODY_SHORT, _BODY_LONG) + 4)
    if b.empty:
        return b.out
    h4, l4 = b.h(4), b.lo(4)
    col4 = b.color(4)
    inside = np.ones(b.n - b.start, dtype=bool)
    for k in (3, 2, 1):
        ok, ck = b.o(k), b.c(k)
        inside &= (_cmin(ok, ck) < h4) & (_cmax(ok, ck) > l4)
    cond = (
        (col4 == -b.color(3))
        & (b.color(3) == b.color(2))
        & (b.color(2) == b.color(1))
        & (b.color(1) == -b.color(0))
        & inside
        & (b.c(2) * col4 < b.c(3) * col4)
        & (b.c(1) * col4 < b.c(2) * col4)
        & (b.o(0) * col4 > b.c(1) * col4)
        & (b.c(0) * col4 > b.c(4) * col4)
        & (b.body(4) > b.avg(_BODY_LONG, 4))
        & (b.body(3) < b.avg(_BODY_SHORT, 3))
        & (b.body(2) < b.avg(_BODY_SHORT, 2))
        & (b.body(1) < b.avg(_BODY_SHORT, 1))
        & (b.body(0) > b.avg(_BODY_LONG, 0))
    )
    return b.emit(_signal(cond, 100 * col4))


def cdl_mat_hold(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    penetration: float = 0.5,
) -> np.ndarray:
    """Mat hold (TA-Lib CDLMATHOLD).

    Long white candle, a short black candle gapping up, two more short candles
    that hold above ``close1 - body1 * penetration`` with falling body tops,
    then a white candle opening above the prior close and closing above the
    three reaction highs. Emits +100.

    Args:
        penetration: Maximum fraction of the first body the reaction candles
            may retrace (TA-Lib default 0.5).
    """
    pen = _check_penetration(penetration)
    b = _Bars(open_, high, low, close, _period(_BODY_SHORT, _BODY_LONG) + 4)
    if b.empty:
        return b.out
    o1, o2, c1, c2, c4 = b.o(1), b.o(2), b.c(1), b.c(2), b.c(4)
    body4 = b.body(4)
    bottom2, bottom1 = _cmin(o2, c2), _cmin(o1, c1)
    floor = c4 - body4 * pen
    cond = (
        (b.color(4) == 1)
        & (b.color(3) == -1)
        & (b.color(0) == 1)
        & b.gap_up(3, 4)
        & (bottom2 < c4)
        & (bottom1 < c4)
        & (bottom2 > floor)
        & (bottom1 > floor)
        & (_cmax(c2, o2) < b.o(3))
        & (_cmax(c1, o1) < _cmax(c2, o2))
        & (b.o(0) > c1)
        & (b.c(0) > _cmax(_cmax(b.h(3), b.h(2)), b.h(1)))
        & (body4 > b.avg(_BODY_LONG, 4))
        & (b.body(3) < b.avg(_BODY_SHORT, 3))
        & (b.body(2) < b.avg(_BODY_SHORT, 2))
        & (b.body(1) < b.avg(_BODY_SHORT, 1))
    )
    return b.emit(_signal(cond, 100))


# ── Stateful patterns ────────────────────────────────────────────────


def cdl_hikkake(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Hikkake (TA-Lib CDLHIKKAKE).

    An inside bar followed by a bar breaking out of it on both high and low
    (the setup) emits +100 when the breakout is downward and -100 when upward.
    If within the next three bars the close reverses beyond the inside bar's
    high (bullish) or low (bearish), that bar emits +200 / -200. Setups formed
    during the warm-up bars can still confirm, as in TA-Lib.
    """
    lookback = 5
    b = _Bars(open_, high, low, close, lookback)
    if b.empty:
        return b.out
    h = b._h.tolist()
    lo = b._l.tolist()
    c = b._c.tolist()
    out: List[int] = []
    count = 0
    result = 0
    saved_high = 0.0
    saved_low = 0.0
    for i in range(b.start - 3, b.n):
        value = 0
        if (
            h[i - 1] < h[i - 2]
            and lo[i - 1] > lo[i - 2]
            and ((h[i] < h[i - 1] and lo[i] < lo[i - 1]) or (h[i] > h[i - 1] and lo[i] > lo[i - 1]))
        ):
            result = 100 if h[i] < h[i - 1] else -100
            saved_high = h[i - 1]
            saved_low = lo[i - 1]
            count = 4
            value = result
        elif count > 0 and (
            (result > 0 and c[i] > saved_high) or (result < 0 and c[i] < saved_low)
        ):
            value = result + (100 if result > 0 else -100)
            count = 0
        if count > 0:
            count -= 1
        if i >= b.start:
            out.append(value)
    return b.emit(np.asarray(out, dtype=np.int32))


def cdl_hikkake_modified(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Modified hikkake (TA-Lib CDLHIKKAKEMOD).

    Two consecutive inside bars, then a breakout bar; the second bar's close
    must sit near its low for a bullish setup (+100, downward breakout) or near
    its high for a bearish setup (-100, upward breakout). A close beyond the
    last inside bar's high/low within the next three bars emits +200 / -200.
    """
    lookback = max(1, _NEAR.avg_period) + 5
    b = _Bars(open_, high, low, close, lookback)
    if b.empty:
        return b.out
    warmup = 3
    near = b.avg(_NEAR, 2, init_back=warmup).tolist()
    h = b._h.tolist()
    lo = b._l.tolist()
    c = b._c.tolist()
    out: List[int] = []
    count = 0
    result = 0
    pattern_high = 0.0
    pattern_low = 0.0
    first = b.start - warmup
    for i in range(first, b.n):
        value = 0
        nr = near[i - first]
        if (
            h[i - 2] < h[i - 3]
            and lo[i - 2] > lo[i - 3]
            and h[i - 1] < h[i - 2]
            and lo[i - 1] > lo[i - 2]
            and (
                (h[i] < h[i - 1] and lo[i] < lo[i - 1] and c[i - 2] <= lo[i - 2] + nr)
                or (h[i] > h[i - 1] and lo[i] > lo[i - 1] and c[i - 2] >= h[i - 2] - nr)
            )
        ):
            result = 100 if h[i] < h[i - 1] else -100
            pattern_high = h[i - 1]
            pattern_low = lo[i - 1]
            count = 4
            value = result
        elif count > 0 and (
            (result > 0 and c[i] > pattern_high) or (result < 0 and c[i] < pattern_low)
        ):
            value = result + (100 if result > 0 else -100)
            count = 0
        if count > 0:
            count -= 1
        if i >= b.start:
            out.append(value)
    return b.emit(np.asarray(out, dtype=np.int32))


# ── Registry ─────────────────────────────────────────────────────────

CANDLE_PATTERNS: Dict[str, Callable[..., np.ndarray]] = {
    "two_crows": cdl_two_crows,
    "three_black_crows": cdl_three_black_crows,
    "three_inside": cdl_three_inside,
    "three_line_strike": cdl_three_line_strike,
    "three_outside": cdl_three_outside,
    "three_stars_in_south": cdl_three_stars_in_south,
    "three_white_soldiers": cdl_three_white_soldiers,
    "abandoned_baby": cdl_abandoned_baby,
    "advance_block": cdl_advance_block,
    "belt_hold": cdl_belt_hold,
    "breakaway": cdl_breakaway,
    "closing_marubozu": cdl_closing_marubozu,
    "concealing_baby_swallow": cdl_concealing_baby_swallow,
    "counterattack": cdl_counterattack,
    "dark_cloud_cover": cdl_dark_cloud_cover,
    "doji": cdl_doji,
    "doji_star": cdl_doji_star,
    "dragonfly_doji": cdl_dragonfly_doji,
    "engulfing": cdl_engulfing,
    "evening_doji_star": cdl_evening_doji_star,
    "evening_star": cdl_evening_star,
    "gap_side_side_white": cdl_gap_side_side_white,
    "gravestone_doji": cdl_gravestone_doji,
    "hammer": cdl_hammer,
    "hanging_man": cdl_hanging_man,
    "harami": cdl_harami,
    "harami_cross": cdl_harami_cross,
    "high_wave": cdl_high_wave,
    "hikkake": cdl_hikkake,
    "hikkake_modified": cdl_hikkake_modified,
    "homing_pigeon": cdl_homing_pigeon,
    "identical_three_crows": cdl_identical_three_crows,
    "in_neck": cdl_in_neck,
    "inverted_hammer": cdl_inverted_hammer,
    "kicking": cdl_kicking,
    "kicking_by_length": cdl_kicking_by_length,
    "ladder_bottom": cdl_ladder_bottom,
    "long_legged_doji": cdl_long_legged_doji,
    "long_line": cdl_long_line,
    "marubozu": cdl_marubozu,
    "matching_low": cdl_matching_low,
    "mat_hold": cdl_mat_hold,
    "morning_doji_star": cdl_morning_doji_star,
    "morning_star": cdl_morning_star,
    "on_neck": cdl_on_neck,
    "piercing": cdl_piercing,
    "rickshaw_man": cdl_rickshaw_man,
    "rise_fall_three_methods": cdl_rise_fall_three_methods,
    "separating_lines": cdl_separating_lines,
    "shooting_star": cdl_shooting_star,
    "short_line": cdl_short_line,
    "spinning_top": cdl_spinning_top,
    "stalled_pattern": cdl_stalled_pattern,
    "stick_sandwich": cdl_stick_sandwich,
    "takuri": cdl_takuri,
    "tasuki_gap": cdl_tasuki_gap,
    "thrusting": cdl_thrusting,
    "tristar": cdl_tristar,
    "unique_three_river": cdl_unique_three_river,
    "upside_gap_two_crows": cdl_upside_gap_two_crows,
    "xside_gap_three_methods": cdl_xside_gap_three_methods,
}

TALIB_NAMES: Dict[str, str] = {
    "two_crows": "CDL2CROWS",
    "three_black_crows": "CDL3BLACKCROWS",
    "three_inside": "CDL3INSIDE",
    "three_line_strike": "CDL3LINESTRIKE",
    "three_outside": "CDL3OUTSIDE",
    "three_stars_in_south": "CDL3STARSINSOUTH",
    "three_white_soldiers": "CDL3WHITESOLDIERS",
    "abandoned_baby": "CDLABANDONEDBABY",
    "advance_block": "CDLADVANCEBLOCK",
    "belt_hold": "CDLBELTHOLD",
    "breakaway": "CDLBREAKAWAY",
    "closing_marubozu": "CDLCLOSINGMARUBOZU",
    "concealing_baby_swallow": "CDLCONCEALBABYSWALL",
    "counterattack": "CDLCOUNTERATTACK",
    "dark_cloud_cover": "CDLDARKCLOUDCOVER",
    "doji": "CDLDOJI",
    "doji_star": "CDLDOJISTAR",
    "dragonfly_doji": "CDLDRAGONFLYDOJI",
    "engulfing": "CDLENGULFING",
    "evening_doji_star": "CDLEVENINGDOJISTAR",
    "evening_star": "CDLEVENINGSTAR",
    "gap_side_side_white": "CDLGAPSIDESIDEWHITE",
    "gravestone_doji": "CDLGRAVESTONEDOJI",
    "hammer": "CDLHAMMER",
    "hanging_man": "CDLHANGINGMAN",
    "harami": "CDLHARAMI",
    "harami_cross": "CDLHARAMICROSS",
    "high_wave": "CDLHIGHWAVE",
    "hikkake": "CDLHIKKAKE",
    "hikkake_modified": "CDLHIKKAKEMOD",
    "homing_pigeon": "CDLHOMINGPIGEON",
    "identical_three_crows": "CDLIDENTICAL3CROWS",
    "in_neck": "CDLINNECK",
    "inverted_hammer": "CDLINVERTEDHAMMER",
    "kicking": "CDLKICKING",
    "kicking_by_length": "CDLKICKINGBYLENGTH",
    "ladder_bottom": "CDLLADDERBOTTOM",
    "long_legged_doji": "CDLLONGLEGGEDDOJI",
    "long_line": "CDLLONGLINE",
    "marubozu": "CDLMARUBOZU",
    "matching_low": "CDLMATCHINGLOW",
    "mat_hold": "CDLMATHOLD",
    "morning_doji_star": "CDLMORNINGDOJISTAR",
    "morning_star": "CDLMORNINGSTAR",
    "on_neck": "CDLONNECK",
    "piercing": "CDLPIERCING",
    "rickshaw_man": "CDLRICKSHAWMAN",
    "rise_fall_three_methods": "CDLRISEFALL3METHODS",
    "separating_lines": "CDLSEPARATINGLINES",
    "shooting_star": "CDLSHOOTINGSTAR",
    "short_line": "CDLSHORTLINE",
    "spinning_top": "CDLSPINNINGTOP",
    "stalled_pattern": "CDLSTALLEDPATTERN",
    "stick_sandwich": "CDLSTICKSANDWICH",
    "takuri": "CDLTAKURI",
    "tasuki_gap": "CDLTASUKIGAP",
    "thrusting": "CDLTHRUSTING",
    "tristar": "CDLTRISTAR",
    "unique_three_river": "CDLUNIQUE3RIVER",
    "upside_gap_two_crows": "CDLUPSIDEGAP2CROWS",
    "xside_gap_three_methods": "CDLXSIDEGAP3METHODS",
}
