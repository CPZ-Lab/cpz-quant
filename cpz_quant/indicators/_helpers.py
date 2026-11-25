"""Shared utilities for indicator computation.

Centralises array extraction, constants, and common operations
so individual indicator modules stay focused on math.

Inputs accept a polars or pandas ``DataFrame`` (columns picked by name),
a polars or pandas ``Series``, a 1-D ``numpy.ndarray``, or a plain
``list``. pandas is never imported: frames are recognised by type name.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import polars as pl

# ── Constants ────────────────────────────────────────────────────────
#
# These are the defaults bound into function signatures when the module
# is imported. They are not global settings: assigning a new value to
# ``TRADING_DAYS_PER_YEAR`` after import changes nothing. Pass
# ``trading_days=`` per call instead (for example 365 for crypto).

EPSILON: float = 1e-15
TRADING_DAYS_PER_YEAR: int = 252
ANNUALIZATION_FACTOR: float = np.sqrt(TRADING_DAYS_PER_YEAR)

# Standard OHLCV column names (overridable per-call)
DEFAULT_COL_OPEN: str = "open"
DEFAULT_COL_HIGH: str = "high"
DEFAULT_COL_LOW: str = "low"
DEFAULT_COL_CLOSE: str = "close"
DEFAULT_COL_VOLUME: str = "volume"

# Type alias for anything that can carry price data
PriceInput = Union[pl.DataFrame, pl.Series, np.ndarray, list]


# ── Array extraction ─────────────────────────────────────────────────


def _is_pandas(data: Any, name: str) -> bool:
    cls = type(data)
    return cls.__module__.split(".")[0] == "pandas" and cls.__name__ == name


def is_frame(data: Any) -> bool:
    """``True`` for a polars or pandas ``DataFrame``."""
    return isinstance(data, pl.DataFrame) or _is_pandas(data, "DataFrame")


def _series_to_numpy(s: Any) -> np.ndarray:
    if isinstance(s, pl.Series):
        return s.to_numpy().astype(np.float64)
    # pandas Series: nullable dtypes carry pd.NA, which must become NaN.
    return np.asarray(s.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64)


def ensure_array(data: PriceInput, col: Optional[str] = None) -> np.ndarray:
    """Convert any supported input into a 1-D float64 numpy array.

    Args:
        data: polars/pandas DataFrame or Series, 1-D ndarray, or list.
        col: Column name to extract when *data* is a DataFrame.
             Defaults to ``"close"`` when *col* is ``None`` and
             *data* is a DataFrame.

    Raises:
        KeyError: the DataFrame has no such column.
        ValueError: the input is not one-dimensional (for example a 2-D
            array of several columns), which would otherwise be read as
            one long price series.
    """
    if is_frame(data):
        return _col_to_numpy(data, col or DEFAULT_COL_CLOSE)
    if isinstance(data, pl.Series) or _is_pandas(data, "Series"):
        return _series_to_numpy(data)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"expected a 1-D series of values, got an array of shape {arr.shape}; pass a "
            f"DataFrame (the '{col or DEFAULT_COL_CLOSE}' column is used) or a single series"
        )
    return arr


def _col_to_numpy(df: Any, col: str) -> np.ndarray:
    """Extract a column of a polars or pandas DataFrame as float64."""
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found. Available: {list(df.columns)}")
    return _series_to_numpy(df[col])


def require_frame(bars: Any, fn: str, cols: Tuple[str, ...]) -> Any:
    """Return *bars* when it is a DataFrame, else raise naming the columns needed."""
    if not is_frame(bars):
        who = f"{fn}()" if fn.isidentifier() else fn
        raise TypeError(
            f"{who} needs a polars or pandas DataFrame with columns {list(cols)}, "
            f"got {type(bars).__name__}"
        )
    return bars


def ensure_ohlcv(
    bars: Any,
    *,
    col_open: str = DEFAULT_COL_OPEN,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract OHLCV arrays from a polars or pandas DataFrame.

    Column names are fully configurable so users with non-standard
    schemas don't have to rename anything.
    """
    cols = (col_open, col_high, col_low, col_close, col_volume)
    require_frame(bars, "this indicator", cols)
    return tuple(_col_to_numpy(bars, c) for c in cols)  # type: ignore[return-value]


def ensure_ohlc(
    bars: Any,
    *,
    col_open: str = DEFAULT_COL_OPEN,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract OHLC arrays (no volume) from a polars or pandas DataFrame."""
    cols = (col_open, col_high, col_low, col_close)
    require_frame(bars, "this indicator", cols)
    return tuple(_col_to_numpy(bars, c) for c in cols)  # type: ignore[return-value]


def ensure_hlc(
    bars: Any,
    *,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract High / Low / Close arrays from a polars or pandas DataFrame."""
    cols = (col_high, col_low, col_close)
    require_frame(bars, "this indicator", cols)
    return tuple(_col_to_numpy(bars, c) for c in cols)  # type: ignore[return-value]


# ── Smoothing primitives (reused across indicators) ─────────────────


def wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing (equivalent to EMA with alpha = 1/period).

    Seeded with the mean of the first *period* valid values. A leading
    NaN warm-up (an indicator of an indicator) is skipped; a NaN after
    the seed makes every later value NaN rather than carrying an old one.
    Used by ATR and other Wilder-family indicators.
    """
    return ema_sma_seed(values, period, alpha=1.0 / period)


# ── Source price helpers ─────────────────────────────────────────────


def source_price(
    bars: PriceInput,
    source: str = "close",
    *,
    col_open: str = DEFAULT_COL_OPEN,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> np.ndarray:
    """Compute a derived price source from OHLC data.

    A derived source (anything but ``"close"``) needs a DataFrame with the
    OHLC columns; on a single series it raises instead of silently using
    that series.

    Supported *source* values:
        ``"close"``  — close price (default)
        ``"open"``   — open price
        ``"hl2"``    — (high + low) / 2
        ``"hlc3"``   — (high + low + close) / 3  (typical price)
        ``"ohlc4"``  — (open + high + low + close) / 4
    """
    if not is_frame(bars):
        if source != "close":
            raise TypeError(
                f"source='{source}' needs a DataFrame with OHLC columns; "
                f"got {type(bars).__name__}"
            )
        return ensure_array(bars)

    if source == "close":
        return _col_to_numpy(bars, col_close)
    if source == "open":
        return _col_to_numpy(bars, col_open)
    if source == "hl2":
        h = _col_to_numpy(bars, col_high)
        low = _col_to_numpy(bars, col_low)
        return (h + low) / 2.0
    if source == "hlc3":
        h = _col_to_numpy(bars, col_high)
        low = _col_to_numpy(bars, col_low)
        c = _col_to_numpy(bars, col_close)
        return (h + low + c) / 3.0
    if source == "ohlc4":
        o = _col_to_numpy(bars, col_open)
        h = _col_to_numpy(bars, col_high)
        low = _col_to_numpy(bars, col_low)
        c = _col_to_numpy(bars, col_close)
        return (o + h + low + c) / 4.0
    raise ValueError(f"Unknown source '{source}'. Use: close, open, hl2, hlc3, ohlc4")


# ── NaN-aware building blocks for chained indicators ────────────────
#
# Many indicators feed the output of one smoother into another. The
# helpers below skip the leading NaN warm-up of their input so chains
# line up exactly with reference implementations such as TA-Lib.


def first_valid_index(values: np.ndarray) -> int:
    """Index of the first non-NaN element (``len(values)`` if none)."""
    idx = np.flatnonzero(~np.isnan(values))
    return int(idx[0]) if idx.size else len(values)


def ema_sma_seed(
    values: np.ndarray,
    period: int,
    *,
    alpha: Optional[float] = None,
) -> np.ndarray:
    """EMA seeded with the SMA of the first *period* valid values.

    This is the TA-Lib (and TradingView ``ta.ema``) convention: the output
    is NaN until ``first_valid + period - 1``, where it equals the simple
    mean of the first *period* valid inputs, and follows
    ``y[t] = alpha * x[t] + (1 - alpha) * y[t-1]`` afterwards. Leading NaNs
    in *values* are skipped, so the helper can be chained.

    Args:
        values: Input array (leading NaNs allowed, no interior NaNs).
        period: Look-back used for the seed and the default alpha.
        alpha: Smoothing factor; defaults to ``2 / (period + 1)``.
    """
    n = len(values)
    out = np.full(n, np.nan)
    if period < 1:
        raise ValueError("period must be >= 1")
    start = first_valid_index(values)
    seed_idx = start + period - 1
    if seed_idx >= n:
        return out
    a = 2.0 / (period + 1) if alpha is None else float(alpha)
    prev = float(np.mean(values[start : seed_idx + 1]))
    out[seed_idx] = prev
    one_minus = 1.0 - a
    for i in range(seed_idx + 1, n):
        prev = a * values[i] + one_minus * prev
        out[i] = prev
    return out


def wilder_sma_seed(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing (alpha = 1/period) with an SMA seed, NaN-aware."""
    return ema_sma_seed(values, period, alpha=1.0 / period)


def _rolling_view(values: np.ndarray, period: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(values, period)


def rolling_apply(values: np.ndarray, period: int, reducer: str) -> np.ndarray:
    """Rolling ``sum`` / ``mean`` / ``max`` / ``min`` over a trailing window.

    The output at index ``t`` covers ``values[t - period + 1 : t + 1]`` and
    is NaN for ``t < period - 1``. NaNs inside a window propagate.
    """
    n = len(values)
    out = np.full(n, np.nan)
    if period < 1:
        raise ValueError("period must be >= 1")
    if n < period:
        return out
    win = _rolling_view(values, period)
    if reducer == "sum":
        out[period - 1 :] = win.sum(axis=1)
    elif reducer == "mean":
        out[period - 1 :] = win.mean(axis=1)
    elif reducer == "max":
        out[period - 1 :] = win.max(axis=1)
    elif reducer == "min":
        out[period - 1 :] = win.min(axis=1)
    else:
        raise ValueError(f"Unknown reducer '{reducer}'. Use: sum, mean, max, min")
    return out


def shift(values: np.ndarray, periods: int) -> np.ndarray:
    """Shift an array by *periods* (positive = lag), padding with NaN."""
    n = len(values)
    out = np.full(n, np.nan)
    if periods == 0:
        return values.astype(np.float64, copy=True)
    if abs(periods) >= n:
        return out
    if periods > 0:
        out[periods:] = values[:-periods]
    else:
        out[:periods] = values[-periods:]
    return out


def ratio(num: float, den: float, fill: float) -> float:
    """Scalar ``num / den``: NaN when either is NaN, *fill* only for a zero denominator.

    The guard ``x / d if d > EPSILON else fill`` sends a NaN denominator
    (a missing bar in the window) to *fill*, which reads as a real value.
    """
    if num != num or den != den:  # NaN in, NaN out
        return float("nan")
    return float(num / den) if abs(den) > EPSILON else fill


def safe_divide(num: np.ndarray, den: np.ndarray, fill: float = np.nan) -> np.ndarray:
    """Element-wise ``num / den`` with *fill* where ``|den| <= EPSILON``.

    NaN in either operand always yields NaN (warm-up is never filled).
    """
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    shape = np.broadcast(num, den).shape
    out = np.full(shape, fill, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        ok = np.abs(den) > EPSILON
    np.divide(num, den, out=out, where=ok)
    out[np.broadcast_to(np.isnan(num), shape) | np.broadcast_to(np.isnan(den), shape)] = np.nan
    return out
