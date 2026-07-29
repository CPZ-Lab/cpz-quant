"""Shared utilities for indicator computation.

Centralises array extraction, constants, and common operations
so individual indicator modules stay focused on math.

As of cpz-ai 4.x the DataFrame layer is polars-only. Inputs accept
``pl.DataFrame``, ``pl.Series``, ``numpy.ndarray``, or plain ``list``.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import polars as pl

# ── Constants ────────────────────────────────────────────────────────

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


def ensure_array(data: PriceInput, col: Optional[str] = None) -> np.ndarray:
    """Convert any supported input into a float64 numpy array.

    Accepts polars DataFrames/Series, numpy arrays, and plain lists.

    Args:
        data: DataFrame, Series, ndarray, or list of floats.
        col: Column name to extract when *data* is a DataFrame.
             Defaults to ``"close"`` when *col* is ``None`` and
             *data* is a DataFrame.
    """
    if isinstance(data, pl.DataFrame):
        col = col or DEFAULT_COL_CLOSE
        if col not in data.columns:
            raise KeyError(f"Column '{col}' not found. Available: {data.columns}")
        return data[col].to_numpy().astype(np.float64)
    if isinstance(data, pl.Series):
        return data.to_numpy().astype(np.float64)
    return np.asarray(data, dtype=np.float64)


def _col_to_numpy(df: pl.DataFrame, col: str) -> np.ndarray:
    """Extract a column as float64 numpy array from a polars DataFrame."""
    return df[col].to_numpy().astype(np.float64)


def ensure_ohlcv(
    bars: pl.DataFrame,
    *,
    col_open: str = DEFAULT_COL_OPEN,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
    col_volume: str = DEFAULT_COL_VOLUME,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract OHLCV arrays from a polars DataFrame.

    Column names are fully configurable so users with non-standard
    schemas don't have to rename anything.
    """
    return (
        _col_to_numpy(bars, col_open),
        _col_to_numpy(bars, col_high),
        _col_to_numpy(bars, col_low),
        _col_to_numpy(bars, col_close),
        _col_to_numpy(bars, col_volume),
    )


def ensure_ohlc(
    bars: pl.DataFrame,
    *,
    col_open: str = DEFAULT_COL_OPEN,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract OHLC arrays (no volume) from a polars DataFrame."""
    return (
        _col_to_numpy(bars, col_open),
        _col_to_numpy(bars, col_high),
        _col_to_numpy(bars, col_low),
        _col_to_numpy(bars, col_close),
    )


def ensure_hlc(
    bars: pl.DataFrame,
    *,
    col_high: str = DEFAULT_COL_HIGH,
    col_low: str = DEFAULT_COL_LOW,
    col_close: str = DEFAULT_COL_CLOSE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract High / Low / Close arrays from a polars DataFrame."""
    return (
        _col_to_numpy(bars, col_high),
        _col_to_numpy(bars, col_low),
        _col_to_numpy(bars, col_close),
    )


# ── Smoothing primitives (reused across indicators) ─────────────────


def wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing (equivalent to EMA with alpha = 1/period).

    Used by RSI, ATR, ADX, and other Wilder-family indicators.
    """
    n = len(values)
    out = np.full(n, np.nan)
    if n < period:
        return out
    out[period - 1] = np.mean(values[:period])
    alpha = 1.0 / period
    for i in range(period, n):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


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

    Accepts polars DataFrames, numpy arrays, or plain lists.

    Supported *source* values:
        ``"close"``  — close price (default)
        ``"open"``   — open price
        ``"hl2"``    — (high + low) / 2
        ``"hlc3"``   — (high + low + close) / 3  (typical price)
        ``"ohlc4"``  — (open + high + low + close) / 4
    """
    if not isinstance(bars, pl.DataFrame):
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
