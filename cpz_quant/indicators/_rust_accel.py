"""Rust acceleration for indicator computation.

Mirrors the pattern in ``cpz.risk._rust_accel``: try to import the
compiled extension, expose thin wrappers that return ``None`` when Rust
is unavailable so callers can fall back to Python/NumPy.

Build with:  ``cd rust && maturin develop --release``
"""

from __future__ import annotations

from typing import List, Optional, Tuple

try:
    import cpz_risk_rs

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def has_rust() -> bool:
    """``True`` when the compiled Rust extension is installed."""
    return RUST_AVAILABLE


# ── Trend ────────────────────────────────────────────────────────────

def fast_sma(close: List[float], period: int) -> Optional[List[float]]:
    if not RUST_AVAILABLE or not hasattr(cpz_risk_rs, "rolling_sma"):
        return None
    return cpz_risk_rs.rolling_sma(close, period)


def fast_ema(close: List[float], period: int) -> Optional[List[float]]:
    if not RUST_AVAILABLE or not hasattr(cpz_risk_rs, "ema"):
        return None
    return cpz_risk_rs.ema(close, period)


def fast_wma(close: List[float], period: int) -> Optional[List[float]]:
    if not RUST_AVAILABLE or not hasattr(cpz_risk_rs, "wma"):
        return None
    return cpz_risk_rs.wma(close, period)


def fast_kama(
    close: List[float], period: int, fast_sc: float, slow_sc: float,
) -> Optional[List[float]]:
    if not RUST_AVAILABLE or not hasattr(cpz_risk_rs, "kama"):
        return None
    return cpz_risk_rs.kama(close, period, fast_sc, slow_sc)


# ── Momentum ─────────────────────────────────────────────────────────

def fast_rsi(close: List[float], period: int) -> Optional[List[float]]:
    if not RUST_AVAILABLE or not hasattr(cpz_risk_rs, "rsi"):
        return None
    return cpz_risk_rs.rsi(close, period)


def fast_macd(
    close: List[float], fast: int, slow: int, signal: int,
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    if not RUST_AVAILABLE or not hasattr(cpz_risk_rs, "macd"):
        return None
    return cpz_risk_rs.macd(close, fast, slow, signal)


def fast_adx(
    high: List[float], low: List[float], close: List[float], period: int,
) -> Optional[List[float]]:
    if not RUST_AVAILABLE or not hasattr(cpz_risk_rs, "adx"):
        return None
    return cpz_risk_rs.adx(high, low, close, period)


# ── Volatility ───────────────────────────────────────────────────────

def fast_atr(
    high: List[float], low: List[float], close: List[float], period: int,
) -> Optional[List[float]]:
    if not RUST_AVAILABLE or not hasattr(cpz_risk_rs, "atr"):
        return None
    return cpz_risk_rs.atr(high, low, close, period)


def fast_bollinger(
    close: List[float], period: int, num_std: float,
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    if not RUST_AVAILABLE or not hasattr(cpz_risk_rs, "bollinger"):
        return None
    return cpz_risk_rs.bollinger(close, period, num_std)


def fast_garman_klass(
    open_: List[float],
    high: List[float],
    low: List[float],
    close: List[float],
    period: int,
) -> Optional[List[float]]:
    if not RUST_AVAILABLE or not hasattr(cpz_risk_rs, "garman_klass_vol"):
        return None
    return cpz_risk_rs.garman_klass_vol(open_, high, low, close, period)
