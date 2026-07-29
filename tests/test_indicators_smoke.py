"""Smoke tests for cpz_quant.indicators: every family imports and computes."""

from __future__ import annotations

# Direct module imports: the package __init__ re-exports a `momentum`
# *function* that shadows the submodule of the same name.
import cpz_quant.indicators.momentum as momentum
import cpz_quant.indicators.statistical as statistical
import cpz_quant.indicators.trend as trend
import cpz_quant.indicators.volatility as volatility
import cpz_quant.indicators.volume as volume
import numpy as np


def _series(n: int = 200, seed: int = 3):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = rng.integers(1_000, 100_000, n).astype(float)
    return close, high, low, vol


def test_families_importable():
    assert momentum and trend and volatility and volume and statistical


def test_trend_sma_ema_shapes():
    close, *_ = _series()
    sma = trend.sma_series(close, 20)
    ema = trend.ema_series(close, 20)
    assert len(sma) == len(close)
    assert len(ema) == len(close)
    assert np.isfinite(sma[-1]) and np.isfinite(ema[-1])


def test_momentum_rsi_bounds():
    close, *_ = _series()
    rsi = momentum.rsi_series(close, 14)
    tail = rsi[-50:]
    tail = tail[np.isfinite(tail)]
    assert tail.size and ((tail >= 0.0) & (tail <= 100.0)).all()


def test_volatility_atr_positive():
    close, high, low, _ = _series()
    atr = volatility.atr_series(high, low, close, 14)
    assert np.isfinite(atr[-1]) and atr[-1] > 0
