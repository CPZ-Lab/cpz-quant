"""Statistical indicators: z-score, rolling correlation, rolling beta, Hurst exponent,
linear regression (slope, intercept, R^2, forecast, end value, angle), Kaufman efficiency
ratio, rolling Sharpe and Sortino.

All functions operate on raw numpy arrays and return numpy arrays.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from ._helpers import (
    EPSILON,
    TRADING_DAYS_PER_YEAR,
    _rolling_view,
    rolling_apply,
    safe_divide,
)

# ── Z-Score ──────────────────────────────────────────────────────────

def zscore_series(
    values: np.ndarray,
    period: int = 60,
    *,
    ddof: int = 1,
) -> np.ndarray:
    """Rolling z-score: ``(x - rolling_mean) / rolling_std``.

    Args:
        values: Input array.
        period: Rolling window size.
        ddof: Degrees of freedom for std (default 1 = sample std).
    """
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        mean = np.mean(window)
        std = np.std(window, ddof=ddof)
        out[i] = (values[i] - mean) / std if std > EPSILON else 0.0
    return out


# ── Rolling Correlation ─────────────────────────────────────────────

def rolling_corr_series(
    series_a: np.ndarray,
    series_b: np.ndarray,
    period: int = 60,
) -> np.ndarray:
    """Rolling Pearson correlation between two series.

    Args:
        series_a, series_b: Input arrays (must be same length).
        period: Rolling window size.
    """
    n = min(len(series_a), len(series_b))
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        a = series_a[i - period + 1 : i + 1]
        b = series_b[i - period + 1 : i + 1]
        std_a = np.std(a, ddof=1)
        std_b = np.std(b, ddof=1)
        if std_a > EPSILON and std_b > EPSILON:
            out[i] = np.corrcoef(a, b)[0, 1]
        else:
            out[i] = 0.0
    return out


# ── Rolling Beta ─────────────────────────────────────────────────────

def rolling_beta_series(
    asset_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    period: int = 60,
) -> np.ndarray:
    """Rolling OLS beta (slope of asset vs benchmark returns).

    Args:
        asset_returns: Asset return array.
        benchmark_returns: Benchmark return array.
        period: Rolling window size.
    """
    n = min(len(asset_returns), len(benchmark_returns))
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        y = asset_returns[i - period + 1 : i + 1]
        x = benchmark_returns[i - period + 1 : i + 1]
        var_x = np.var(x, ddof=1)
        if var_x > EPSILON:
            cov_xy = np.cov(x, y, ddof=1)[0, 1]
            out[i] = cov_xy / var_x
        else:
            out[i] = 0.0
    return out


# ── Hurst Exponent ───────────────────────────────────────────────────

def _expected_rescaled_range(n: int) -> float:
    """Expected R/S of ``n`` i.i.d. Gaussian values.

    Anis and Lloyd (1976) with Peters' (1994) small-sample factor
    ``(n - 0.5) / n``::

        E[R/S]_n = (n - 0.5)/n * Gamma((n-1)/2) / (sqrt(pi) Gamma(n/2))
                   * sum_{i=1}^{n-1} sqrt((n - i) / i)

    The gamma ratio is taken through ``lgamma`` so it does not overflow.
    """
    i = np.arange(1, n, dtype=np.float64)
    total = float(np.sum(np.sqrt((n - i) / i)))
    gamma_ratio = math.exp(math.lgamma((n - 1) / 2.0) - math.lgamma(n / 2.0)) / math.sqrt(math.pi)
    return (n - 0.5) / n * gamma_ratio * total


def hurst_series(
    close: np.ndarray,
    max_lag: int = 100,
    *,
    min_lag: int = 10,
) -> float:
    """Hurst exponent by bias-corrected rescaled range (Anis-Lloyd-Peters R/S).

    H < 0.5 → mean-reverting
    H = 0.5 → random walk
    H > 0.5 → trending

    Plain R/S on short blocks overstates H: a random walk reads about
    0.63 instead of 0.5. This estimator subtracts the expected R/S of
    i.i.d. noise at each block size (Anis and Lloyd 1976, with Peters 1994)
    and reports ``0.5 + slope`` of the excess, the corrected R/S of Weron
    (2002, Physica A 312). On simulated random walks of 300 to 1,500 bars
    it averages within about 0.01 of 0.5; R/S remains a noisy estimator
    (standard error roughly 0.03 to 0.07 at those lengths), so read small
    departures from 0.5 with care.

    Args:
        close: Price array (positive, finite).
        max_lag: Largest block size, capped at half the number of returns.
        min_lag: Smallest block size (default 10; Peters recommends at
            least 10, below which the expected-R/S formula is unreliable).

    Returns:
        Scalar Hurst exponent, or NaN when fewer than two block sizes fit
        between *min_lag* and *max_lag* (too little data).

    Raises:
        ValueError: on non-finite or non-positive prices, or ``min_lag < 2``.
    """
    close = np.asarray(close, dtype=np.float64)
    if min_lag < 2:
        raise ValueError(f"min_lag must be >= 2, got {min_lag}")
    if close.size and (not np.all(np.isfinite(close)) or np.any(close <= 0)):
        raise ValueError("hurst() needs positive, finite prices; drop or fill missing bars first")
    returns = np.diff(np.log(close))
    n = len(returns)
    max_lag = min(max_lag, n // 2)

    lag_values = []
    excess = []
    for lag in range(min_lag, max_lag + 1):
        n_blocks = n // lag
        ratios = []
        for b in range(n_blocks):
            block = returns[b * lag : (b + 1) * lag]
            deviations = np.cumsum(block - np.mean(block))
            s = np.std(block, ddof=1)
            if s > EPSILON:
                ratios.append((np.max(deviations) - np.min(deviations)) / s)
        if ratios:
            excess.append(math.log(float(np.mean(ratios))) - math.log(_expected_rescaled_range(lag)))
            lag_values.append(math.log(lag))

    if len(lag_values) < 2:
        return float("nan")

    slope = np.polyfit(lag_values, excess, 1)[0]
    return float(0.5 + slope)


# ── Linear Regression ────────────────────────────────────────────────

def linear_reg_series(
    close: np.ndarray,
    period: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rolling OLS linear regression on close prices.

    Args:
        close: Price array.
        period: Rolling window size.

    Returns:
        ``(slope, r_squared, intercept, forecast)`` arrays where
        *forecast* is the next-bar predicted value.
    """
    n = len(close)
    slope = np.full(n, np.nan)
    r_sq = np.full(n, np.nan)
    intercept = np.full(n, np.nan)
    forecast = np.full(n, np.nan)

    x = np.arange(period, dtype=np.float64)
    x_mean = np.mean(x)
    ss_x = np.sum((x - x_mean) ** 2)

    for i in range(period - 1, n):
        y = close[i - period + 1 : i + 1]
        y_mean = np.mean(y)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_y = np.sum((y - y_mean) ** 2)

        b = ss_xy / ss_x if ss_x > EPSILON else 0.0
        a = y_mean - b * x_mean
        slope[i] = b
        intercept[i] = a
        r_sq[i] = (ss_xy ** 2) / (ss_x * ss_y) if ss_y > EPSILON else 0.0
        forecast[i] = a + b * period

    return slope, r_sq, intercept, forecast


# ═══════════════════════════════════════════════════════════════════
#  Extended statistical family
# ═══════════════════════════════════════════════════════════════════


# ── Linear regression value / angle ──────────────────────────────────

def linear_reg_value_series(close: np.ndarray, period: int = 14) -> np.ndarray:
    """End-point of the rolling least-squares line (TA-Lib ``LINEARREG``)."""
    slope, _, intercept, _ = linear_reg_series(close, period)
    return intercept + slope * (period - 1)


def linear_reg_angle_series(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Angle of the rolling regression slope in degrees (TA-Lib ``LINEARREG_ANGLE``).

    ``atan(slope) * 180 / pi`` with the slope in price units per bar, so the
    value depends on the price scale.
    """
    slope = linear_reg_series(close, period)[0]
    return np.degrees(np.arctan(slope))


# ── Kaufman Efficiency Ratio ─────────────────────────────────────────

def efficiency_ratio_series(close: np.ndarray, period: int = 10) -> np.ndarray:
    """Kaufman Efficiency Ratio (0..1).

    ``|close - close[t - period]| / sum(|close[i] - close[i-1]|)`` over the
    last *period* changes; a zero path length gives NaN.
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    direction = np.abs(close[period:] - close[:-period])
    path = rolling_apply(np.abs(np.diff(close)), period, "sum")[period - 1 :]
    out[period:] = safe_divide(direction, path)
    return out


# ── Rolling Sharpe / Sortino ─────────────────────────────────────────

def rolling_sharpe_series(
    returns: np.ndarray,
    period: int = 63,
    *,
    risk_free: float = 0.0,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Rolling Sharpe ratio of periodic returns.

    ``mean(r - rf) / std(r - rf, ddof=1)`` over *period* returns, times
    ``sqrt(trading_days)`` when annualised. *risk_free* is a per-period
    rate. A zero standard deviation gives NaN.
    """
    excess = returns - risk_free
    n = len(excess)
    out = np.full(n, np.nan)
    if period < 2 or n < period:
        return out
    win = _rolling_view(excess, period)
    out[period - 1 :] = safe_divide(win.mean(axis=1), win.std(axis=1, ddof=1))
    return out * np.sqrt(trading_days) if annualize else out


def rolling_sortino_series(
    returns: np.ndarray,
    period: int = 63,
    *,
    target: float = 0.0,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Rolling Sortino ratio of periodic returns.

    ``mean(r - target) / DD`` with the target downside deviation
    ``DD = sqrt(mean(min(r - target, 0)^2))`` over all *period* returns
    (Sortino and Price, 1994), times ``sqrt(trading_days)`` when annualised.
    *target* is a per-period rate. No downside observations gives NaN.
    """
    excess = returns - target
    n = len(excess)
    out = np.full(n, np.nan)
    if period < 1 or n < period:
        return out
    win = _rolling_view(excess, period)
    downside = np.sqrt(np.mean(np.minimum(win, 0.0) ** 2, axis=1))
    out[period - 1 :] = safe_divide(win.mean(axis=1), downside)
    return out * np.sqrt(trading_days) if annualize else out
