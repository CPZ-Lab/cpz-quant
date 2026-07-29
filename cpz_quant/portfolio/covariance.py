"""Covariance estimation: sample, Ledoit-Wolf, OAS, EWMA, factor-model, Marchenko-Pastur denoised, Gerber.

All functions are pure: data in, results out. No DB access, no API calls.
Requires numpy (core dep) and scipy (core dep).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from pydantic import BaseModel, Field

TRADING_DAYS: int = 252
EPSILON: float = 1e-15


class LedoitWolfResult(BaseModel):
    covariance: List[List[float]] = Field(default_factory=list)
    shrinkage_intensity: float = 0.0
    target: str = "single_factor"


class FactorCovResult(BaseModel):
    covariance: List[List[float]] = Field(default_factory=list)
    factor_loadings: List[List[float]] = Field(default_factory=list)
    factor_covariance: List[List[float]] = Field(default_factory=list)
    specific_variance: List[float] = Field(default_factory=list)
    n_factors: int = 0


def sample_cov(
    returns: Dict[str, List[float]],
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
    ddof: int = 1,
) -> np.ndarray:
    """Annualised sample covariance matrix.

    Args:
        returns: ``{asset_id: [daily_returns]}`` mapping.
        annualize: Multiply by *trading_days* (default ``True``).
        trading_days: Trading days per year.
        ddof: Degrees of freedom for np.cov.
    """
    ids = list(returns.keys())
    min_len = min(len(returns[i]) for i in ids)
    R = np.column_stack([np.array(returns[i][-min_len:], dtype=np.float64) for i in ids])
    cov = np.cov(R.T, ddof=ddof)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    if annualize:
        cov *= trading_days
    return cov


def ledoit_wolf(
    returns: Dict[str, List[float]],
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
    target: str = "constant_correlation",
) -> LedoitWolfResult:
    """Ledoit-Wolf (2004) linear shrinkage.

    Optimal shrinkage intensity minimises expected Frobenius loss between
    the shrunk estimator and the true covariance.

    Args:
        returns: ``{asset_id: [daily_returns]}`` mapping.
        annualize: Annualise the result.
        trading_days: Trading days per year.
        target: Shrinkage target — ``"constant_correlation"`` (default),
                ``"identity"``, or ``"single_factor"``.
    """
    ids = list(returns.keys())
    min_len = min(len(returns[i]) for i in ids)
    R = np.column_stack([np.array(returns[i][-min_len:], dtype=np.float64) for i in ids])
    t, n = R.shape

    sample = np.cov(R.T, ddof=1)
    if sample.ndim == 0:
        sample = np.array([[float(sample)]])

    if target == "identity":
        mu = np.trace(sample) / n
        F = mu * np.eye(n)
    elif target == "single_factor":
        var = np.diag(sample)
        sqd = np.sqrt(var)
        corr = sample / np.outer(sqd, sqd + EPSILON)
        avg_corr = (np.sum(corr) - n) / (n * (n - 1))
        F = avg_corr * np.outer(sqd, sqd)
        np.fill_diagonal(F, var)
    elif target == "constant_correlation":
        var = np.diag(sample)
        sqd = np.sqrt(var)
        corr = sample / np.outer(sqd, sqd + EPSILON)
        avg_corr = (np.sum(corr) - n) / (n * (n - 1))
        F = avg_corr * np.outer(sqd, sqd)
        np.fill_diagonal(F, var)
    else:
        raise ValueError(f"Unknown target '{target}'. Use: constant_correlation, identity, single_factor")

    X = R - R.mean(axis=0)
    S2 = (X.T @ X) / t

    delta = sample - F
    d2 = np.sum(delta ** 2) / n

    b_bar2 = 0.0
    for k in range(t):
        xk = X[k : k + 1, :]
        mk = xk.T @ xk - S2
        b_bar2 += np.sum(mk ** 2) / n
    b_bar2 /= t ** 2

    b2 = min(b_bar2, d2)
    alpha = b2 / max(d2, EPSILON)
    alpha = max(0.0, min(1.0, alpha))

    shrunk = alpha * F + (1.0 - alpha) * sample
    if annualize:
        shrunk *= trading_days

    return LedoitWolfResult(
        covariance=shrunk.tolist(),
        shrinkage_intensity=round(alpha, 6),
        target=target,
    )


def oracle_approximating(
    returns: Dict[str, List[float]],
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
) -> np.ndarray:
    """Oracle Approximating Shrinkage (Chen, Wiesel, Eldar, Hero 2010).

    Non-linear shrinkage of eigenvalues — better than Ledoit-Wolf for
    large N/T ratios.
    """
    ids = list(returns.keys())
    min_len = min(len(returns[i]) for i in ids)
    R = np.column_stack([np.array(returns[i][-min_len:], dtype=np.float64) for i in ids])
    t, n = R.shape

    sample = np.cov(R.T, ddof=1)
    if sample.ndim == 0:
        sample = np.array([[float(sample)]])

    eigvals, eigvecs = np.linalg.eigh(sample)
    eigvals = np.maximum(eigvals, 0)

    gamma = n / t
    mu = np.trace(sample) / n

    shrunk_eigvals = np.zeros_like(eigvals)
    for i, lam in enumerate(eigvals):
        shrunk_eigvals[i] = lam / max(1.0 + gamma * (lam - mu) / max(mu, EPSILON), EPSILON)

    shrunk_eigvals = np.maximum(shrunk_eigvals, EPSILON)
    result = eigvecs @ np.diag(shrunk_eigvals) @ eigvecs.T
    result = (result + result.T) / 2.0

    if annualize:
        result *= trading_days
    return result


def ewma_cov(
    returns: Dict[str, List[float]],
    *,
    halflife: int = 60,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
    min_periods: int = 10,
) -> np.ndarray:
    """Exponentially weighted covariance matrix (RiskMetrics-style).

    Adapts faster to regime changes than sample covariance.

    Args:
        returns: ``{asset_id: [daily_returns]}`` mapping.
        halflife: Decay half-life in trading days.
        annualize: Annualise the result.
        trading_days: Trading days per year.
        min_periods: Minimum observations before computing.
    """
    ids = list(returns.keys())
    min_len = min(len(returns[i]) for i in ids)
    R = np.column_stack([np.array(returns[i][-min_len:], dtype=np.float64) for i in ids])
    t, n = R.shape

    lam = 1.0 - np.log(2.0) / halflife
    means = R.mean(axis=0)
    X = R - means

    cov = np.zeros((n, n))
    for k in range(t):
        cov = lam * cov + (1.0 - lam) * np.outer(X[k], X[k])

    if annualize:
        cov *= trading_days
    return cov


def factor_model_cov(
    returns: Dict[str, List[float]],
    *,
    n_factors: int = 5,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
) -> FactorCovResult:
    """Factor-model covariance: ``B @ F @ B' + D``.

    Uses PCA to extract statistical factors. Reduces estimation error
    for large universes.

    Args:
        returns: ``{asset_id: [daily_returns]}`` mapping.
        n_factors: Number of PCA factors to extract.
        annualize: Annualise the result.
        trading_days: Trading days per year.
    """
    ids = list(returns.keys())
    min_len = min(len(returns[i]) for i in ids)
    R = np.column_stack([np.array(returns[i][-min_len:], dtype=np.float64) for i in ids])
    t, n = R.shape
    n_factors = min(n_factors, n, t)

    X = R - R.mean(axis=0)
    cov_sample = X.T @ X / (t - 1)

    eigvals, eigvecs = np.linalg.eigh(cov_sample)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    B = eigvecs[:, :n_factors]
    factor_returns = X @ B
    F = np.cov(factor_returns.T, ddof=1)
    if F.ndim == 0:
        F = np.array([[float(F)]])

    residuals = X - factor_returns @ B.T
    D = np.diag(np.var(residuals, axis=0, ddof=1))

    full_cov = B @ F @ B.T + D
    scale = trading_days if annualize else 1.0

    return FactorCovResult(
        covariance=(full_cov * scale).tolist(),
        factor_loadings=B.tolist(),
        factor_covariance=(F * scale).tolist(),
        specific_variance=(np.diag(D) * scale).tolist(),
        n_factors=n_factors,
    )


def denoise_mp(
    returns: Dict[str, List[float]],
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
    bandwidth: float = 0.01,
) -> np.ndarray:
    """Marchenko-Pastur denoised covariance (De Prado 2020).

    Removes noise eigenvalues below the MP threshold and replaces
    them with their average, preserving the trace.

    Args:
        returns: ``{asset_id: [daily_returns]}`` mapping.
        annualize: Annualise the result.
        trading_days: Trading days per year.
        bandwidth: Kernel bandwidth for fitting (not used in basic MP).
    """
    ids = list(returns.keys())
    min_len = min(len(returns[i]) for i in ids)
    R = np.column_stack([np.array(returns[i][-min_len:], dtype=np.float64) for i in ids])
    t, n = R.shape

    corr = np.corrcoef(R.T)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 0)

    q = t / n
    lambda_plus = (1.0 + 1.0 / np.sqrt(q)) ** 2
    lambda_minus = (1.0 - 1.0 / np.sqrt(q)) ** 2

    noise_mask = (eigvals >= lambda_minus) & (eigvals <= lambda_plus)

    if np.any(noise_mask):
        noise_avg = np.mean(eigvals[noise_mask])
        eigvals[noise_mask] = noise_avg

    trace_original = n
    trace_current = np.sum(eigvals)
    if trace_current > EPSILON:
        eigvals *= trace_original / trace_current

    denoised_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    denoised_corr = (denoised_corr + denoised_corr.T) / 2.0

    sample = np.cov(R.T, ddof=1)
    if sample.ndim == 0:
        sample = np.array([[float(sample)]])
    vols = np.sqrt(np.diag(sample))
    result = denoised_corr * np.outer(vols, vols)

    if annualize:
        result *= trading_days
    return result


def gerber_cov(
    returns: Dict[str, List[float]],
    *,
    threshold: float = 0.5,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
) -> np.ndarray:
    """Gerber Statistic covariance (2021).

    Measures co-movement via co-exceedance counts, making it robust
    to outliers that contaminate Pearson correlation.

    Args:
        returns: ``{asset_id: [daily_returns]}`` mapping.
        threshold: Exceedance threshold in units of standard deviation.
        annualize: Annualise the result.
        trading_days: Trading days per year.
    """
    ids = list(returns.keys())
    min_len = min(len(returns[i]) for i in ids)
    R = np.column_stack([np.array(returns[i][-min_len:], dtype=np.float64) for i in ids])
    t, n = R.shape

    stds = np.std(R, axis=0, ddof=1)
    thresholds = threshold * stds

    concordant = np.zeros((n, n))
    discordant = np.zeros((n, n))
    neutral = np.zeros((n, n))

    for k in range(t):
        for i in range(n):
            for j in range(i, n):
                exc_i = R[k, i] > thresholds[i]
                exc_j = R[k, j] > thresholds[j]
                fall_i = R[k, i] < -thresholds[i]
                fall_j = R[k, j] < -thresholds[j]

                if (exc_i and exc_j) or (fall_i and fall_j):
                    concordant[i, j] += 1
                    concordant[j, i] += 1
                elif (exc_i and fall_j) or (fall_i and exc_j):
                    discordant[i, j] += 1
                    discordant[j, i] += 1
                else:
                    neutral[i, j] += 1
                    neutral[j, i] += 1

    total = concordant + discordant
    gerber_corr = np.where(
        total > EPSILON,
        (concordant - discordant) / total,
        0.0,
    )
    np.fill_diagonal(gerber_corr, 1.0)

    result = gerber_corr * np.outer(stds, stds)
    if annualize:
        result *= trading_days
    return result


def detone_cov(
    returns: Dict[str, List[float]],
    *,
    n_detones: int = 1,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
) -> np.ndarray:
    """Detoned covariance (Lopez de Prado 2020).

    Removes the top ``n_detones`` eigen-components (the market / systematic mode)
    from the correlation matrix, then rescales to unit diagonal. Detoning
    sharpens the residual (idiosyncratic) structure — useful before clustering
    allocators (HRP/HERC) where the dominant market mode otherwise swamps the
    dendrogram.

    Args:
        returns: ``{asset_id: [daily_returns]}`` mapping.
        n_detones: number of leading eigen-components to remove (default 1).
    """
    ids = list(returns.keys())
    min_len = min(len(returns[i]) for i in ids)
    R = np.column_stack([np.array(returns[i][-min_len:], dtype=np.float64) for i in ids])
    n = R.shape[1]
    k = int(max(0, min(n_detones, n - 1)))

    corr = np.corrcoef(R.T)
    if corr.ndim == 0:
        corr = np.array([[1.0]])
    eigvals, eigvecs = np.linalg.eigh(corr)  # ascending
    eigvals = np.maximum(eigvals, 0.0)

    if k > 0:
        # Zero out the k largest (market) eigenvalues, reconstruct, rescale.
        detoned = corr - (eigvecs[:, n - k:] @ np.diag(eigvals[n - k:]) @ eigvecs[:, n - k:].T)
        d = np.sqrt(np.clip(np.diag(detoned), EPSILON, None))
        detoned = detoned / np.outer(d, d)
        np.fill_diagonal(detoned, 1.0)
    else:
        detoned = corr  # type: ignore[assignment]

    sample = np.cov(R.T, ddof=1)
    if sample.ndim == 0:
        sample = np.array([[float(sample)]])
    vols = np.sqrt(np.diag(sample))
    result = detoned * np.outer(vols, vols)
    if annualize:
        result *= trading_days
    return result
