"""Covariance estimation: sample, Ledoit-Wolf, OAS, EWMA, factor-model, Marchenko-Pastur denoised, Gerber.

All functions are pure: data in, results out. No DB access, no API calls.
Requires numpy (core dep) and scipy (core dep).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, Field

from cpz_quant.frames import frame_friendly

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


def _returns_matrix(
    returns: Dict[str, List[float]], label: str, *, min_obs: int = 2
) -> Tuple[List[str], np.ndarray]:
    """Stack the common (most recent) window of each series into a (T, N) matrix.

    Raises instead of estimating from missing data: one NaN in a single series
    would otherwise turn most of the matrix into NaN, or (for rank and
    threshold statistics) be counted as an observation.
    """
    ids = list(returns.keys())
    if not ids:
        raise ValueError(f"{label}: returns must contain at least one asset")
    arrays = [np.asarray(returns[i], dtype=np.float64) for i in ids]
    for aid, values in zip(ids, arrays):
        if values.ndim != 1:
            raise ValueError(f"{label}: returns for {aid!r} must be a 1D series")
    min_len = min(len(values) for values in arrays)
    if min_len < min_obs:
        raise ValueError(
            f"{label}: needs at least {min_obs} observations per asset, got {min_len}"
        )
    R = np.column_stack([values[-min_len:] for values in arrays])
    bad = ~np.isfinite(R)
    if bad.any():
        row, col = (int(v[0]) for v in np.nonzero(bad))
        raise ValueError(
            f"{label}: returns contain {int(bad.sum())} missing or non-finite value(s); "
            f"the first is {ids[col]!r} at row {row} of the {min_len}-row window. "
            "Drop or fill them before estimating (for example frame.dropna())."
        )
    return ids, R


@frame_friendly
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
    _, R = _returns_matrix(returns, "sample_cov", min_obs=ddof + 1)
    cov = np.cov(R.T, ddof=ddof)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    if annualize:
        cov *= trading_days
    return cov


def _lw_identity(X: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ledoit & Wolf (2004, JMVA) shrinkage toward ``mean variance * I``.

    Same estimator as ``sklearn.covariance.ledoit_wolf`` (centred data).
    """
    t, n = X.shape
    mu = float(np.trace(S)) / n
    X2 = X ** 2
    beta_ = float(np.sum(X2.T @ X2))
    delta_ = float(np.sum((X.T @ X) ** 2)) / t ** 2
    beta = (beta_ / t - delta_) / (n * t)
    delta = (delta_ - 2.0 * mu * float(np.trace(S)) + n * mu ** 2) / n
    beta = min(beta, delta)
    shrinkage = 0.0 if beta == 0 else beta / delta
    return mu * np.eye(n), shrinkage


def _lw_constant_correlation(X: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ledoit & Wolf (2004), "Honey, I shrunk the sample covariance matrix".

    Transcribes the authors' published ``covCor.m``: shrinkage
    ``max(0, min(1, (pi - rho) / gamma / T))`` toward the constant-correlation
    matrix, where ``rho`` carries the covariance between the target and the
    sample (the term the identity-target formula does not have).
    """
    t, n = X.shape
    var = np.diag(S)
    sd = np.sqrt(var)
    r_bar = (np.sum(S / np.outer(sd, sd)) - n) / (n * (n - 1))
    F = r_bar * np.outer(sd, sd)
    np.fill_diagonal(F, var)
    y = X ** 2
    phi_mat = y.T @ y / t - 2.0 * (X.T @ X) * S / t + S ** 2
    phi = float(np.sum(phi_mat))
    term1 = (X ** 3).T @ X / t
    help_ = X.T @ X / t
    term2 = np.diag(help_)[:, None] * S
    term3 = help_ * var[:, None]
    term4 = var[:, None] * S
    theta = term1 - term2 - term3 + term4
    np.fill_diagonal(theta, 0.0)
    rho = float(np.sum(np.diag(phi_mat)) + r_bar * np.sum(np.outer(1.0 / sd, sd) * theta))
    gamma = float(np.linalg.norm(S - F, "fro") ** 2)
    if gamma == 0.0:
        return F, 0.0  # the sample already equals the target
    kappa = (phi - rho) / gamma
    return F, max(0.0, min(1.0, kappa / t))


def _lw_single_factor(X: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ledoit & Wolf (2003), "Improved estimation of the covariance matrix of
    stock returns with an application to portfolio selection".

    Transcribes the authors' published ``covMarket.m``: the target is Sharpe's
    single-index model with the equal-weighted cross-sectional mean as the
    market, ``F = beta beta' var_m`` with the sample variances on the diagonal.
    """
    t, n = X.shape
    xmkt = X.mean(axis=1)
    covmkt = X.T @ xmkt / t
    varmkt = float(xmkt @ xmkt) / t
    if varmkt == 0.0:
        raise ValueError("ledoit_wolf: the equal-weighted market series has zero variance")
    F = np.outer(covmkt, covmkt) / varmkt
    np.fill_diagonal(F, np.diag(S))
    c = float(np.linalg.norm(S - F, "fro") ** 2)
    if c == 0.0:
        return F, 0.0  # the sample already equals the target
    y = X ** 2
    p = float(np.sum(y.T @ y)) / t - float(np.sum(S ** 2))
    rdiag = float(np.sum(y ** 2)) / t - float(np.sum(np.diag(S) ** 2))
    z = X * xmkt[:, None]
    v1 = y.T @ z / t - covmkt[:, None] * S
    roff1 = (float(np.sum(v1 * covmkt[None, :])) - float(np.sum(np.diag(v1) * covmkt))) / varmkt
    v3 = z.T @ z / t - varmkt * S
    roff3 = (
        float(np.sum(v3 * np.outer(covmkt, covmkt)))
        - float(np.sum(np.diag(v3) * covmkt ** 2))
    ) / varmkt ** 2
    r = rdiag + 2.0 * roff1 - roff3
    k = (p - r) / c
    return F, max(0.0, min(1.0, k / t))


_LW_TARGETS = {
    "constant_correlation": _lw_constant_correlation,
    "identity": _lw_identity,
    "single_factor": _lw_single_factor,
}


@frame_friendly
def ledoit_wolf(
    returns: Dict[str, List[float]],
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
    target: str = "constant_correlation",
) -> LedoitWolfResult:
    """Ledoit-Wolf linear shrinkage with the published optimal intensity per target.

    ``shrunk = delta * F + (1 - delta) * S`` where ``S`` is the maximum-likelihood
    sample covariance of the demeaned returns (divided by T, as in the papers)
    and ``delta`` is each paper's consistent estimator of the optimal intensity:

    - ``"constant_correlation"`` (default): Ledoit & Wolf (2004), "Honey, I
      shrunk the sample covariance matrix", the authors' ``covCor.m``.
    - ``"single_factor"``: Ledoit & Wolf (2003), Sharpe single-index target,
      the authors' ``covMarket.m``.
    - ``"identity"``: Ledoit & Wolf (2004, JMVA), "A well-conditioned estimator
      for large-dimensional covariance matrices"; equals
      ``sklearn.covariance.ledoit_wolf``.

    The intensity is invariant to the units of the returns (decimal or percent).

    Args:
        returns: ``{asset_id: [daily_returns]}`` mapping.
        annualize: Annualise the result.
        trading_days: Trading days per year.
        target: Shrinkage target, one of the three above.
    """
    estimator = _LW_TARGETS.get(target)
    if estimator is None:
        raise ValueError(
            f"Unknown target '{target}'. Use: constant_correlation, identity, single_factor"
        )
    _, R = _returns_matrix(returns, "ledoit_wolf")
    t, n = R.shape
    X = R - R.mean(axis=0)
    S = X.T @ X / t
    if np.any(np.diag(S) <= 0.0):
        raise ValueError("ledoit_wolf: an asset has zero return variance")
    if n == 1:
        F, delta = S.copy(), 0.0  # every target equals the sample variance
    else:
        F, delta = estimator(X, S)

    shrunk = delta * F + (1.0 - delta) * S
    if annualize:
        shrunk = shrunk * trading_days

    return LedoitWolfResult(
        covariance=shrunk.tolist(),
        shrinkage_intensity=round(delta, 6),
        target=target,
    )


@frame_friendly
def oracle_approximating(
    returns: Dict[str, List[float]],
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
) -> np.ndarray:
    """Oracle Approximating Shrinkage (Chen, Wiesel, Eldar, Hero 2010).

    Linear shrinkage of the maximum-likelihood sample covariance ``S`` toward
    ``(tr(S) / N) * I`` with the OAS intensity of the paper's Eq. 23, in the
    form scikit-learn implements (the ``2/N`` terms dropped), so the result
    equals ``sklearn.covariance.OAS``. Suited to many assets relative to the
    number of observations.
    """
    _, R = _returns_matrix(returns, "oracle_approximating")
    t, n = R.shape
    X = R - R.mean(axis=0)
    S = X.T @ X / t
    if n == 1:
        result = S
    else:
        mu = float(np.trace(S)) / n
        alpha = float(np.mean(S ** 2))
        num = alpha + mu ** 2
        den = (t + 1.0) * (alpha - mu ** 2 / n)
        shrinkage = 1.0 if den == 0 else min(num / den, 1.0)
        result = (1.0 - shrinkage) * S + shrinkage * mu * np.eye(n)

    if annualize:
        result = result * trading_days
    return result


@frame_friendly
def ewma_cov(
    returns: Dict[str, List[float]],
    *,
    halflife: float = 60,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
    min_periods: int = 10,
) -> np.ndarray:
    """Exponentially weighted covariance matrix.

    Observation ``k`` periods before the latest gets weight ``0.5 ** (k / halflife)``;
    the weights are normalised to sum to one and the series are centred on
    their weighted means. This equals ``numpy.cov(R.T, aweights=w, ddof=0)`` and
    the last row of ``pandas.DataFrame.ewm(halflife=..., adjust=True).cov(bias=True)``.
    Without the normalisation a short history understates risk (60 days at a
    60-day half-life carry only about half the weight).

    Args:
        returns: ``{asset_id: [daily_returns]}`` mapping.
        halflife: Decay half-life in trading days.
        annualize: Annualise the result.
        trading_days: Trading days per year.
        min_periods: Minimum observations required; fewer raises ``ValueError``.
    """
    if not np.isfinite(halflife) or halflife <= 0:
        raise ValueError(f"ewma_cov: halflife must be positive, got {halflife}")
    if min_periods < 1:
        raise ValueError(f"ewma_cov: min_periods must be >= 1, got {min_periods}")
    _, R = _returns_matrix(returns, "ewma_cov", min_obs=max(int(min_periods), 2))
    t = R.shape[0]

    decay = 0.5 ** (1.0 / halflife)
    weights = decay ** np.arange(t - 1, -1, -1, dtype=np.float64)
    weights /= weights.sum()
    X = R - weights @ R
    cov = (X * weights[:, None]).T @ X

    if annualize:
        cov = cov * trading_days
    return cov


@frame_friendly
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
    _, R = _returns_matrix(returns, "factor_model_cov")
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


def _mp_pdf(var: float, q: float, pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """Marchenko-Pastur density of a correlation matrix's eigenvalues, q = T/N."""
    e_min = var * (1.0 - (1.0 / q) ** 0.5) ** 2
    e_max = var * (1.0 + (1.0 / q) ** 0.5) ** 2
    e_val = np.linspace(e_min, e_max, pts)
    pdf = q / (2.0 * np.pi * var * e_val) * np.sqrt(np.clip((e_max - e_val) * (e_val - e_min), 0.0, None))
    return e_val, pdf


def _gaussian_kde(obs: np.ndarray, bandwidth: float, x: np.ndarray) -> np.ndarray:
    """Gaussian kernel density of *obs* at *x* (same as sklearn KernelDensity)."""
    z = (x[:, None] - obs[None, :]) / bandwidth
    return np.exp(-0.5 * z * z).sum(axis=1) / (obs.size * bandwidth * np.sqrt(2.0 * np.pi))


def _fit_mp_max_eigenvalue(e_val: np.ndarray, q: float, bandwidth: float) -> Tuple[float, float]:
    """Fit the MP noise variance to the eigenvalue density (MLAM Snippet 2.4).

    Returns ``(lambda_plus, sigma2)``. Raises when the fit does not converge
    instead of assuming ``sigma2 = 1``.
    """
    from scipy.optimize import minimize

    def sse(x: np.ndarray) -> float:
        grid, pdf0 = _mp_pdf(float(x[0]), q, 1000)
        pdf1 = _gaussian_kde(e_val, bandwidth, grid)
        return float(np.sum((pdf1 - pdf0) ** 2))

    out = minimize(sse, np.array([0.5]), bounds=((1e-5, 1.0 - 1e-5),))
    if not out.success:
        raise RuntimeError(f"denoise_mp: Marchenko-Pastur fit did not converge: {out.message}")
    var = float(out.x[0])
    return var * (1.0 + (1.0 / q) ** 0.5) ** 2, var


@frame_friendly
def denoise_mp(
    returns: Dict[str, List[float]],
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
    bandwidth: float = 0.01,
) -> np.ndarray:
    """Marchenko-Pastur denoised covariance, constant residual eigenvalue method.

    Lopez de Prado (2020), *Machine Learning for Asset Managers*, Snippets
    2.1-2.5: fit the Marchenko-Pastur noise variance to a Gaussian kernel
    density of the correlation matrix's eigenvalues, treat every eigenvalue
    below the fitted ``lambda_plus`` as noise and replace those by their
    average (preserving the trace), rescale to a unit diagonal, then restore
    the sample volatilities. Each asset keeps its sample variance.

    Args:
        returns: ``{asset_id: [daily_returns]}`` mapping, more observations
            than assets.
        annualize: Annualise the result.
        trading_days: Trading days per year.
        bandwidth: Gaussian kernel bandwidth for the eigenvalue density fit.
    """
    if not bandwidth > 0:
        raise ValueError(f"denoise_mp: bandwidth must be positive, got {bandwidth}")
    _, R = _returns_matrix(returns, "denoise_mp")
    t, n = R.shape
    vols = np.std(R, axis=0, ddof=1)
    if np.any(vols <= 0):
        raise ValueError("denoise_mp: an asset has zero return variance")
    if n == 1:
        result = np.array([[vols[0] ** 2]])
    else:
        if t <= n:
            raise ValueError(
                f"denoise_mp: needs more observations than assets (T={t}, N={n}); "
                "the Marchenko-Pastur fit assumes T/N > 1"
            )
        corr = np.corrcoef(R.T)
        e_val, e_vec = np.linalg.eigh(corr)
        order = np.argsort(e_val)[::-1]
        e_val, e_vec = e_val[order], e_vec[:, order]

        e_max, _ = _fit_mp_max_eigenvalue(e_val, t / n, bandwidth)
        n_facts = int(np.sum(e_val >= e_max))
        if n_facts < n:
            e_val = e_val.copy()
            e_val[n_facts:] = e_val[n_facts:].sum() / (n - n_facts)
        denoised = e_vec @ np.diag(e_val) @ e_vec.T
        d = np.sqrt(np.diag(denoised))
        denoised = denoised / np.outer(d, d)
        denoised = np.clip((denoised + denoised.T) / 2.0, -1.0, 1.0)
        np.fill_diagonal(denoised, 1.0)
        result = denoised * np.outer(vols, vols)

    if annualize:
        result = result * trading_days
    return result


@frame_friendly
def gerber_cov(
    returns: Dict[str, List[float]],
    *,
    threshold: float = 0.5,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
) -> np.ndarray:
    """Gerber statistic covariance (Gerber et al. 2021).

    Gerber, Markowitz, Ernst, Miao, Javid & Sargen, "The Gerber Statistic: A
    Robust Co-Movement Measure for Portfolio Optimization": a day counts for a
    pair when a return crosses ``threshold`` standard deviations; with ``n_c``
    concordant, ``n_d`` discordant and ``n_nn`` days on which neither asset
    crosses,

        ``g_ij = (n_c - n_d) / (T - n_nn)``

    This is the paper's statistic, which is positive semidefinite (Gerber et
    al. 2021, "Proofs that the Gerber Statistic is Positive Semidefinite"),
    unlike the earlier ``(n_c - n_d) / (n_c + n_d)`` form, whose matrices can
    give a long/short book negative variance. The covariance is
    ``g_ij * sd_i * sd_j`` with population standard deviations, matching
    riskfolio-lib's ``gerber_cov_stat1``.

    Args:
        returns: ``{asset_id: [daily_returns]}`` mapping.
        threshold: Exceedance threshold in units of standard deviation, in (0, 1).
        annualize: Annualise the result.
        trading_days: Trading days per year.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"gerber_cov: threshold must be in (0, 1), got {threshold}")
    ids, R = _returns_matrix(returns, "gerber_cov")
    t = R.shape[0]

    sd = np.std(R, axis=0)
    if np.any(sd <= 0):
        raise ValueError("gerber_cov: an asset has zero return variance")
    up = (R >= threshold * sd).astype(np.float64)
    down = (R <= -threshold * sd).astype(np.float64)
    neutral = 1.0 - up - down
    concordant = up.T @ up + down.T @ down
    discordant = up.T @ down + down.T @ up
    denom = t - neutral.T @ neutral
    if np.any(np.diag(denom) == 0):
        quiet = [ids[i] for i in np.flatnonzero(np.diag(denom) == 0)]
        raise ValueError(f"gerber_cov: {quiet} never cross the threshold; lower it")
    gerber_corr = (concordant - discordant) / denom

    result = gerber_corr * np.outer(sd, sd)
    if annualize:
        result = result * trading_days
    return result


@frame_friendly
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
    _, R = _returns_matrix(returns, "detone_cov")
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
