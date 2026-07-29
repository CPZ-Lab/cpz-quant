"""Copulas + synthetic scenario generation.

Separates the *dependence structure* from the *marginals* so you can model
tail-dependence (assets crashing together) explicitly and generate synthetic
stress scenarios that preserve both — the capability an institutional risk desk
uses for what-if and capacity analysis, and the biggest thing a plain
covariance matrix misses.

Implemented natively (numpy/scipy only):
  - Elliptical copulas: Gaussian (no tail dependence) and Student-t (symmetric
    tail dependence), with the correlation matrix estimated robustly from
    Kendall's tau (ρ = sin(πτ/2)) and PSD-projected.
  - Archimedean bivariate copulas: Clayton (lower-tail) and Gumbel (upper-tail),
    fit from Kendall's tau.
  - Empirical marginals (rank -> ECDF -> inverse via interpolation).
  - ``synthetic_returns`` — fit a copula to historical returns and sample new
    scenarios preserving marginals + dependence.
  - Closed-form tail-dependence coefficients per family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import stats

EPSILON = 1e-12


def pseudo_observations(X: np.ndarray) -> np.ndarray:
    """Map each column to (0,1) uniforms via ranks: u = rank / (T+1)."""
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    U = np.empty_like(X)
    for j in range(X.shape[1]):
        U[:, j] = stats.rankdata(X[:, j], method="average") / (T + 1.0)
    return U


def _nearest_psd_corr(R: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD correlation (eigen-clip)."""
    R = (R + R.T) / 2.0
    vals, vecs = np.linalg.eigh(R)
    vals = np.clip(vals, EPSILON, None)
    R = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(R), EPSILON, None))
    R = R / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def _kendall_correlation(X: np.ndarray) -> np.ndarray:
    """Robust elliptical correlation from Kendall's tau: rho = sin(pi*tau/2)."""
    N = X.shape[1]
    R = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            tau, _ = stats.kendalltau(X[:, i], X[:, j])
            if not np.isfinite(tau):
                tau = 0.0
            R[i, j] = R[j, i] = np.sin(np.pi * tau / 2.0)
    return _nearest_psd_corr(R)


# ── Elliptical copulas ───────────────────────────────────────────────────────
@dataclass
class GaussianCopula:
    correlation: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "GaussianCopula":
        return cls(correlation=_kendall_correlation(np.asarray(X, dtype=float)))

    def sample(self, n: int, random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        L = np.linalg.cholesky(self.correlation)
        Z = rng.standard_normal((n, self.correlation.shape[0])) @ L.T
        return stats.norm.cdf(Z)

    def tail_dependence(self) -> Tuple[float, float]:
        return 0.0, 0.0  # Gaussian copula is tail-independent


@dataclass
class StudentTCopula:
    correlation: np.ndarray
    dof: float

    @classmethod
    def fit(cls, X: np.ndarray, dof_grid=None) -> "StudentTCopula":
        X = np.asarray(X, dtype=float)
        R = _kendall_correlation(X)
        U = pseudo_observations(X)
        U = np.clip(U, 1e-6, 1 - 1e-6)
        # MLE for dof over a grid using the t-copula log-likelihood.
        grid = dof_grid if dof_grid is not None else [3, 4, 5, 6, 8, 10, 15, 20, 30]
        best_dof, best_ll = grid[0], -np.inf
        Rinv = np.linalg.inv(R)
        _, logdet = np.linalg.slogdet(R)
        d = R.shape[0]
        for nu in grid:
            Z = stats.t.ppf(U, nu)
            # log t-copula density = log multivariate-t density - Σ log marginal-t
            log_mvt = _log_mvt_density(Z, Rinv, logdet, nu, d)
            log_marg = np.log(stats.t.pdf(Z, nu)).sum(axis=1)
            ll = float(np.sum(log_mvt - log_marg))
            if np.isfinite(ll) and ll > best_ll:
                best_ll, best_dof = ll, nu
        return cls(correlation=R, dof=float(best_dof))

    def sample(self, n: int, random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        d = self.correlation.shape[0]
        L = np.linalg.cholesky(self.correlation)
        Z = rng.standard_normal((n, d)) @ L.T
        g = rng.chisquare(self.dof, size=n) / self.dof
        X = Z / np.sqrt(g)[:, None]
        return stats.t.cdf(X, self.dof)

    def tail_dependence(self) -> Tuple[float, float]:
        # Symmetric elliptical tail dependence (average off-diagonal rho).
        R = self.correlation
        off = R[np.triu_indices_from(R, k=1)]
        rho = float(np.mean(off)) if off.size else 0.0
        nu = self.dof
        lam = 2.0 * stats.t.cdf(-np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho)), nu + 1.0)
        lam = float(np.clip(lam, 0.0, 1.0))
        return lam, lam


def _log_mvt_density(Z: np.ndarray, Rinv: np.ndarray, logdet: float, nu: float, d: int) -> np.ndarray:
    from scipy.special import gammaln
    quad = np.einsum("ij,jk,ik->i", Z, Rinv, Z)
    c = gammaln((nu + d) / 2.0) - gammaln(nu / 2.0) - 0.5 * d * np.log(nu * np.pi) - 0.5 * logdet
    return c - 0.5 * (nu + d) * np.log1p(quad / nu)


# ── Archimedean bivariate copulas ────────────────────────────────────────────
@dataclass
class ClaytonCopula:
    theta: float  # > 0; lower-tail dependence

    @classmethod
    def fit(cls, X: np.ndarray) -> "ClaytonCopula":
        tau, _ = stats.kendalltau(X[:, 0], X[:, 1])
        tau = float(np.clip(tau, 1e-3, 0.99))
        return cls(theta=2.0 * tau / (1.0 - tau))

    def sample(self, n: int, random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        th = self.theta
        u = rng.uniform(size=n)
        w = rng.uniform(size=n)
        # Conditional sampling: v = (u^-th (w^(-th/(1+th)) - 1) + 1)^(-1/th)
        v = (u ** (-th) * (w ** (-th / (1.0 + th)) - 1.0) + 1.0) ** (-1.0 / th)
        return np.column_stack([u, v])

    def tail_dependence(self) -> Tuple[float, float]:
        return float(2.0 ** (-1.0 / self.theta)), 0.0


@dataclass
class GumbelCopula:
    theta: float  # >= 1; upper-tail dependence

    @classmethod
    def fit(cls, X: np.ndarray) -> "GumbelCopula":
        tau, _ = stats.kendalltau(X[:, 0], X[:, 1])
        tau = float(np.clip(tau, 1e-3, 0.99))
        return cls(theta=1.0 / (1.0 - tau))

    def sample(self, n: int, random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        th = self.theta
        alpha = 1.0 / th  # in (0,1) since theta > 1
        V = _positive_stable(alpha, n, rng)  # Gumbel frailty
        e1 = rng.exponential(1.0, n)
        e2 = rng.exponential(1.0, n)
        u = np.exp(-((e1 / V) ** alpha))
        v = np.exp(-((e2 / V) ** alpha))
        return np.column_stack([np.clip(u, EPSILON, 1 - EPSILON),
                                np.clip(v, EPSILON, 1 - EPSILON)])

    def tail_dependence(self) -> Tuple[float, float]:
        return 0.0, float(2.0 - 2.0 ** (1.0 / self.theta))


def _positive_stable(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Positive alpha-stable sampler (Kanter, 1975) with Laplace transform
    exp(-s^alpha), alpha in (0,1). Used as the Gumbel copula frailty."""
    theta = rng.uniform(EPSILON, np.pi - EPSILON, n)
    w = rng.exponential(1.0, n)
    a = np.sin(alpha * theta) / (np.sin(theta) ** (1.0 / alpha))
    b = (np.sin((1.0 - alpha) * theta) / w) ** ((1.0 - alpha) / alpha)
    return np.clip(a * b, EPSILON, None)


# ── Empirical marginals + synthetic scenarios ────────────────────────────────
def _empirical_quantile(col: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse ECDF: map uniforms back to the data's empirical distribution."""
    xs = np.sort(col)
    T = xs.size
    # Interpolate on the ECDF grid ((i+1)/(T+1)).
    grid = (np.arange(1, T + 1)) / (T + 1.0)
    return np.interp(u, grid, xs)


_FAMILIES = {
    "gaussian": GaussianCopula, "student-t": StudentTCopula,
    "clayton": ClaytonCopula, "gumbel": GumbelCopula,
}


def fit_copula(returns: np.ndarray, family: str = "student-t"):
    """Fit a copula of the given family to a return matrix."""
    fam = _FAMILIES.get(family)
    if fam is None:
        raise ValueError(f"Unknown copula family {family!r}. Use: {list(_FAMILIES)}")
    if family in ("clayton", "gumbel") and np.asarray(returns).shape[1] != 2:
        raise ValueError(f"{family} copula is bivariate; pass exactly 2 assets")
    return fam.fit(np.asarray(returns, dtype=float))  # type: ignore[attr-defined]


def synthetic_returns(
    returns: np.ndarray,
    n_samples: int = 10000,
    *,
    family: str = "student-t",
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic return scenarios preserving marginals + dependence.

    Fits the chosen copula to the historical returns, samples uniforms, and maps
    them back through each asset's empirical marginal. Use for stress testing,
    capacity/what-if analysis, and Monte-Carlo of tail-aware portfolios.
    """
    R = np.asarray(returns, dtype=float)
    copula = fit_copula(R, family)
    U = copula.sample(n_samples, random_state=random_state)
    out = np.empty_like(U)
    for j in range(R.shape[1]):
        out[:, j] = _empirical_quantile(R[:, j], U[:, j])
    return out
