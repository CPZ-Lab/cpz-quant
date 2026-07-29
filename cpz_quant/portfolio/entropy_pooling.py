"""Entropy Pooling (Meucci) — fully-flexible views on a scenario set.

The posterior scenario probabilities are the distribution closest to the prior
(minimum relative entropy / KL divergence) that satisfies the analyst's views —
a strict generalization of Black-Litterman that handles views on any statistic
(mean, volatility, correlation, tail) and inequality views, on any distribution.

Solved via the convex dual: with equality views ``A q = b`` and inequality views
``G q <= h`` the posterior is ``q_i ∝ p_i * exp(A_i·λ + G_i·μ)`` where the
multipliers minimize the dual free-energy; equality multipliers are free and
inequality multipliers are non-negative.

Reference: Meucci (2008), "Fully Flexible Views: Theory and Practice".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
from scipy import optimize as sp_opt

EPSILON = 1e-12


@dataclass
class EntropyPoolingResult:
    posterior: np.ndarray                      # (T,) posterior probabilities
    prior: np.ndarray                          # (T,) prior probabilities
    effective_number_of_scenarios: float       # exp(entropy(q)); <= T
    relative_entropy: float                     # KL(q || p)
    converged: bool

    def effective_ratio(self) -> float:
        """Effective scenarios as a fraction of the sample (1.0 = undistorted)."""
        return float(self.effective_number_of_scenarios / self.posterior.size)


def _normalize_prior(prior: Optional[np.ndarray], T: int) -> np.ndarray:
    if prior is None:
        return np.full(T, 1.0 / T)
    p = np.asarray(prior, dtype=float)
    p = np.clip(p, EPSILON, None)
    return p / p.sum()


def entropy_pooling(
    *,
    n_scenarios: Optional[int] = None,
    prior: Optional[np.ndarray] = None,
    a_eq: Optional[np.ndarray] = None,
    b_eq: Optional[np.ndarray] = None,
    a_ineq: Optional[np.ndarray] = None,
    b_ineq: Optional[np.ndarray] = None,
) -> EntropyPoolingResult:
    """Compute posterior scenario probabilities under views via entropy pooling.

    Args:
        n_scenarios: number of scenarios T (required if ``prior`` is None).
        prior: (T,) prior probabilities (defaults to uniform).
        a_eq, b_eq: equality views — ``A q = b`` with ``A`` shape (K_eq, T).
        a_ineq, b_ineq: inequality views — ``G q <= h`` with ``G`` shape (K_ineq, T).

    Each row of ``A``/``G`` is the value of a view statistic across scenarios
    (e.g. asset returns for a mean view; squared deviations for a variance view).
    """
    if prior is not None:
        T = np.asarray(prior).size
    elif n_scenarios is not None:
        T = int(n_scenarios)
    else:
        raise ValueError("provide prior or n_scenarios")
    p = _normalize_prior(prior, T)
    log_p = np.log(p)

    A = np.asarray(a_eq, dtype=float).reshape(-1, T) if a_eq is not None else np.zeros((0, T))
    b = np.asarray(b_eq, dtype=float).reshape(-1) if b_eq is not None else np.zeros(0)
    G = np.asarray(a_ineq, dtype=float).reshape(-1, T) if a_ineq is not None else np.zeros((0, T))
    h = np.asarray(b_ineq, dtype=float).reshape(-1) if b_ineq is not None else np.zeros(0)
    Keq, Kin = A.shape[0], G.shape[0]

    if Keq == 0 and Kin == 0:
        q = p.copy()
        return EntropyPoolingResult(q, p, _eff_scenarios(q), 0.0, True)

    M = np.vstack([A, G]) if (Keq + Kin) else np.zeros((0, T))
    rhs = np.concatenate([b, h])

    # Posterior q_i ∝ p_i exp(-(M_i·λ)); with this sign, λ_j >= 0 on an
    # inequality row enforces (M q)_j <= rhs_j (Meucci's standard form).
    def q_of(lmbd: np.ndarray) -> np.ndarray:
        x = log_p - M.T @ lmbd
        x -= x.max()
        w = np.exp(x)
        return w / w.sum()

    def dual(lmbd: np.ndarray) -> float:
        # g(λ) = ln( Σ p_i exp(-M_i·λ) ) + λ·rhs  (minimize; convex)
        x = log_p - M.T @ lmbd
        xmax = x.max()
        logZ = xmax + np.log(np.sum(np.exp(x - xmax)))
        return float(logZ + lmbd @ rhs)

    def dual_grad(lmbd: np.ndarray) -> np.ndarray:
        return rhs - M @ q_of(lmbd)

    # Equality multipliers free; inequality multipliers >= 0.
    bounds = [(None, None)] * Keq + [(0.0, None)] * Kin
    x0 = np.zeros(Keq + Kin)
    res = sp_opt.minimize(dual, x0, jac=dual_grad, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 1000, "ftol": 1e-12})
    q = q_of(res.x)
    kl = float(np.sum(q * (np.log(np.clip(q, EPSILON, None)) - log_p)))
    return EntropyPoolingResult(q, p, _eff_scenarios(q), kl, bool(res.success))


def _eff_scenarios(q: np.ndarray) -> float:
    ent = -np.sum(q * np.log(np.clip(q, EPSILON, None)))
    return float(np.exp(ent))


# ── convenience view builders ────────────────────────────────────────────────
def mean_view_rows(
    returns: np.ndarray,
    views: Mapping[int, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build equality-view rows for ``E_q[asset_k] = target``.

    Args:
        returns: (T, N) scenario matrix.
        views: ``{asset_index: target_mean}``.
    Returns ``(A, b)`` for :func:`entropy_pooling`.
    """
    R = np.asarray(returns, dtype=float)
    rows = [R[:, k] for k in views]
    b = np.array([views[k] for k in views], dtype=float)
    return np.vstack(rows), b


def posterior_moments(returns: np.ndarray, q: np.ndarray) -> Dict[str, np.ndarray]:
    """Posterior mean and covariance of the asset returns under probabilities q."""
    R = np.asarray(returns, dtype=float)
    q = np.asarray(q, dtype=float)
    mu = q @ R
    d = R - mu
    cov = (d * q[:, None]).T @ d
    return {"mean": mu, "cov": cov}
