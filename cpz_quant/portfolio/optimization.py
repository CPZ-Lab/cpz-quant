"""Portfolio optimization: MVO, Black-Litterman, HRP, mean-CVaR, robust, max-diversification,
tracking error, turnover-penalized, and the Grinold-Kahn alpha-risk-cost framework.

All functions are pure: data in, results out. No DB access, no API calls.
Requires numpy (core dep) and scipy (core dep).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize as sp_opt

from cpz_quant.frames import frame_friendly

EPSILON: float = 1e-15
TRADING_DAYS: int = 252


# ── Result models ────────────────────────────────────────────────────

@dataclass
class OptResult:
    method: str = ""
    weights: Dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BLResult:
    posterior_returns: Dict[str, float] = field(default_factory=dict)
    optimal_weights: Dict[str, float] = field(default_factory=dict)
    tilts: Dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0


@dataclass
class ARCResult:
    """Alpha-Risk-Cost optimisation result."""
    optimal_weights: Dict[str, float] = field(default_factory=dict)
    expected_alpha: float = 0.0
    expected_risk: float = 0.0
    expected_cost: float = 0.0
    information_ratio: float = 0.0
    transfer_coefficient: float = 0.0


@dataclass
class Constraints:
    """Optimisation constraints — every limit is configurable."""
    long_only: bool = False
    max_weight: float = 1.0
    min_weight: float = -1.0
    max_gross_exposure: float = 2.0
    max_net_exposure: float = 1.0
    sector_limits: Optional[Dict[str, float]] = None
    factor_limits: Optional[Dict[str, Tuple[float, float]]] = None
    max_turnover: Optional[float] = None
    max_tracking_error: Optional[float] = None


# ── Helpers ──────────────────────────────────────────────────────────

def _build_matrices(
    returns: Dict[str, List[float]],
    risk_free_rate: float = 0.0,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    ids = list(returns.keys())
    min_len = min(len(returns[i]) for i in ids)
    R = np.column_stack([np.array(returns[i][-min_len:], dtype=np.float64) for i in ids])
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(all="ignore"):
        mu = np.mean(R, axis=0) * TRADING_DAYS
        cov = np.cov(R.T, ddof=1) * TRADING_DAYS
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    return ids, mu, cov


def _bounds(n: int, constraints: Optional[Constraints] = None):
    c = constraints or Constraints()
    lo = 0.0 if c.long_only else c.min_weight
    hi = c.max_weight
    return [(lo, hi)] * n


def _metrics(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float, ids: List[str]) -> OptResult:
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sr = (ret - rf) / vol if vol > EPSILON else 0.0
    return OptResult(
        weights={ids[j]: round(float(w[j]), 6) for j in range(len(ids))},
        expected_return=round(ret * 100, 4),
        volatility=round(vol * 100, 4),
        sharpe_ratio=round(sr, 4),
    )


# ── Core optimisers ─────────────────────────────────────────────────

@frame_friendly
def mean_variance(
    returns: Dict[str, List[float]],
    *,
    risk_free_rate: float = 0.0,
    target_return: Optional[float] = None,
    constraints: Optional[Constraints] = None,
) -> OptResult:
    """Classic Markowitz mean-variance optimisation.

    When *target_return* is given, finds the minimum-variance portfolio
    achieving that return. Otherwise maximises Sharpe ratio.
    """
    ids, mu, cov = _build_matrices(returns, risk_free_rate)
    n = len(ids)
    bounds = _bounds(n, constraints)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        cons.append({"type": "eq", "fun": lambda w: w @ mu - target_return})

    def objective(w):
        return float(w @ cov @ w)

    x0 = np.ones(n) / n
    result = sp_opt.minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                             options={"maxiter": 500, "ftol": 1e-12})
    w = result.x / max(np.sum(result.x), EPSILON) if np.sum(result.x) > EPSILON else np.ones(n) / n
    r = _metrics(w, mu, cov, risk_free_rate, ids)
    r.method = "mean_variance"
    return r


@frame_friendly
def min_variance(
    returns: Dict[str, List[float]],
    *,
    constraints: Optional[Constraints] = None,
) -> OptResult:
    """Global minimum variance — no return estimates needed."""
    ids, mu, cov = _build_matrices(returns)
    n = len(ids)
    bounds = _bounds(n, constraints)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w):
        return float(w @ cov @ w)

    x0 = np.ones(n) / n
    result = sp_opt.minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                             options={"maxiter": 500, "ftol": 1e-12})
    w = np.abs(result.x)
    w /= max(np.sum(w), EPSILON)
    r = _metrics(w, mu, cov, 0.0, ids)
    r.method = "min_variance"
    return r


@frame_friendly
def max_sharpe(
    returns: Dict[str, List[float]],
    *,
    risk_free_rate: float = 0.0,
    constraints: Optional[Constraints] = None,
) -> OptResult:
    """Maximum Sharpe ratio (tangency portfolio)."""
    ids, mu, cov = _build_matrices(returns, risk_free_rate)
    n = len(ids)
    bounds = _bounds(n, constraints)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret - risk_free_rate) / max(vol, EPSILON)

    x0 = np.ones(n) / n
    result = sp_opt.minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons,
                             options={"maxiter": 500, "ftol": 1e-12})
    w = result.x
    r = _metrics(w, mu, cov, risk_free_rate, ids)
    r.method = "max_sharpe"
    return r


@frame_friendly
def risk_parity(
    returns: Dict[str, List[float]],
    *,
    budget: Optional[Dict[str, float]] = None,
    max_iter: int = 500,
    tolerance: float = 1e-10,
) -> OptResult:
    """Risk parity: equal (or budgeted) risk contribution.

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        budget: ``{asset_id: target_risk_pct}`` — if None, equal budget.
        max_iter: Maximum solver iterations.
        tolerance: Convergence tolerance.
    """
    ids, mu, cov = _build_matrices(returns)
    n = len(ids)

    target = np.ones(n) / n
    if budget:
        for j, aid in enumerate(ids):
            if aid in budget:
                target[j] = budget[aid]
        target /= max(np.sum(target), EPSILON)

    w = np.ones(n) / n
    for _ in range(max_iter):
        sigma_w = cov @ w
        port_vol = float(np.sqrt(w @ sigma_w))
        if port_vol < EPSILON:
            break
        rc = w * sigma_w / port_vol
        new_w = w * (target * port_vol / (rc + EPSILON)) ** 0.5
        new_w /= max(np.sum(new_w), EPSILON)
        if np.max(np.abs(new_w - w)) < tolerance:
            w = new_w
            break
        w = new_w

    r = _metrics(w, mu, cov, 0.0, ids)
    r.method = "risk_parity"
    return r


def equal_weight(
    asset_ids: List[str],
) -> OptResult:
    """1/N equal-weight portfolio."""
    n = len(asset_ids)
    w = 1.0 / n if n > 0 else 0.0
    return OptResult(
        method="equal_weight",
        weights={a: round(w, 6) for a in asset_ids},
    )


# ── Advanced optimisers ──────────────────────────────────────────────

def black_litterman(
    market_caps: Dict[str, float],
    cov: np.ndarray,
    views: Dict[str, float],
    view_confidence: Dict[str, float],
    *,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    asset_ids: Optional[List[str]] = None,
    risk_free_rate: float = 0.0,
) -> BLResult:
    """Black-Litterman (1992) model.

    Combines market equilibrium with active views to produce
    posterior expected returns, then optimises.

    Args:
        market_caps: ``{asset_id: market_cap}``.
        cov: ``(n, n)`` covariance matrix (annualised).
        views: ``{asset_id: expected_return_view}``.
        view_confidence: ``{asset_id: confidence_0_to_1}``.
        risk_aversion: Market risk aversion parameter (delta).
        tau: Uncertainty scaling of the prior (typically 0.01-0.05).
        asset_ids: Asset ordering matching *cov*.
        risk_free_rate: Annual risk-free rate.
    """
    if asset_ids is None:
        asset_ids = list(market_caps.keys())
    n = len(asset_ids)
    Sigma = np.asarray(cov, dtype=np.float64)

    caps = np.array([market_caps.get(a, 0.0) for a in asset_ids], dtype=np.float64)
    w_mkt = caps / max(np.sum(caps), EPSILON)
    pi = risk_aversion * Sigma @ w_mkt

    view_assets = [a for a in asset_ids if a in views]
    k = len(view_assets)
    if k == 0:
        w = w_mkt
        ret = float(w @ pi)
        vol = float(np.sqrt(w @ Sigma @ w))
        return BLResult(
            posterior_returns={asset_ids[j]: round(float(pi[j]), 6) for j in range(n)},
            optimal_weights={asset_ids[j]: round(float(w[j]), 6) for j in range(n)},
            tilts={},
            expected_return=round(ret * 100, 4),
            volatility=round(vol * 100, 4),
            sharpe_ratio=round((ret - risk_free_rate) / max(vol, EPSILON), 4),
        )

    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega_diag = np.zeros(k)
    for i, a in enumerate(view_assets):
        j = asset_ids.index(a)
        P[i, j] = 1.0
        Q[i] = views[a]
        conf = max(min(view_confidence.get(a, 0.5), 0.999), 0.001)
        omega_diag[i] = (1.0 - conf) / conf * (P[i] @ (tau * Sigma) @ P[i])

    Omega = np.diag(omega_diag)
    tau_sigma = tau * Sigma
    tau_sigma_inv = np.linalg.inv(tau_sigma)
    omega_inv = np.linalg.inv(Omega)

    post_cov = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
    post_mu = post_cov @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

    try:
        inv_cov = np.linalg.inv(Sigma)
        w_opt = inv_cov @ post_mu
        w_opt /= max(np.sum(np.abs(w_opt)), EPSILON)
    except np.linalg.LinAlgError:
        w_opt = w_mkt

    ret = float(w_opt @ post_mu)
    vol = float(np.sqrt(w_opt @ Sigma @ w_opt))

    return BLResult(
        posterior_returns={asset_ids[j]: round(float(post_mu[j]), 6) for j in range(n)},
        optimal_weights={asset_ids[j]: round(float(w_opt[j]), 6) for j in range(n)},
        tilts={asset_ids[j]: round(float(w_opt[j] - w_mkt[j]), 6) for j in range(n)},
        expected_return=round(ret * 100, 4),
        volatility=round(vol * 100, 4),
        sharpe_ratio=round((ret - risk_free_rate) / max(vol, EPSILON), 4),
    )


@frame_friendly
def hierarchical_risk_parity(
    returns: Dict[str, List[float]],
    *,
    linkage_method: str = "single",
) -> OptResult:
    """Hierarchical Risk Parity (Lopez de Prado 2016).

    No covariance inversion — robust to estimation error.

    1. Distance matrix from correlation
    2. Hierarchical clustering
    3. Quasi-diagonalise
    4. Recursive bisection

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        linkage_method: ``"single"`` (default), ``"complete"``, ``"average"``,
                        or ``"ward"``.
    """
    from scipy.cluster.hierarchy import linkage as sp_linkage
    from scipy.spatial.distance import squareform

    ids, mu, cov = _build_matrices(returns)
    n = len(ids)

    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = cov / np.outer(std + EPSILON, std + EPSILON)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1, 1)

    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=0.0)
    condensed = squareform(dist, checks=False)

    Z = sp_linkage(condensed, method=linkage_method)

    def _get_quasi_diag(link: np.ndarray, n_items: int) -> list:
        link = link.astype(int)
        clusters = {i: [i] for i in range(n_items)}
        for i in range(link.shape[0]):
            c1 = link[i, 0]
            c2 = link[i, 1]
            new_id = n_items + i
            clusters[new_id] = clusters[c1] + clusters[c2]
        return clusters[n_items + link.shape[0] - 1]

    sorted_idx = _get_quasi_diag(Z, n)

    def _recursive_bisect(cov_mat: np.ndarray, sorted_items: list) -> np.ndarray:
        w = np.ones(len(sorted_items))
        cluster_items = [sorted_items]
        while len(cluster_items) > 0:
            new_clusters = []
            for cluster in cluster_items:
                if len(cluster) <= 1:
                    continue
                half = len(cluster) // 2
                left = cluster[:half]
                right = cluster[half:]

                left_idx = [sorted_items.index(i) for i in left]
                right_idx = [sorted_items.index(i) for i in right]

                cov_left = cov_mat[np.ix_(left, left)]
                cov_right = cov_mat[np.ix_(right, right)]

                inv_left = 1.0 / max(np.sqrt(_cluster_var(cov_left)), EPSILON)
                inv_right = 1.0 / max(np.sqrt(_cluster_var(cov_right)), EPSILON)
                alpha = inv_left / (inv_left + inv_right)

                for i in left_idx:
                    w[i] *= alpha
                for i in right_idx:
                    w[i] *= (1.0 - alpha)

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)
            cluster_items = new_clusters
        return w

    def _cluster_var(cov_sub: np.ndarray) -> float:
        n_sub = cov_sub.shape[0]
        if n_sub == 1:
            return float(cov_sub[0, 0])
        inv_diag = 1.0 / np.maximum(np.diag(cov_sub), EPSILON)
        w_sub = inv_diag / np.sum(inv_diag)
        return float(w_sub @ cov_sub @ w_sub)

    cov_reordered = cov[np.ix_(sorted_idx, sorted_idx)]
    hrp_w = _recursive_bisect(cov_reordered, list(range(n)))

    final_w = np.zeros(n)
    for i, orig_idx in enumerate(sorted_idx):
        final_w[orig_idx] = hrp_w[i]
    final_w /= max(np.sum(final_w), EPSILON)

    r = _metrics(final_w, mu, cov, 0.0, ids)
    r.method = "hierarchical_risk_parity"
    r.info["linkage_method"] = linkage_method
    return r


def _quasi_diag_order(link: np.ndarray, n_items: int) -> list:
    link = link.astype(int)
    clusters = {i: [i] for i in range(n_items)}
    for i in range(link.shape[0]):
        clusters[n_items + i] = clusters[link[i, 0]] + clusters[link[i, 1]]
    return clusters[n_items + link.shape[0] - 1]


def _cluster_variance(cov_mat: np.ndarray, items: list) -> float:
    sub = cov_mat[np.ix_(items, items)]
    inv = 1.0 / np.maximum(np.diag(sub), EPSILON)
    w = inv / inv.sum()
    return float(w @ sub @ w)


def _recursive_bisection(cov_mat: np.ndarray, order: list) -> np.ndarray:
    """Inverse-risk recursive bisection over `order` (indices into cov_mat)."""
    w = np.ones(cov_mat.shape[0])
    clusters = [list(order)]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]
            vl = _cluster_variance(cov_mat, left)
            vr = _cluster_variance(cov_mat, right)
            a = (1.0 / vl) / (1.0 / vl + 1.0 / vr) if (vl > 0 and vr > 0) else 0.5
            for i in left:
                w[i] *= a
            for i in right:
                w[i] *= (1.0 - a)
            if len(left) > 1:
                nxt.append(left)
            if len(right) > 1:
                nxt.append(right)
        clusters = nxt
    return w


@frame_friendly
def hierarchical_equal_risk_contribution(
    returns: Dict[str, List[float]],
    *,
    linkage_method: str = "ward",
    n_clusters: Optional[int] = None,
) -> OptResult:
    """Hierarchical Equal Risk Contribution (Raffinot 2018).

    Extends HRP: cluster the assets, allocate BETWEEN clusters by inverse
    cluster-risk along the dendrogram, and WITHIN each cluster by inverse
    variance. Reduces the concentration HRP can leave in a single dendrogram
    branch. The number of clusters is chosen from the largest gap in the merge
    heights unless given.

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        linkage_method: scipy linkage ("ward" default, "single", "average", ...).
        n_clusters: force a cluster count; ``None`` auto-selects.
    """
    from scipy.cluster.hierarchy import fcluster
    from scipy.cluster.hierarchy import linkage as sp_linkage
    from scipy.spatial.distance import squareform

    ids, mu, cov = _build_matrices(returns)
    n = len(ids)
    if n == 1:
        r = _metrics(np.array([1.0]), mu, cov, 0.0, ids)
        r.method = "hierarchical_equal_risk_contribution"
        return r

    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = np.clip(np.nan_to_num(cov / np.outer(std + EPSILON, std + EPSILON)), -1, 1)
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    Z = sp_linkage(squareform(np.nan_to_num(dist, nan=1.0), checks=False), method=linkage_method)

    if n_clusters is None:
        heights = Z[:, 2]
        if heights.size >= 2:
            K = int(min(max(n - (int(np.argmax(np.diff(heights))) + 1), 2), n))
        else:
            K = 2
    else:
        K = int(min(max(n_clusters, 1), n))
    labels = fcluster(Z, K, criterion="maxclust")
    uniq = list(dict.fromkeys(labels[i] for i in _quasi_diag_order(Z, n)))  # dendrogram order

    # Within-cluster inverse-variance weights.
    within = np.zeros(n)
    cluster_assets = {}
    for c in uniq:
        idx = list(np.where(labels == c)[0])
        cluster_assets[c] = idx
        iv = 1.0 / np.maximum(np.diag(cov)[idx], EPSILON)
        within[idx] = iv / iv.sum()

    # Cluster-level covariance, then inverse-risk bisection between clusters.
    Kn = len(uniq)
    Kc = np.zeros((Kn, Kn))
    for a, ca in enumerate(uniq):
        for b, cb in enumerate(uniq):
            ia, ib = cluster_assets[ca], cluster_assets[cb]
            Kc[a, b] = within[ia] @ cov[np.ix_(ia, ib)] @ within[ib]
    between = _recursive_bisection(Kc, list(range(Kn)))
    between = between / max(between.sum(), EPSILON)

    w = np.zeros(n)
    for a, ca in enumerate(uniq):
        idx = cluster_assets[ca]
        w[idx] = between[a] * within[idx]
    w /= max(w.sum(), EPSILON)

    r = _metrics(w, mu, cov, 0.0, ids)
    r.method = "hierarchical_equal_risk_contribution"
    r.info["linkage_method"] = linkage_method
    r.info["n_clusters"] = int(Kn)
    return r


def _analytic_min_variance(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    inv = np.linalg.pinv(cov + np.eye(n) * EPSILON)
    ones = np.ones(n)
    w = inv @ ones
    s = w.sum()
    return w / s if abs(s) > EPSILON else ones / n


def _analytic_max_sharpe(cov: np.ndarray, mu: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    inv = np.linalg.pinv(cov + np.eye(n) * EPSILON)
    w = inv @ mu
    s = w.sum()
    return w / s if abs(s) > EPSILON else np.ones(n) / n


def _cluster_labels(cov: np.ndarray, linkage_method: str, n_clusters):
    """Hierarchical clusters + dendrogram-ordered unique labels (shared by NCO)."""
    from scipy.cluster.hierarchy import fcluster
    from scipy.cluster.hierarchy import linkage as sp_linkage
    from scipy.spatial.distance import squareform

    n = cov.shape[0]
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = np.clip(np.nan_to_num(cov / np.outer(std + EPSILON, std + EPSILON)), -1, 1)
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    Z = sp_linkage(squareform(np.nan_to_num(dist, nan=1.0), checks=False), method=linkage_method)
    if n_clusters is None:
        heights = Z[:, 2]
        K = int(min(max(n - (int(np.argmax(np.diff(heights))) + 1), 2), n)) if heights.size >= 2 else 2
    else:
        K = int(min(max(n_clusters, 1), n))
    labels = fcluster(Z, K, criterion="maxclust")
    order = list(dict.fromkeys(labels[i] for i in _quasi_diag_order(Z, n)))
    return labels, order


@frame_friendly
def nested_clustered_optimization(
    returns: Dict[str, List[float]],
    *,
    objective: str = "min_variance",     # 'min_variance' | 'max_sharpe'
    linkage_method: str = "ward",
    n_clusters: Optional[int] = None,
) -> OptResult:
    """Nested Clustered Optimization (Lopez de Prado 2019).

    Optimizes WITHIN each cluster and ACROSS clusters on a reduced covariance —
    de-noising the allocation by never inverting the full (unstable) covariance
    matrix. Sub-problems are solved analytically (min-variance or max-Sharpe).

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        objective: ``"min_variance"`` or ``"max_sharpe"`` at both levels.
        n_clusters: force a cluster count; ``None`` auto-selects.
    """
    ids, mu, cov = _build_matrices(returns)
    n = len(ids)
    if n == 1:
        r = _metrics(np.array([1.0]), mu, cov, 0.0, ids)
        r.method = "nested_clustered_optimization"
        return r

    labels, uniq = _cluster_labels(cov, linkage_method, n_clusters)
    cluster_assets = {c: list(np.where(labels == c)[0]) for c in uniq}

    def _sub(cov_s, mu_s):
        return _analytic_min_variance(cov_s) if objective == "min_variance" \
            else _analytic_max_sharpe(cov_s, mu_s)

    # Within-cluster weights.
    intra = np.zeros(n)
    for c in uniq:
        idx = cluster_assets[c]
        intra[idx] = _sub(cov[np.ix_(idx, idx)], mu[idx])

    # Reduced (cluster-level) moments, then across-cluster weights.
    Kn = len(uniq)
    Kcov = np.zeros((Kn, Kn))
    Kmu = np.zeros(Kn)
    for a, ca in enumerate(uniq):
        ia = cluster_assets[ca]
        Kmu[a] = intra[ia] @ mu[ia]
        for b, cb in enumerate(uniq):
            ib = cluster_assets[cb]
            Kcov[a, b] = intra[ia] @ cov[np.ix_(ia, ib)] @ intra[ib]
    across = _sub(Kcov, Kmu)

    w = np.zeros(n)
    for a, ca in enumerate(uniq):
        idx = cluster_assets[ca]
        w[idx] = across[a] * intra[idx]
    tot = w.sum()
    w = w / tot if abs(tot) > EPSILON else np.ones(n) / n

    r = _metrics(w, mu, cov, 0.0, ids)
    r.method = "nested_clustered_optimization"
    r.info["objective"] = objective
    r.info["n_clusters"] = int(Kn)
    return r


@frame_friendly
def schur_complementary_allocation(
    returns: Dict[str, List[float]],
    *,
    gamma: float = 0.5,
    linkage_method: str = "single",
) -> OptResult:
    """Schur complementary allocation (Cotton 2022).

    Interpolates between HRP (``gamma=0``) and full min-variance (``gamma=1``):
    at each dendrogram bisection the cluster covariances are augmented with the
    Schur complement of the cross-cluster block, so cross-cluster covariance is
    respected during top-down risk splitting.

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        gamma: 0 -> HRP, 1 -> fully Schur-augmented (toward min-variance).
    """
    from scipy.cluster.hierarchy import linkage as sp_linkage
    from scipy.spatial.distance import squareform

    ids, mu, cov = _build_matrices(returns)
    n = len(ids)
    if n == 1:
        r = _metrics(np.array([1.0]), mu, cov, 0.0, ids)
        r.method = "schur_complementary_allocation"
        return r
    g = float(np.clip(gamma, 0.0, 1.0))

    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = np.clip(np.nan_to_num(cov / np.outer(std + EPSILON, std + EPSILON)), -1, 1)
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    Z = sp_linkage(squareform(np.nan_to_num(dist, nan=1.0), checks=False), method=linkage_method)
    order = _quasi_diag_order(Z, n)

    def _schur_var(items_a, items_b):
        """Inverse-variance risk of cluster a after Schur-augmenting with b."""
        Caa = cov[np.ix_(items_a, items_a)]
        if g > 0 and len(items_b) > 0:
            Cab = cov[np.ix_(items_a, items_b)]
            Cbb = cov[np.ix_(items_b, items_b)]
            Cbb_inv = np.linalg.pinv(Cbb + np.eye(len(items_b)) * EPSILON)
            Caa = Caa - g * (Cab @ Cbb_inv @ Cab.T)
        Caa = Caa + np.eye(len(items_a)) * EPSILON
        inv = 1.0 / np.maximum(np.diag(Caa), EPSILON)
        wv = inv / inv.sum()
        return float(abs(wv @ Caa @ wv))

    w = np.ones(n)
    clusters = [order]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]
            # Inverse-std split (matches this module's HRP so gamma=0 == HRP).
            sl = np.sqrt(_schur_var(left, right))
            sr = np.sqrt(_schur_var(right, left))
            alpha = (1.0 / sl) / (1.0 / sl + 1.0 / sr) if (sl > 0 and sr > 0) else 0.5
            for i in left:
                w[i] *= alpha
            for i in right:
                w[i] *= (1.0 - alpha)
            if len(left) > 1:
                nxt.append(left)
            if len(right) > 1:
                nxt.append(right)
        clusters = nxt

    w = w / max(w.sum(), EPSILON)
    r = _metrics(w, mu, cov, 0.0, ids)
    r.method = "schur_complementary_allocation"
    r.info["gamma"] = g
    return r


@frame_friendly
def mean_cvar(
    returns: Dict[str, List[float]],
    *,
    confidence: float = 0.95,
    target_return: Optional[float] = None,
    constraints: Optional[Constraints] = None,
) -> OptResult:
    """Minimise Conditional VaR (Expected Shortfall).

    Uses linear programming on empirical scenarios. Better than MVO for
    fat-tailed distributions.

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        confidence: CVaR confidence level (default 0.95).
        target_return: Minimum expected return constraint.
        constraints: Portfolio constraints.
    """
    ids, mu, cov = _build_matrices(returns)
    n = len(ids)
    min_len = min(len(returns[i]) for i in ids)
    R = np.column_stack([np.array(returns[i][-min_len:], dtype=np.float64) for i in ids])
    t = R.shape[0]

    alpha = 1.0 - confidence
    bounds = _bounds(n, constraints)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        cons.append({"type": "ineq", "fun": lambda w: w @ mu - target_return})

    def cvar_obj(w):
        port_ret = R @ w
        sorted_ret = np.sort(port_ret)
        cutoff = max(int(t * alpha), 1)
        es = -np.mean(sorted_ret[:cutoff])
        return es

    x0 = np.ones(n) / n
    result = sp_opt.minimize(cvar_obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                             options={"maxiter": 500, "ftol": 1e-12})
    w = result.x
    r = _metrics(w, mu, cov, 0.0, ids)
    r.method = "mean_cvar"
    r.info["confidence"] = confidence
    return r


@frame_friendly
def robust_mvo(
    returns: Dict[str, List[float]],
    *,
    epsilon: float = 0.1,
    risk_free_rate: float = 0.0,
    constraints: Optional[Constraints] = None,
) -> OptResult:
    """Robust MVO with uncertainty around expected returns.

    Implements a simplified version of Goldfarb & Iyengar (2003):
    maximise ``w'mu - epsilon * ||Sigma^{1/2} w|| - lambda * w'Sigma w``.

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        epsilon: Size of the uncertainty set around mu.
        risk_free_rate: Annual risk-free rate.
        constraints: Portfolio constraints.
    """
    ids, mu, cov = _build_matrices(returns, risk_free_rate)
    n = len(ids)
    bounds = _bounds(n, constraints)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    try:
        L = np.linalg.cholesky(cov + np.eye(n) * EPSILON)
    except np.linalg.LinAlgError:
        L = np.eye(n) * np.sqrt(np.diag(cov) + EPSILON)

    def objective(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        uncertainty = epsilon * np.linalg.norm(L.T @ w)
        return -(ret - uncertainty - 0.5 * vol ** 2)

    x0 = np.ones(n) / n
    result = sp_opt.minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                             options={"maxiter": 500, "ftol": 1e-12})
    w = result.x
    r = _metrics(w, mu, cov, risk_free_rate, ids)
    r.method = "robust_mvo"
    r.info["epsilon"] = epsilon
    return r


@frame_friendly
def max_diversification(
    returns: Dict[str, List[float]],
    *,
    constraints: Optional[Constraints] = None,
) -> OptResult:
    """Maximum Diversification Ratio (Choueifaty & Coignard 2008).

    Maximise DR = w'sigma / sqrt(w'Sigma w).
    """
    ids, mu, cov = _build_matrices(returns)
    n = len(ids)
    sigma = np.sqrt(np.diag(cov))
    bounds = _bounds(n, constraints)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def neg_dr(w):
        port_vol = np.sqrt(w @ cov @ w)
        weighted_vol = w @ sigma
        return -weighted_vol / max(port_vol, EPSILON)

    x0 = np.ones(n) / n
    result = sp_opt.minimize(neg_dr, x0, method="SLSQP", bounds=bounds, constraints=cons,
                             options={"maxiter": 500, "ftol": 1e-12})
    w = result.x
    r = _metrics(w, mu, cov, 0.0, ids)
    r.method = "max_diversification"
    r.info["diversification_ratio"] = round(-neg_dr(w), 4)
    return r


@frame_friendly
def min_tracking_error(
    returns: Dict[str, List[float]],
    benchmark_weights: Dict[str, float],
    *,
    constraints: Optional[Constraints] = None,
) -> OptResult:
    """Minimise tracking error vs benchmark.

    TE = sqrt((w - w_b)' Sigma (w - w_b)).
    """
    ids, mu, cov = _build_matrices(returns)
    n = len(ids)
    w_b = np.array([benchmark_weights.get(a, 0.0) for a in ids], dtype=np.float64)
    bounds = _bounds(n, constraints)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def te(w):
        diff = w - w_b
        return float(diff @ cov @ diff)

    x0 = w_b.copy()
    if np.sum(np.abs(x0)) < EPSILON:
        x0 = np.ones(n) / n
    result = sp_opt.minimize(te, x0, method="SLSQP", bounds=bounds, constraints=cons,
                             options={"maxiter": 500, "ftol": 1e-12})
    w = result.x
    r = _metrics(w, mu, cov, 0.0, ids)
    r.method = "min_tracking_error"
    r.info["tracking_error"] = round(float(np.sqrt((w - w_b) @ cov @ (w - w_b))) * 100, 4)
    return r


@frame_friendly
def turnover_penalized(
    returns: Dict[str, List[float]],
    current_weights: Dict[str, float],
    *,
    turnover_penalty: float = 0.001,
    risk_free_rate: float = 0.0,
    constraints: Optional[Constraints] = None,
) -> OptResult:
    """MVO with turnover penalty.

    Maximise: ``w'mu - (lambda/2) w'Sigma w - kappa |w - w_0|``

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        current_weights: Current portfolio weights.
        turnover_penalty: Cost per unit of turnover (kappa).
        risk_free_rate: Annual risk-free rate.
        constraints: Portfolio constraints.
    """
    ids, mu, cov = _build_matrices(returns, risk_free_rate)
    n = len(ids)
    w0 = np.array([current_weights.get(a, 0.0) for a in ids], dtype=np.float64)
    bounds = _bounds(n, constraints)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w):
        ret = w @ mu
        risk = 0.5 * w @ cov @ w
        cost = turnover_penalty * np.sum(np.abs(w - w0))
        return -(ret - risk - cost)

    x0 = w0.copy() if np.sum(np.abs(w0)) > EPSILON else np.ones(n) / n
    result = sp_opt.minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                             options={"maxiter": 500, "ftol": 1e-12})
    w = result.x
    r = _metrics(w, mu, cov, risk_free_rate, ids)
    r.method = "turnover_penalized"
    r.info["turnover"] = round(float(np.sum(np.abs(w - w0))), 6)
    r.info["turnover_penalty"] = turnover_penalty
    return r


# ── Grinold-Kahn Alpha-Risk-Cost Framework ───────────────────────────

def alpha_risk_cost_optimize(
    alphas: Dict[str, float],
    cov: np.ndarray,
    *,
    current_weights: Optional[Dict[str, float]] = None,
    risk_aversion: float = 1.0,
    cost_model: Optional[Any] = None,
    constraints: Optional[Constraints] = None,
    asset_ids: Optional[List[str]] = None,
) -> ARCResult:
    """The complete alpha-risk-cost optimisation.

    Maximise: ``alpha'w - (lambda/2) w'Sigma w - c(w, w_0)``

    This is THE optimisation every institutional quant desk runs.

    Args:
        alphas: ``{asset_id: expected_alpha}`` (annualised).
        cov: ``(n, n)`` covariance matrix (annualised).
        current_weights: Current weights (for cost computation).
        risk_aversion: Lambda — higher = more risk averse.
        cost_model: Object with ``.estimate(notional, adv, vol)`` method,
                    or None for zero cost.
        constraints: Portfolio constraints.
        asset_ids: Asset ordering matching *cov*.
    """
    if asset_ids is None:
        asset_ids = list(alphas.keys())
    n = len(asset_ids)
    Sigma = np.asarray(cov, dtype=np.float64)[:n, :n]

    alpha_vec = np.array([alphas.get(a, 0.0) for a in asset_ids], dtype=np.float64)
    w0 = np.array([
        (current_weights or {}).get(a, 0.0) for a in asset_ids
    ], dtype=np.float64)

    c = constraints or Constraints()
    lo = 0.0 if c.long_only else c.min_weight
    bounds = [(lo, c.max_weight)] * n
    cons_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w):
        alpha_term = alpha_vec @ w
        risk_term = 0.5 * risk_aversion * w @ Sigma @ w
        cost_term = 0.0
        if cost_model is not None and hasattr(cost_model, "estimate"):
            for j in range(n):
                cost_term += cost_model.estimate(abs(w[j] - w0[j]), 0, 0)
        else:
            cost_term = 0.001 * np.sum(np.abs(w - w0))
        return -(alpha_term - risk_term - cost_term)

    x0 = w0.copy() if np.sum(np.abs(w0)) > EPSILON else np.ones(n) / n
    result = sp_opt.minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons_list,
                             options={"maxiter": 500, "ftol": 1e-12})
    w = result.x

    expected_alpha = float(alpha_vec @ w)
    expected_risk = float(np.sqrt(w @ Sigma @ w))
    expected_cost = float(np.sum(np.abs(w - w0)) * 0.001)
    ir = expected_alpha / max(expected_risk, EPSILON)

    unconstrained_w = np.linalg.solve(risk_aversion * Sigma + np.eye(n) * EPSILON, alpha_vec)
    unconstrained_w /= max(np.sum(np.abs(unconstrained_w)), EPSILON)
    tc_num = np.corrcoef(alpha_vec, w * alpha_vec)[0, 1] if n > 1 else 1.0
    tc_den = np.corrcoef(alpha_vec, unconstrained_w * alpha_vec)[0, 1] if n > 1 else 1.0
    tc = tc_num / max(abs(tc_den), EPSILON)

    return ARCResult(
        optimal_weights={asset_ids[j]: round(float(w[j]), 6) for j in range(n)},
        expected_alpha=round(expected_alpha * 100, 4),
        expected_risk=round(expected_risk * 100, 4),
        expected_cost=round(expected_cost * 100, 4),
        information_ratio=round(ir, 4),
        transfer_coefficient=round(float(tc), 4),
    )


# ── Quantum-Inspired Optimisers ──────────────────────────────────────

@dataclass
class QUBOResult(OptResult):
    """Result from QUBO-based optimisation with additional diagnostics."""
    selected_assets: List[str] = field(default_factory=list)
    objective_value: float = 0.0


def _quasi_block_diag_objective(similarity: np.ndarray, perm: np.ndarray) -> float:
    """Compute W(A) = sum_ij A_ij * (pi(i) - pi(j))^2 for a given permutation."""
    n = len(perm)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += similarity[i, j] * (perm[i] - perm[j]) ** 2
    return total


def _rcm_permutation(similarity: np.ndarray) -> np.ndarray:
    """Find a good permutation via Reverse Cuthill-McKee bandwidth reduction."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    sparse_sim = csr_matrix(similarity)
    perm = reverse_cuthill_mckee(sparse_sim, symmetric_mode=True)
    return perm


def _brute_force_permutation(similarity: np.ndarray) -> np.ndarray:
    """Find the exact optimal permutation by enumeration. Feasible for N <= 10."""
    import itertools

    n = similarity.shape[0]
    if n > 10:
        raise ValueError(f"Brute-force permutation infeasible for n={n}. Use 'heuristic'.")

    best_perm = np.arange(n)
    best_obj = _quasi_block_diag_objective(similarity, best_perm)

    for perm_tuple in itertools.permutations(range(n)):
        perm = np.array(perm_tuple)
        obj = _quasi_block_diag_objective(similarity, perm)
        if obj < best_obj:
            best_obj = obj
            best_perm = perm.copy()

    return best_perm


def build_permutation_qubo(similarity: np.ndarray, penalty: float = 10.0) -> np.ndarray:
    """Build the N^2 QUBO matrix for the permutation optimisation problem.

    Binary variables x_{i,i'} = 1 iff asset i is placed at position i'.
    Objective: minimise W(A) subject to permutation constraints.

    This is the QUBO that can be submitted to a quantum annealer.

    Args:
        similarity: (N, N) similarity matrix (typically |C_ij|).
        penalty: Constraint penalty weight.

    Returns:
        (N^2, N^2) QUBO matrix.
    """
    n = similarity.shape[0]
    size = n * n
    Q = np.zeros((size, size))

    def idx(i: int, ip: int) -> int:
        return i * n + ip

    for i in range(n):
        for j in range(n):
            for ip in range(n):
                for jp in range(n):
                    qi = idx(i, ip)
                    qj = idx(j, jp)
                    Q[qi, qj] += similarity[i, j] * (ip - jp) ** 2

    for i in range(n):
        for ip in range(n):
            qi = idx(i, ip)
            Q[qi, qi] += penalty * (1 - 2)
            for jp in range(n):
                if ip != jp:
                    Q[qi, idx(i, jp)] += penalty

    for ip in range(n):
        for i in range(n):
            qi = idx(i, ip)
            Q[qi, qi] += penalty * (1 - 2)
            for j in range(n):
                if i != j:
                    Q[qi, idx(j, ip)] += penalty

    Q = (Q + Q.T) / 2.0
    return Q


def _decode_permutation_from_qubo(x: np.ndarray, n: int) -> np.ndarray:
    """Decode a permutation from the N^2 binary QUBO solution vector."""
    X = x.reshape(n, n)
    perm = np.zeros(n, dtype=int)
    used_positions = set()

    for i in range(n):
        row = X[i]
        candidates = np.argsort(-row)
        for pos in candidates:
            if pos not in used_positions:
                perm[i] = pos
                used_positions.add(pos)
                break

    remaining_positions = set(range(n)) - used_positions
    for i in range(n):
        if perm[i] in used_positions and i not in {
            j for j in range(n) if perm[j] in used_positions
        }:
            perm[i] = remaining_positions.pop()

    return perm


def _optimal_bisect(similarity: np.ndarray, indices: List[int]) -> Tuple[List[int], List[int]]:
    """Find the best split point that minimises off-diagonal coupling."""
    n = len(indices)
    if n <= 1:
        return indices, []

    best_split = n // 2
    best_cost = float("inf")

    for split in range(1, n):
        left = indices[:split]
        right = indices[split:]

        cost = 0.0
        for li in left:
            for ri in right:
                cost += abs(similarity[li, ri])

        avg_cost = cost / max(len(left) * len(right), 1)
        if avg_cost < best_cost:
            best_cost = avg_cost
            best_split = split

    return indices[:best_split], indices[best_split:]


@frame_friendly
def quantum_inspired_hrp(
    returns: Dict[str, List[float]],
    *,
    backend: str = "heuristic",
    solver: Optional[Any] = None,
) -> OptResult:
    """Quantum-Inspired Hierarchical Risk Parity (1QBit, Alipour et al. 2016).

    Improves on standard HRP by optimising the asset permutation to minimise
    information loss during hierarchical bisection. The permutation problem
    is formulated as a QUBO with N^2 binary variables, solvable on quantum
    annealers (e.g. D-Wave via Amazon Braket).

    Three-step process:
        1. **Quasi-block-diagonalisation**: find permutation pi that minimises
           W(A) = sum_ij |C_ij| * (pi(i) - pi(j))^2
        2. **Recursive bisection**: split reordered assets at the point that
           minimises off-diagonal coupling
        3. **Inverse-variance weight allocation**: walk the tree, scaling
           cluster weights by 1/cluster_variance

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        backend: ``"heuristic"`` (Reverse Cuthill-McKee, default),
                 ``"brute_force"`` (exact, N<=10),
                 ``"simulated_annealing"`` (dwave-neal, ``pip install cpz-ai[quantum]``),
                 ``"braket"`` (Amazon Braket, ``pip install cpz-ai[quantum-braket]``).
        solver: Optional custom :class:`QUBOSolver` instance. Overrides *backend*.

    Returns:
        OptResult with method ``"quantum_inspired_hrp"``.
    """
    ids, mu, cov = _build_matrices(returns)
    n = len(ids)

    if n <= 1:
        r = _metrics(np.ones(1), mu, cov, 0.0, ids)
        r.method = "quantum_inspired_hrp"
        return r

    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = cov / np.outer(std + EPSILON, std + EPSILON)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1, 1)
    similarity = np.abs(corr)

    if solver is not None:
        Q = build_permutation_qubo(similarity)
        x = solver.solve(Q)
        sorted_idx = list(_decode_permutation_from_qubo(x, n))
    elif backend == "heuristic":
        sorted_idx = list(_rcm_permutation(similarity))
    elif backend == "brute_force":
        perm = _brute_force_permutation(similarity)
        sorted_idx = list(perm)
    elif backend in ("simulated_annealing", "braket"):
        from .quantum_backends import get_solver
        qubo_solver = get_solver(backend)
        Q = build_permutation_qubo(similarity)
        x = qubo_solver.solve(Q)
        sorted_idx = list(_decode_permutation_from_qubo(x, n))
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Use: 'heuristic', 'brute_force', 'simulated_annealing', or 'braket'."
        )

    def _cluster_var_q(cov_sub: np.ndarray) -> float:
        n_sub = cov_sub.shape[0]
        if n_sub == 1:
            return float(cov_sub[0, 0])
        inv_diag = 1.0 / np.maximum(np.diag(cov_sub), EPSILON)
        w_sub = inv_diag / np.sum(inv_diag)
        return float(w_sub @ cov_sub @ w_sub)

    def _recursive_bisect_qhrp(
        cov_mat: np.ndarray, sim_mat: np.ndarray, ordered: List[int],
    ) -> np.ndarray:
        w = np.ones(len(ordered))

        clusters: List[List[int]] = [list(range(len(ordered)))]
        while clusters:
            new_clusters: List[List[int]] = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                orig_indices = [ordered[i] for i in cluster]
                sim_sub = sim_mat[np.ix_(orig_indices, orig_indices)]
                local_indices = list(range(len(cluster)))

                left_local, right_local = _optimal_bisect(sim_sub, local_indices)
                if not left_local or not right_local:
                    continue

                left_cluster = [cluster[i] for i in left_local]
                right_cluster = [cluster[i] for i in right_local]

                left_orig = [ordered[i] for i in left_cluster]
                right_orig = [ordered[i] for i in right_cluster]

                cov_left = cov_mat[np.ix_(left_orig, left_orig)]
                cov_right = cov_mat[np.ix_(right_orig, right_orig)]

                var_left = max(_cluster_var_q(cov_left), EPSILON)
                var_right = max(_cluster_var_q(cov_right), EPSILON)

                inv_left = 1.0 / np.sqrt(var_left)
                inv_right = 1.0 / np.sqrt(var_right)
                alpha = inv_left / (inv_left + inv_right)

                for i in left_cluster:
                    w[i] *= alpha
                for i in right_cluster:
                    w[i] *= (1.0 - alpha)

                if len(left_cluster) > 1:
                    new_clusters.append(left_cluster)
                if len(right_cluster) > 1:
                    new_clusters.append(right_cluster)
            clusters = new_clusters
        return w

    hrp_w = _recursive_bisect_qhrp(cov, similarity, sorted_idx)

    final_w = np.zeros(n)
    for i, orig_idx in enumerate(sorted_idx):
        final_w[orig_idx] = hrp_w[i]
    final_w /= max(np.sum(final_w), EPSILON)

    r = _metrics(final_w, mu, cov, 0.0, ids)
    r.method = "quantum_inspired_hrp"
    r.info["backend"] = backend if solver is None else "custom_solver"
    r.info["permutation"] = sorted_idx
    return r


def build_portfolio_qubo(
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    risk_aversion: float = 0.5,
    budget_penalty: float = 10.0,
    target_k: int = 3,
) -> np.ndarray:
    """Build QUBO matrix for binary portfolio selection.

    Binary decision: x_i in {0,1} = whether to include asset i.

    Objective: minimise ``risk_aversion * x^T cov x - mu^T x
                          + budget_penalty * (sum(x) - target_k)^2``

    The returned matrix can be submitted to any :class:`QUBOSolver`,
    including :class:`BraketSolver` for real quantum hardware.

    Args:
        mu: (N,) expected returns.
        cov: (N, N) covariance matrix.
        risk_aversion: Weight on risk term.
        budget_penalty: Penalty for deviating from target_k selected assets.
        target_k: Target number of assets to select.

    Returns:
        (N, N) QUBO matrix.
    """
    n = len(mu)
    Q = np.zeros((n, n))

    Q += risk_aversion * cov

    for i in range(n):
        Q[i, i] -= mu[i]

    for i in range(n):
        Q[i, i] += budget_penalty * (1 - 2 * target_k)
        for j in range(i + 1, n):
            Q[i, j] += 2 * budget_penalty

    Q = (Q + Q.T) / 2.0
    return Q


@frame_friendly
def qubo_portfolio_selection(
    returns: Dict[str, List[float]],
    *,
    target_k: int = 5,
    risk_aversion: float = 0.5,
    budget_penalty: float = 10.0,
    backend: str = "brute_force",
    solver: Optional[Any] = None,
) -> QUBOResult:
    """Cardinality-constrained portfolio selection via QUBO.

    Selects exactly *target_k* assets from the universe by solving a
    Quadratic Unconstrained Binary Optimisation problem. The QUBO can
    run on classical solvers or real quantum hardware via Amazon Braket.

    After selection, allocates equally among chosen assets (use
    :func:`quantum_inspired_hrp` or :func:`risk_parity` on the
    selected subset for smarter weight allocation).

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        target_k: Number of assets to select.
        risk_aversion: Weight on risk vs return in objective.
        budget_penalty: Penalty for violating cardinality constraint.
        backend: ``"brute_force"`` (exact, N<=20),
                 ``"simulated_annealing"`` (``pip install cpz-ai[quantum]``),
                 ``"braket"`` (``pip install cpz-ai[quantum-braket]``).
        solver: Optional custom :class:`QUBOSolver` instance. Overrides *backend*.

    Returns:
        QUBOResult with selected assets and equal weights among them.
    """
    ids, mu, cov = _build_matrices(returns)
    n = len(ids)
    target_k = min(target_k, n)

    Q = build_portfolio_qubo(mu, cov, risk_aversion=risk_aversion,
                             budget_penalty=budget_penalty, target_k=target_k)

    if solver is not None:
        x = solver.solve(Q)
    elif backend == "brute_force":
        from .quantum_backends import ClassicalBruteForceSolver
        x = ClassicalBruteForceSolver().solve(Q)
    elif backend in ("simulated_annealing", "braket"):
        from .quantum_backends import get_solver
        qubo_solver = get_solver(backend)
        x = qubo_solver.solve(Q)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Use: 'brute_force', 'simulated_annealing', or 'braket'."
        )

    selected_mask = x.astype(bool)
    selected = [ids[i] for i in range(n) if selected_mask[i]]

    if not selected:
        top_k = np.argsort(-mu)[:target_k]
        selected = [ids[i] for i in top_k]
        selected_mask = np.zeros(n, dtype=bool)
        for i in top_k:
            selected_mask[i] = True

    w_equal = 1.0 / len(selected)
    weights = {aid: round(w_equal, 6) if aid in selected else 0.0 for aid in ids}

    final_w = np.array([weights[aid] for aid in ids])
    obj_val = float(x @ Q @ x)

    r = QUBOResult(
        weights=weights,
        selected_assets=selected,
        objective_value=round(obj_val, 6),
    )
    ret = float(final_w @ mu)
    vol = float(np.sqrt(final_w @ cov @ final_w))
    r.expected_return = round(ret * 100, 4)
    r.volatility = round(vol * 100, 4)
    r.sharpe_ratio = round(ret / max(vol, EPSILON), 4)
    r.method = "qubo_portfolio_selection"
    r.info["target_k"] = target_k
    r.info["actual_k"] = len(selected)
    r.info["backend"] = backend if solver is None else "custom_solver"
    r.info["risk_aversion"] = risk_aversion
    r.info["budget_penalty"] = budget_penalty
    return r
