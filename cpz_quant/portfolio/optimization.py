"""Portfolio optimization: MVO, Black-Litterman, HRP, mean-CVaR, robust, max-diversification,
tracking error, turnover-penalized, and the Grinold-Kahn alpha-risk-cost framework.

All functions are pure: data in, results out. No DB access, no API calls.
Requires numpy (core dep) and scipy (core dep).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    """Requested limits; supported fields depend on the chosen optimizer.

    The core mean/min-variance and max-Sharpe functions enforce full investment,
    weight bounds and gross exposure. They reject unsupported optional limits.
    """
    long_only: bool = False
    max_weight: float = 1.0
    min_weight: float = -1.0
    max_gross_exposure: Optional[float] = 2.0
    max_net_exposure: Optional[float] = 1.0
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
    if not ids:
        raise ValueError("returns must contain at least one asset")
    arrays = [np.asarray(returns[i], dtype=np.float64) for i in ids]
    for aid, values in zip(ids, arrays):
        if values.ndim != 1 or values.size < 2:
            raise ValueError(f"returns for {aid!r} must be a 1D series with at least two observations")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"returns for {aid!r} must contain only finite observations")
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")
    min_len = min(len(values) for values in arrays)
    R = np.column_stack([values[-min_len:] for values in arrays])
    with np.errstate(all="ignore"):
        mu = np.mean(R, axis=0) * TRADING_DAYS
        cov = np.cov(R.T, ddof=1) * TRADING_DAYS
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    if not np.all(np.isfinite(cov)) or not np.all(np.isfinite(mu)):
        raise ValueError("returns must produce finite means and covariance")
    return ids, mu, cov


def _bounds(n: int, constraints: Optional[Constraints] = None):
    c = constraints or Constraints()
    if not np.isfinite(c.min_weight) or not np.isfinite(c.max_weight):
        raise ValueError("min_weight and max_weight must be finite")
    lo = max(0.0, c.min_weight) if c.long_only else c.min_weight
    hi = c.max_weight
    if lo > hi:
        raise ValueError("effective min_weight must not exceed max_weight")
    return [(lo, hi)] * n


def _check_limits(c: Constraints, label: str, *, allow_turnover: bool = False) -> None:
    """Reject limits a fully invested native optimizer cannot enforce.

    Sector and factor limits need a sector map or factor loadings and a
    tracking-error limit needs benchmark weights, none of which
    :class:`Constraints` carries, so they raise instead of being dropped.
    """
    unsupported = ["sector_limits", "factor_limits", "max_tracking_error"]
    if not allow_turnover:
        unsupported.append("max_turnover")
    for name in unsupported:
        if getattr(c, name) is not None:
            raise NotImplementedError(f"{label}: {name} is not supported by this optimizer")
    for name in ("max_gross_exposure", "max_net_exposure"):
        limit = getattr(c, name)
        if limit is not None and (not np.isfinite(limit) or limit < 1.0):
            raise ValueError(f"{label}: {name} must be finite and >= 1 for full investment")
    if allow_turnover and c.max_turnover is not None and (
        not np.isfinite(c.max_turnover) or c.max_turnover < 0
    ):
        raise ValueError(f"{label}: max_turnover must be finite and >= 0")


def _solve_weights(
    objective: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    c: Constraints,
    x0: np.ndarray,
    *,
    jac: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    extra_constraints: Sequence[Dict[str, Any]] = (),
    trades_from: Optional[np.ndarray] = None,
    trade_cost: Optional[Callable[[np.ndarray], float]] = None,
    trade_cost_jac: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[Any, np.ndarray]:
    """SLSQP over fully invested weights with smooth exposure and trade terms.

    Absolute values enter through epigraph variables so every constraint
    stays differentiable: ``s >= |w|`` with ``sum(s) <= max_gross_exposure``
    when a negative lower bound lets the gross cap bind, and ``t >= |w - w0|``
    for trade costs (``trade_cost(t)``) and ``max_turnover``. When every lower
    bound is >= 0, full investment already fixes gross exposure at 1, so the
    cap is not added: a duplicate of the budget row stalls SLSQP short of the
    optimum. ``extra_constraints`` are SLSQP dicts on the weights.

    Returns the raw solver result and the weight vector.
    """
    n = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    gross = c.max_gross_exposure is not None and bool(np.any(lo < 0))
    trades = trades_from is not None
    k = n * (1 + int(gross) + int(trades))
    eye = np.eye(n)

    def block(*parts: Tuple[int, np.ndarray]) -> np.ndarray:
        """Place (offset, matrix) blocks into a row block of width k."""
        rows = parts[0][1].shape[0]
        out = np.zeros((rows, k))
        for offset, mat in parts:
            out[:, offset:offset + mat.shape[1]] = mat
        return out

    s_at = n
    t_at = n * (1 + int(gross))
    cons: List[Dict[str, Any]] = [{
        "type": "eq",
        "fun": lambda x: np.sum(x[:n]) - 1.0,
        "jac": lambda x: block((0, np.ones((1, n))))[0],
    }]
    for con in extra_constraints:
        fun, cjac = con["fun"], con.get("jac")
        wrapped: Dict[str, Any] = {"type": con["type"], "fun": (lambda x, f=fun: f(x[:n]))}
        if cjac is not None:
            wrapped["jac"] = lambda x, g=cjac: block((0, np.atleast_2d(g(x[:n]))))
        cons.append(wrapped)
    if gross:
        cap = float(c.max_gross_exposure)  # type: ignore[arg-type]
        cons.append({
            "type": "ineq",
            "fun": lambda x: np.concatenate([x[s_at:s_at + n] - x[:n], x[s_at:s_at + n] + x[:n]]),
            "jac": lambda x: np.vstack([block((0, -eye), (s_at, eye)), block((0, eye), (s_at, eye))]),
        })
        cons.append({
            "type": "ineq",
            "fun": lambda x: cap - np.sum(x[s_at:s_at + n]),
            "jac": lambda x: block((s_at, -np.ones((1, n))))[0],
        })
    if trades:
        ref = np.asarray(trades_from, dtype=float)
        cons.append({
            "type": "ineq",
            "fun": lambda x: np.concatenate([
                x[t_at:t_at + n] - (x[:n] - ref), x[t_at:t_at + n] + (x[:n] - ref)
            ]),
            "jac": lambda x: np.vstack([block((0, -eye), (t_at, eye)), block((0, eye), (t_at, eye))]),
        })
        if c.max_turnover is not None:
            limit = float(c.max_turnover)
            cons.append({
                "type": "ineq",
                "fun": lambda x: limit - np.sum(x[t_at:t_at + n]),
                "jac": lambda x: block((t_at, -np.ones((1, n))))[0],
            })

    def full_objective(x: np.ndarray) -> float:
        value = objective(x[:n])
        if trade_cost is not None:
            value += trade_cost(x[t_at:t_at + n])
        return float(value)

    full_jac = None
    if jac is not None and (trade_cost is None or trade_cost_jac is not None):
        def full_jac(x: np.ndarray) -> np.ndarray:
            g = np.zeros(k)
            g[:n] = jac(x[:n])
            if trade_cost_jac is not None:
                g[t_at:t_at + n] = trade_cost_jac(x[t_at:t_at + n])
            return g

    w0 = np.asarray(x0, dtype=float)
    start = [w0]
    all_bounds: List[Tuple[float, Optional[float]]] = list(bounds)
    if gross:
        start.append(np.abs(w0))
        all_bounds += [(0.0, float(max(abs(a), abs(b)))) for a, b in zip(lo, hi)]
    if trades:
        start.append(np.abs(w0 - np.asarray(trades_from, dtype=float)))
        all_bounds += [(0.0, None)] * n
    result = sp_opt.minimize(full_objective, np.concatenate(start), jac=full_jac,
                             method="SLSQP", bounds=all_bounds, constraints=cons,
                             options={"maxiter": 1000, "ftol": 1e-12})
    w = np.asarray(result.x, dtype=float)[:n]
    return result, w


def _core_result(result, w, bounds, c, label, mu, cov, rf, ids, target_return=None,
                 min_return=None):
    """Never normalize or substitute a failed or infeasible solver iterate."""
    if not result.success:
        raise RuntimeError(f"{label}: optimization failed: {result.message}")
    w = np.asarray(w, dtype=float)
    tolerance = 1e-7
    if w.shape != (len(ids),) or not np.all(np.isfinite(w)):
        raise RuntimeError(f"{label}: solver returned invalid weights")
    if abs(float(w.sum()) - 1.0) > tolerance:
        raise RuntimeError(f"{label}: solver weights do not sum to one")
    lower, upper = np.asarray(bounds, dtype=float).T
    if np.any(w < lower - tolerance) or np.any(w > upper + tolerance):
        raise RuntimeError(f"{label}: solver weights violate the requested bounds")
    if c.max_gross_exposure is not None and np.abs(w).sum() > c.max_gross_exposure + tolerance:
        raise RuntimeError(f"{label}: solver weights violate max_gross_exposure")
    if target_return is not None and abs(float(w @ mu) - target_return) > tolerance:
        raise RuntimeError(f"{label}: solver weights do not achieve target_return")
    if min_return is not None and float(w @ mu) < min_return - tolerance:
        raise RuntimeError(
            f"{label}: solver weights return {float(w @ mu):.6f}, below the "
            f"target_return {min_return:.6f}"
        )
    r = _metrics(w, mu, cov, rf, ids)
    # Rounding each position can break a tight bound or full investment.
    r.weights = {aid: float(weight) for aid, weight in zip(ids, w)}
    r.method = label
    return r


def _metrics(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float, ids: List[str]) -> OptResult:
    if not np.all(np.isfinite(w)):
        # A NaN weight would otherwise be reported with Sharpe 0.0.
        raise RuntimeError("allocation produced non-finite weights")
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

    Minimizes variance, optionally at an exact annualized decimal
    *target_return*. Without a target this is global minimum variance;
    use :func:`max_sharpe` for a tangency portfolio. Supports full investment,
    weight bounds and gross exposure; unsupported optional limits raise.
    """
    ids, mu, cov = _build_matrices(returns, risk_free_rate)
    n = len(ids)
    c = constraints or Constraints()
    bounds = _bounds(n, c)
    _check_limits(c, "mean_variance")
    extra = []
    if target_return is not None:
        if not np.isfinite(target_return):
            raise ValueError("target_return must be finite")
        extra.append({"type": "eq", "fun": lambda w: w @ mu - target_return,
                      "jac": lambda w: mu})

    result, w = _solve_weights(lambda w: float(w @ cov @ w), bounds, c, np.ones(n) / n,
                               jac=lambda w: 2.0 * cov @ w, extra_constraints=extra)
    return _core_result(result, w, bounds, c, "mean_variance", mu, cov,
                        risk_free_rate, ids, target_return)


@frame_friendly
def min_variance(
    returns: Dict[str, List[float]],
    *,
    constraints: Optional[Constraints] = None,
) -> OptResult:
    """Global minimum variance with full investment, weight bounds and gross cap.

    Unsupported optional constraints and failed/infeasible solver results raise.
    """
    ids, mu, cov = _build_matrices(returns)
    n = len(ids)
    c = constraints or Constraints()
    bounds = _bounds(n, c)
    _check_limits(c, "min_variance")
    result, w = _solve_weights(lambda w: float(w @ cov @ w), bounds, c, np.ones(n) / n,
                               jac=lambda w: 2.0 * cov @ w)
    return _core_result(result, w, bounds, c, "min_variance", mu, cov, 0.0, ids)


@frame_friendly
def max_sharpe(
    returns: Dict[str, List[float]],
    *,
    risk_free_rate: float = 0.0,
    constraints: Optional[Constraints] = None,
) -> OptResult:
    """Maximum Sharpe ratio with full investment, weight bounds and gross cap.

    Unsupported optional constraints and failed/infeasible solver results raise.
    """
    ids, mu, cov = _build_matrices(returns, risk_free_rate)
    n = len(ids)
    c = constraints or Constraints()
    bounds = _bounds(n, c)
    _check_limits(c, "max_sharpe")

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret - risk_free_rate) / max(vol, EPSILON)

    result, w = _solve_weights(neg_sharpe, bounds, c, np.ones(n) / n)
    return _core_result(result, w, bounds, c, "max_sharpe", mu, cov, risk_free_rate, ids)


@frame_friendly
def risk_parity(
    returns: Dict[str, List[float]],
    *,
    budget: Optional[Dict[str, float]] = None,
    max_iter: int = 500,
    tolerance: float = 1e-10,
) -> OptResult:
    """Long-only risk parity: equal (or budgeted) risk contribution.

    Solves ``x_i (Sigma x)_i = b_i`` for ``x > 0`` by cyclical coordinate
    descent on the convex program ``min 0.5 x'Sigma x - sum b_i ln x_i``
    (Spinu 2013; Griveau-Billion, Richard & Roncalli 2013), then normalises to
    ``w = x / sum(x)``. The solution exists and is unique for any covariance
    with positive variances, including negatively correlated pairs such as
    stocks and Treasuries; for two assets it is the inverse-volatility split
    ``w_1 = sigma_2 / (sigma_1 + sigma_2)``.

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        budget: ``{asset_id: risk_budget}`` with a positive entry for every
            asset (normalised to sum to one); ``None`` for equal budgets.
        max_iter: Maximum coordinate-descent sweeps.
        tolerance: Convergence tolerance on the relative change in ``x``.

    Raises:
        ValueError: for an invalid budget or a zero-variance asset.
        RuntimeError: when the sweeps do not converge within *max_iter*.
    """
    ids, mu, cov = _build_matrices(returns)
    n = len(ids)

    if budget is None:
        target = np.ones(n) / n
    else:
        unknown = sorted(set(budget) - set(ids))
        missing = [aid for aid in ids if aid not in budget]
        if unknown or missing:
            raise ValueError(
                f"risk_parity: budget must give every asset a value "
                f"(missing {missing}, unknown {unknown})"
            )
        target = np.array([float(budget[aid]) for aid in ids])
        if not np.all(np.isfinite(target)) or np.any(target <= 0):
            raise ValueError("risk_parity: every risk budget must be positive and finite")
        target = target / target.sum()

    var = np.diag(cov)
    if np.any(var <= 0):
        zero = [aid for aid, v in zip(ids, var) if v <= 0]
        raise ValueError(f"risk_parity: {zero} have zero return variance")

    x = target / np.sqrt(var)
    converged = False
    for sweep in range(max_iter):
        previous = x.copy()
        for i in range(n):
            # (Sigma x)_i without the diagonal term, using the latest coordinates.
            c_i = float(cov[i] @ x) - cov[i, i] * x[i]
            x[i] = (-c_i + np.sqrt(c_i * c_i + 4.0 * cov[i, i] * target[i])) / (2.0 * cov[i, i])
        if np.max(np.abs(x - previous)) <= tolerance * np.max(np.abs(x)):
            converged = True
            break
    if not converged:
        raise RuntimeError(f"risk_parity: did not converge in {max_iter} sweeps")

    w = x / x.sum()
    contributions = w * (cov @ w)
    r = _metrics(w, mu, cov, 0.0, ids)
    r.weights = {aid: float(weight) for aid, weight in zip(ids, w)}
    r.method = "risk_parity"
    r.info["risk_contributions"] = {
        aid: float(rc) for aid, rc in zip(ids, contributions / contributions.sum())
    }
    r.info["iterations"] = sweep + 1
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


def _single_asset_result(mu: np.ndarray, cov: np.ndarray, ids: List[str], method: str) -> OptResult:
    r = _metrics(np.ones(1), mu, cov, 0.0, ids)
    r.method = method
    return r


@frame_friendly
def hierarchical_risk_parity(
    returns: Dict[str, List[float]],
    *,
    linkage_method: str = "single",
) -> OptResult:
    """Hierarchical Risk Parity (Lopez de Prado 2016).

    No covariance inversion, so it is robust to estimation error.

    1. Distance ``sqrt((1 - rho) / 2)`` from the correlation matrix
    2. Hierarchical clustering
    3. Quasi-diagonalise (dendrogram leaf order)
    4. Recursive bisection: each half of the ordered list gets
       ``1 - V_half / (V_left + V_right)`` of its parent's weight, where ``V`` is
       the variance of the half's inverse-variance portfolio. This is the
       paper's split and matches PyPortfolioOpt's ``HRPOpt``.

    A single asset gets 100%.

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        linkage_method: ``"single"`` (default), ``"complete"``, ``"average"``,
                        or ``"ward"``.
    """
    from scipy.cluster.hierarchy import linkage as sp_linkage
    from scipy.spatial.distance import squareform

    ids, mu, cov = _build_matrices(returns)
    n = len(ids)
    if n == 1:
        r = _single_asset_result(mu, cov, ids, "hierarchical_risk_parity")
        r.info["linkage_method"] = linkage_method
        return r

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
    sorted_idx = _quasi_diag_order(Z, n)
    final_w = _recursive_bisection(cov, sorted_idx)
    final_w /= final_w.sum()

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


def _inverse_variance_split(v_left: float, v_right: float) -> float:
    """Share of the left half: ``1 - V_l / (V_l + V_r)`` (Lopez de Prado 2016).

    A zero-variance half takes all of its parent's weight (the limit of the
    rule); two zero-variance halves have no defined split and raise.
    """
    total = v_left + v_right
    if not total > 0:
        raise ValueError("recursive bisection: both halves have zero variance")
    return v_right / total


def _recursive_bisection(cov_mat: np.ndarray, order: list) -> np.ndarray:
    """Inverse-variance recursive bisection over `order` (indices into cov_mat)."""
    w = np.ones(cov_mat.shape[0])
    clusters = [list(order)]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]
            a = _inverse_variance_split(_cluster_variance(cov_mat, left),
                                        _cluster_variance(cov_mat, right))
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

    Cuts the dendrogram into ``K`` clusters, then walks the tree top-down: at
    each of the ``K - 1`` splits above the cut, the left branch gets
    ``1 - R_left / (R_left + R_right)`` of its parent's weight, where ``R`` is
    the summed variance of the inverse-variance portfolios of the clusters in
    that branch. Within each cluster the weights are inverse-variance. The
    splits follow the tree, not the halves of the ordered cluster list. Matches
    riskfolio-lib's ``HCPortfolio`` with ``model="HERC"`` and ``rm="MV"``.

    The number of clusters is chosen from the largest gap in the merge
    heights unless given.

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        linkage_method: scipy linkage ("ward" default, "single", "average", ...).
        n_clusters: force a cluster count in ``[1, N]``; ``None`` auto-selects.
    """
    from scipy.cluster.hierarchy import linkage as sp_linkage
    from scipy.cluster.hierarchy import to_tree
    from scipy.spatial.distance import squareform

    ids, mu, cov = _build_matrices(returns)
    n = len(ids)
    if n == 1:
        return _single_asset_result(mu, cov, ids, "hierarchical_equal_risk_contribution")
    if n_clusters is not None and not 1 <= int(n_clusters) <= n:
        raise ValueError(f"n_clusters must be in [1, {n}], got {n_clusters}")

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
        K = int(n_clusters)

    # Split the highest remaining merge until K clusters remain.
    root = to_tree(Z)
    frontier = [root]
    splits = []
    while len(frontier) < K:
        node = max((f for f in frontier if not f.is_leaf()), key=lambda f: (f.dist, f.id))
        frontier.remove(node)
        frontier += [node.get_left(), node.get_right()]
        splits.append(node)
    clusters = [f.pre_order() for f in frontier]

    within = np.zeros(n)
    cluster_var = []
    for idx in clusters:
        iv = 1.0 / np.maximum(np.diag(cov)[idx], EPSILON)
        within[idx] = iv / iv.sum()
        cluster_var.append(float(within[idx] @ cov[np.ix_(idx, idx)] @ within[idx]))

    w = np.ones(n)
    for node in splits:
        left, right = node.get_left().pre_order(), node.get_right().pre_order()
        left_set = set(left)
        left_risk = sum(v for idx, v in zip(clusters, cluster_var)
                        if idx[0] in left_set)
        right_set = set(right)
        right_risk = sum(v for idx, v in zip(clusters, cluster_var)
                         if idx[0] in right_set)
        a = _inverse_variance_split(left_risk, right_risk)
        w[left] *= a
        w[right] *= 1.0 - a
    w *= within
    w /= w.sum()

    r = _metrics(w, mu, cov, 0.0, ids)
    r.method = "hierarchical_equal_risk_contribution"
    r.info["linkage_method"] = linkage_method
    r.info["n_clusters"] = int(len(clusters))
    return r


def _analytic_min_variance(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    inv = np.linalg.pinv(cov + np.eye(n) * EPSILON)
    w = inv @ np.ones(n)
    s = w.sum()
    if not s > 0:
        raise ValueError("min_variance: covariance is not positive definite")
    return w / s


def _tangency_weights(cov: np.ndarray, mu: np.ndarray, where: str) -> np.ndarray:
    """``Sigma^-1 mu`` scaled to sum to one, whatever the sign of ``1'Sigma^-1 mu``.

    Inside NCO a cluster whose sum is negative gets the flipped direction, but
    that is not final: the cluster-level step can weight the cluster
    negatively and flip it back. Whether a fully invested maximum-Sharpe
    portfolio exists is decided by the cluster-level problem, see
    :func:`nested_clustered_optimization`. Raises only when the sum is zero
    and no scaling to full investment exists.
    """
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)  # the only fully invested portfolio
    inv = np.linalg.pinv(cov + np.eye(n) * EPSILON)
    w = inv @ mu
    s = float(w.sum())
    if not abs(s) > 1e-12 * float(np.abs(w).sum()):
        raise ValueError(
            f"{where} has 1'Sigma^-1 mu = 0, so its maximum-Sharpe weights cannot be "
            "scaled to full investment"
        )
    return w / s


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

    Raises:
        ValueError: for ``"max_sharpe"`` when the cluster-level problem has no
            fully invested tangency portfolio: ``1'C^-1 m <= 0`` for the
            cluster portfolios' covariance ``C`` and means ``m`` (their
            minimum-variance mix does not earn more than the zero risk-free
            rate, as in a falling market). Normalising there would return the
            lowest-Sharpe mix, so no weights are returned. A single cluster
            whose own ``1'Sigma^-1 mu`` is negative is fine: the cluster-level
            step weights it negatively, which flips it back to its best
            direction.
    """
    if objective not in ("min_variance", "max_sharpe"):
        raise ValueError(f"objective must be 'min_variance' or 'max_sharpe', got {objective!r}")
    ids, mu, cov = _build_matrices(returns)
    n = len(ids)
    if n == 1:
        r = _metrics(np.array([1.0]), mu, cov, 0.0, ids)
        r.method = "nested_clustered_optimization"
        return r

    labels, uniq = _cluster_labels(cov, linkage_method, n_clusters)
    cluster_assets = {c: list(np.where(labels == c)[0]) for c in uniq}

    label = "nested_clustered_optimization(objective='max_sharpe')"
    names = {c: [ids[i] for i in cluster_assets[c]] for c in uniq}

    # Within-cluster weights.
    intra = np.zeros(n)
    for c in uniq:
        idx = cluster_assets[c]
        sub_cov, sub_mu = cov[np.ix_(idx, idx)], mu[idx]
        if objective == "min_variance":
            intra[idx] = _analytic_min_variance(sub_cov)
        else:
            intra[idx] = _tangency_weights(sub_cov, sub_mu, f"{label}: cluster {names[c]}")

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
    if objective == "min_variance":
        across = _analytic_min_variance(Kcov)
    else:
        # Not the n == 1 shortcut: with one cluster this sum has the sign of
        # that cluster's 1'Sigma^-1 mu, which is the check that matters there.
        direction = np.linalg.pinv(Kcov + np.eye(Kn) * EPSILON) @ Kmu
        total = float(direction.sum())
        if not total > 1e-12 * float(np.abs(direction).sum()):
            clusters = [names[c] for c in uniq]
            raise ValueError(
                f"{label}: no fully invested maximum-Sharpe portfolio exists across the "
                f"cluster portfolios {clusters}: 1'C^-1 m <= 0 for their covariance C and "
                "means m, so their minimum-variance mix does not earn more than the "
                "risk-free rate of 0 (as in a falling market). Normalising would return "
                "the lowest-Sharpe mix. Use objective='min_variance'."
            )
        across = direction / total

    w = np.zeros(n)
    for a, ca in enumerate(uniq):
        idx = cluster_assets[ca]
        w[idx] = across[a] * intra[idx]
    w = w / w.sum()  # intra and across weights each sum to one

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
            # Inverse-variance split, as in this module's HRP, so gamma=0 == HRP.
            alpha = _inverse_variance_split(_schur_var(left, right), _schur_var(right, left))
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
    """Minimise Conditional VaR (Expected Shortfall) as an exact linear program.

    Rockafellar & Uryasev (2000) on the empirical daily scenarios ``r_t``:
    minimise ``zeta + sum(z) / ((1 - confidence) T)`` subject to
    ``z_t >= -r_t'w - zeta`` and ``z >= 0``, solved by HiGHS through
    ``scipy.optimize.linprog``. Full investment, weight bounds and gross
    exposure are hard constraints. Better than MVO for fat-tailed returns.

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        confidence: CVaR confidence level in (0, 1) (default 0.95).
        target_return: Minimum annualised expected return (decimal).
        constraints: Portfolio constraints; sector, factor, turnover and
            tracking-error limits are not supported and raise.

    Raises:
        RuntimeError: when the program is infeasible (for example an
            unattainable *target_return*) or the solver fails.
    """
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, hstack, identity, vstack

    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if target_return is not None and not np.isfinite(target_return):
        raise ValueError("target_return must be finite")
    ids, mu, cov = _build_matrices(returns)
    n = len(ids)
    c = constraints or Constraints()
    bounds = _bounds(n, c)
    _check_limits(c, "mean_cvar")
    min_len = min(len(returns[i]) for i in ids)
    R = np.column_stack([np.asarray(returns[i], dtype=np.float64)[-min_len:] for i in ids])
    t = R.shape[0]

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    gross = c.max_gross_exposure is not None and bool(np.any(lo < 0))
    ns = n if gross else 0
    # Variable layout: w (n) | s >= |w| (ns) | zeta (1) | z (t).
    nv = n + ns + 1 + t
    cost = np.zeros(nv)
    cost[n + ns] = 1.0
    cost[n + ns + 1:] = 1.0 / ((1.0 - confidence) * t)

    def zeros(rows: int, cols: int) -> csr_matrix:
        return csr_matrix((rows, cols))

    rows = [hstack([csr_matrix(-R), zeros(t, ns), csr_matrix(-np.ones((t, 1))), -identity(t)])]
    rhs = [np.zeros(t)]
    if gross:
        eye = identity(n)
        rows += [
            hstack([eye, -eye, zeros(n, 1 + t)]),
            hstack([-eye, -eye, zeros(n, 1 + t)]),
            hstack([zeros(1, n), csr_matrix(np.ones((1, n))), zeros(1, 1 + t)]),
        ]
        rhs += [np.zeros(n), np.zeros(n), np.array([float(c.max_gross_exposure)])]  # type: ignore[arg-type]
    if target_return is not None:
        rows.append(hstack([csr_matrix(-mu[None, :]), zeros(1, ns + 1 + t)]))
        rhs.append(np.array([-float(target_return)]))
    a_eq = np.zeros((1, nv))
    a_eq[0, :n] = 1.0
    var_bounds = (list(bounds)
                  + [(0.0, float(max(abs(a), abs(b)))) for a, b in zip(lo, hi)][:ns]
                  + [(None, None)] + [(0.0, None)] * t)
    result = linprog(cost, A_ub=vstack(rows, format="csr"), b_ub=np.concatenate(rhs),
                     A_eq=a_eq, b_eq=np.array([1.0]), bounds=var_bounds, method="highs")
    if result.status == 2 and target_return is None:
        raise RuntimeError(
            "mean_cvar: the weight and exposure limits admit no fully invested "
            f"portfolio ({result.message})"
        )
    if result.status == 2 and target_return is not None:
        # Say which limit failed: the highest expected return the weight and
        # exposure limits allow (an LP in w and s).
        best = linprog(
            np.concatenate([-mu, np.zeros(ns)]),
            A_ub=vstack(rows[1:4], format="csr")[:, :n + ns] if gross else None,
            b_ub=np.concatenate(rhs[1:4]) if gross else None,
            A_eq=a_eq[:, :n + ns], b_eq=np.array([1.0]),
            bounds=var_bounds[:n + ns], method="highs",
        )
        if best.status != 0:
            raise RuntimeError(
                "mean_cvar: the weight and exposure limits admit no fully invested "
                f"portfolio ({result.message})"
            )
        raise RuntimeError(
            f"mean_cvar: target_return {target_return:.6f} (annualised) is out of "
            f"reach; the highest expected return these limits allow is {-best.fun:.6f} "
            f"({result.message})"
        )
    w = np.asarray(result.x, dtype=float)[:n] if result.x is not None else np.full(n, np.nan)
    r = _core_result(result, w, bounds, c, "mean_cvar", mu, cov, 0.0, ids,
                     min_return=target_return)
    r.info["confidence"] = confidence
    r.info["cvar_daily"] = float(result.fun)
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
    maximise ``w'mu - epsilon * ||Sigma^{1/2} w|| - 0.5 * w'Sigma w`` with full
    investment, weight bounds and gross exposure as hard constraints.

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        epsilon: Size of the uncertainty set around mu (>= 0).
        risk_free_rate: Annual risk-free rate.
        constraints: Portfolio constraints; unsupported limits raise.

    Raises:
        RuntimeError: when the solver fails or returns weights that break a limit.
    """
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"epsilon must be finite and >= 0, got {epsilon}")
    ids, mu, cov = _build_matrices(returns, risk_free_rate)
    n = len(ids)
    c = constraints or Constraints()
    bounds = _bounds(n, c)
    _check_limits(c, "robust_mvo")

    def objective(w):
        var = float(w @ cov @ w)
        return -(float(w @ mu) - epsilon * np.sqrt(max(var, 0.0)) - 0.5 * var)

    def jac(w):
        sw = cov @ w
        vol = float(np.sqrt(max(float(w @ sw), 0.0)))
        penalty = epsilon * sw / vol if vol > EPSILON else np.zeros(n)
        return -(mu - penalty - sw)

    result, w = _solve_weights(objective, bounds, c, np.ones(n) / n, jac=jac)
    r = _core_result(result, w, bounds, c, "robust_mvo", mu, cov, risk_free_rate, ids)
    r.info["epsilon"] = epsilon
    return r


@frame_friendly
def max_diversification(
    returns: Dict[str, List[float]],
    *,
    constraints: Optional[Constraints] = None,
) -> OptResult:
    """Maximum Diversification Ratio (Choueifaty & Coignard 2008).

    Maximise DR = w'sigma / sqrt(w'Sigma w) with full investment, weight
    bounds and gross exposure as hard constraints; unsupported limits raise.
    """
    ids, mu, cov = _build_matrices(returns)
    n = len(ids)
    sigma = np.sqrt(np.diag(cov))
    c = constraints or Constraints()
    bounds = _bounds(n, c)
    _check_limits(c, "max_diversification")

    def neg_dr(w):
        port_vol = np.sqrt(w @ cov @ w)
        weighted_vol = w @ sigma
        return -weighted_vol / max(port_vol, EPSILON)

    result, w = _solve_weights(neg_dr, bounds, c, np.ones(n) / n)
    r = _core_result(result, w, bounds, c, "max_diversification", mu, cov, 0.0, ids)
    r.info["diversification_ratio"] = round(float(-neg_dr(w)), 4)
    return r


def _weight_vector(weights: Dict[str, float], ids: List[str], label: str) -> np.ndarray:
    """Weights in *ids* order; absent ids are not held (0), unknown ids raise."""
    unknown = sorted(set(weights) - set(ids))
    if unknown:
        raise ValueError(f"{label} has assets outside the universe: {unknown}")
    vec = np.array([float(weights.get(a, 0.0)) for a in ids], dtype=np.float64)
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{label} must be finite")
    return vec


@frame_friendly
def min_tracking_error(
    returns: Dict[str, List[float]],
    benchmark_weights: Dict[str, float],
    *,
    constraints: Optional[Constraints] = None,
) -> OptResult:
    """Minimise tracking error vs benchmark.

    TE = sqrt((w - w_b)' Sigma (w - w_b)), with full investment, weight bounds
    and gross exposure as hard constraints; unsupported limits raise.
    """
    ids, mu, cov = _build_matrices(returns)
    n = len(ids)
    w_b = _weight_vector(benchmark_weights, ids, "benchmark_weights")
    c = constraints or Constraints()
    bounds = _bounds(n, c)
    _check_limits(c, "min_tracking_error")

    def te(w):
        diff = w - w_b
        return float(diff @ cov @ diff)

    x0 = w_b.copy()
    if np.sum(np.abs(x0)) < EPSILON:
        x0 = np.ones(n) / n
    result, w = _solve_weights(te, bounds, c, x0, jac=lambda w: 2.0 * cov @ (w - w_b))
    r = _core_result(result, w, bounds, c, "min_tracking_error", mu, cov, 0.0, ids)
    r.info["tracking_error"] = round(float(np.sqrt(max(te(w), 0.0))) * 100, 4)
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

    Maximise: ``w'mu - 0.5 w'Sigma w - kappa |w - w_0|`` with full investment,
    weight bounds, gross exposure and ``max_turnover`` as hard constraints.

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        current_weights: Current portfolio weights (absent assets are not held).
        turnover_penalty: Cost per unit of turnover (kappa, >= 0).
        risk_free_rate: Annual risk-free rate.
        constraints: Portfolio constraints; sector, factor and tracking-error
            limits are not supported and raise.
    """
    if not np.isfinite(turnover_penalty) or turnover_penalty < 0:
        raise ValueError(f"turnover_penalty must be finite and >= 0, got {turnover_penalty}")
    ids, mu, cov = _build_matrices(returns, risk_free_rate)
    n = len(ids)
    w0 = _weight_vector(current_weights, ids, "current_weights")
    c = constraints or Constraints()
    bounds = _bounds(n, c)
    _check_limits(c, "turnover_penalized", allow_turnover=True)

    x0 = w0.copy() if np.sum(np.abs(w0)) > EPSILON else np.ones(n) / n
    result, w = _solve_weights(
        lambda w: -(float(w @ mu) - 0.5 * float(w @ cov @ w)), bounds, c, x0,
        jac=lambda w: -(mu - cov @ w), trades_from=w0,
        trade_cost=lambda t: turnover_penalty * float(np.sum(t)),
        trade_cost_jac=lambda t: np.full(n, turnover_penalty),
    )
    r = _core_result(result, w, bounds, c, "turnover_penalized", mu, cov, risk_free_rate, ids)
    turnover = float(np.sum(np.abs(w - w0)))
    if c.max_turnover is not None and turnover > c.max_turnover + 1e-7:
        raise RuntimeError("turnover_penalized: solver weights violate max_turnover")
    r.info["turnover"] = round(turnover, 6)
    r.info["turnover_penalty"] = turnover_penalty
    return r


# ── Grinold-Kahn Alpha-Risk-Cost Framework ───────────────────────────

def _trade_cost_function(
    cost_model: Any,
    asset_ids: List[str],
    portfolio_value: Optional[float],
    adv: Optional[Dict[str, float]],
    daily_volatility: Optional[Dict[str, float]],
) -> Tuple[Optional[Callable[[np.ndarray], float]], Optional[Callable[[np.ndarray], np.ndarray]]]:
    """Cost, as a fraction of the portfolio, of trading ``t_j`` of it in each asset.

    ``cost_j(t) = t * cost_model.estimate(t * portfolio_value, adv_j, vol_j) / 1e4``:
    the model's per-trade bps times the traded fraction, so spread and
    commission scale with turnover and market impact grows with trade size.
    A model whose impact depends on size needs real ``portfolio_value``,
    ``adv`` and ``daily_volatility``; they are never assumed to be zero.
    """
    if cost_model is None:
        return None, None
    if not callable(getattr(cost_model, "estimate", None)):
        raise TypeError("cost_model must have an estimate(notional, adv, volatility) method")
    from .costs import CostModel

    n = len(asset_ids)
    needs_size = needs_vol = True
    if isinstance(cost_model, CostModel):
        if cost_model.impact_model not in ("sqrt", "linear", "almgren_chriss"):
            raise ValueError(f"cost_model: unknown impact_model {cost_model.impact_model!r}")
        needs_size = cost_model.impact_model == "almgren_chriss" or cost_model.impact_coefficient != 0
        needs_vol = needs_size and cost_model.impact_model != "linear"
    inputs = {"portfolio_value": portfolio_value, "adv": adv, "daily_volatility": daily_volatility}
    required = ["portfolio_value", "adv"] * needs_size + ["daily_volatility"] * needs_vol
    missing = [name for name in dict.fromkeys(required) if inputs[name] is None]
    if missing:
        raise ValueError(
            f"cost_model prices market impact by trade size, which needs {missing}; "
            "pass them, or use a CostModel with impact_coefficient=0 for spread and "
            "commission only"
        )

    def per_asset(values: Optional[Dict[str, float]], name: str) -> np.ndarray:
        if values is None:
            return np.zeros(n)
        absent = [a for a in asset_ids if a not in values]
        if absent:
            raise ValueError(f"{name} is missing {absent}")
        vec = np.array([float(values[a]) for a in asset_ids])
        if not np.all(np.isfinite(vec)) or np.any(vec <= 0):
            raise ValueError(f"{name} must be positive and finite")
        return vec

    adv_vec = per_asset(adv, "adv")
    vol_vec = per_asset(daily_volatility, "daily_volatility")
    value = 0.0
    if portfolio_value is not None:
        value = float(portfolio_value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("portfolio_value must be positive and finite")

    def bps(j: int, traded: float) -> float:
        out = float(cost_model.estimate(traded * value, float(adv_vec[j]), float(vol_vec[j])))
        if not np.isfinite(out) or out < 0:
            raise ValueError(f"cost_model returned an invalid cost {out!r} for {asset_ids[j]!r}")
        return out

    if not needs_size:
        rate = np.array([bps(j, 0.0) for j in range(n)]) / 1e4
        return (lambda t: float(rate @ t)), (lambda t: rate)

    def total(t: np.ndarray) -> float:
        traded = np.maximum(np.asarray(t, dtype=float), 0.0)
        return float(sum(traded[j] * bps(j, traded[j]) for j in range(n)) / 1e4)

    return total, None


def alpha_risk_cost_optimize(
    alphas: Dict[str, float],
    cov: np.ndarray,
    *,
    current_weights: Optional[Dict[str, float]] = None,
    risk_aversion: float = 1.0,
    cost_model: Optional[Any] = None,
    constraints: Optional[Constraints] = None,
    asset_ids: Optional[List[str]] = None,
    portfolio_value: Optional[float] = None,
    adv: Optional[Dict[str, float]] = None,
    daily_volatility: Optional[Dict[str, float]] = None,
) -> ARCResult:
    """The complete alpha-risk-cost optimisation.

    Maximise: ``alpha'w - (lambda/2) w'Sigma w - sum_j cost_j(|w_j - w0_j|)``
    with full investment, weight bounds, gross exposure and ``max_turnover`` as
    hard constraints.

    ``cost_j(t) = t * cost_model.estimate(t * portfolio_value, adv_j, vol_j) / 1e4``,
    so a wider spread or higher commission cuts turnover, and a size-dependent
    impact model charges more per unit for larger trades. ``cost_model=None``
    means zero cost.

    The reported ``transfer_coefficient`` is the generalised Clarke, de Silva &
    Thorley (2002) measure: the correlation between risk-adjusted forecasts
    ``Sigma^-1/2 alpha`` and risk-adjusted weights ``Sigma^1/2 w``, equal to the
    portfolio's ex-ante information ratio divided by the unconstrained maximum
    ``sqrt(alpha' Sigma^-1 alpha)``. It lies in [-1, 1], is 1 for the
    unconstrained optimum, and NaN when every alpha is zero.

    Args:
        alphas: ``{asset_id: expected_alpha}`` (annualised); every asset needs one.
        cov: ``(n, n)`` covariance matrix (annualised) ordered like *asset_ids*.
        current_weights: Current weights; absent assets are not held.
        risk_aversion: Lambda (> 0); higher is more risk averse.
        cost_model: Object with ``.estimate(notional, adv, volatility)`` returning
            the cost in bps of the traded notional (e.g. :class:`CostModel`), or
            ``None`` for zero cost.
        constraints: Portfolio constraints; sector, factor and tracking-error
            limits are not supported and raise.
        asset_ids: Asset ordering matching *cov* (default: *alphas* keys).
        portfolio_value: Portfolio value in currency, for size-dependent costs.
        adv: ``{asset_id: average daily traded value}`` in the same currency.
        daily_volatility: ``{asset_id: daily return volatility}``.
    """
    if asset_ids is None:
        asset_ids = list(alphas.keys())
    n = len(asset_ids)
    missing = [a for a in asset_ids if a not in alphas]
    if missing:
        raise ValueError(f"alphas is missing {missing}")
    Sigma = np.asarray(cov, dtype=np.float64)
    if Sigma.shape != (n, n) or not np.all(np.isfinite(Sigma)):
        raise ValueError(f"cov must be a finite ({n}, {n}) matrix ordered like asset_ids")
    if not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError(f"risk_aversion must be positive, got {risk_aversion}")

    alpha_vec = np.array([float(alphas[a]) for a in asset_ids], dtype=np.float64)
    if not np.all(np.isfinite(alpha_vec)):
        raise ValueError("alphas must be finite")
    w0 = _weight_vector(current_weights or {}, asset_ids, "current_weights")

    c = constraints or Constraints()
    bounds = _bounds(n, c)
    _check_limits(c, "alpha_risk_cost_optimize", allow_turnover=True)
    cost_fn, cost_jac = _trade_cost_function(cost_model, asset_ids, portfolio_value, adv,
                                             daily_volatility)

    x0 = w0.copy() if np.sum(np.abs(w0)) > EPSILON else np.ones(n) / n
    use_trades = cost_fn is not None or c.max_turnover is not None
    result, w = _solve_weights(
        lambda w: -(float(alpha_vec @ w) - 0.5 * risk_aversion * float(w @ Sigma @ w)),
        bounds, c, x0,
        jac=lambda w: -(alpha_vec - risk_aversion * Sigma @ w),
        trades_from=w0 if use_trades else None,
        trade_cost=cost_fn,
        trade_cost_jac=cost_jac,
    )
    zero_mu = np.zeros(n)
    _core_result(result, w, bounds, c, "alpha_risk_cost_optimize", zero_mu, Sigma, 0.0,
                 asset_ids)
    trades = np.abs(w - w0)
    if c.max_turnover is not None and trades.sum() > c.max_turnover + 1e-7:
        raise RuntimeError("alpha_risk_cost_optimize: solver weights violate max_turnover")

    expected_alpha = float(alpha_vec @ w)
    expected_risk = float(np.sqrt(max(float(w @ Sigma @ w), 0.0)))
    expected_cost = cost_fn(trades) if cost_fn is not None else 0.0
    ir = expected_alpha / max(expected_risk, EPSILON)

    try:
        ir_max = float(np.sqrt(max(float(alpha_vec @ np.linalg.solve(Sigma, alpha_vec)), 0.0)))
    except np.linalg.LinAlgError as exc:
        raise ValueError("cov must be positive definite") from exc
    if ir_max > 0 and expected_risk > 0:
        tc = expected_alpha / (ir_max * expected_risk)
    else:
        tc = float("nan")

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
    """W(A) = sum_ij A_ij * (pi(i) - pi(j))^2, where ``perm[i]`` is asset i's position."""
    p = np.asarray(perm, dtype=float)
    diff = p[:, None] - p[None, :]
    return float(np.sum(np.asarray(similarity, dtype=float) * diff * diff))


def _laplacian(similarity: np.ndarray) -> np.ndarray:
    """L = D - A, so that W(A) = 2 * p' L p for a position vector p."""
    A = np.asarray(similarity, dtype=float).copy()
    np.fill_diagonal(A, 0.0)
    return np.diag(A.sum(axis=1)) - A


def _order_to_positions(order: List[int]) -> np.ndarray:
    positions = np.empty(len(order), dtype=int)
    positions[np.asarray(order, dtype=int)] = np.arange(len(order))
    return positions


def _spectral_seriation(similarity: np.ndarray, max_passes: int = 10_000) -> List[int]:
    """Heuristic order minimising W(A) for any (dense or sparse) similarity.

    Sorts assets by the Fiedler vector of the similarity Laplacian, the
    continuous relaxation of exactly this objective (Atkins, Boman &
    Hendrickson 1998, "A spectral algorithm for seriation"), then applies
    best-improvement pairwise position swaps until none lowers W(A). The
    result depends on the matrix, not on the order the columns arrive in
    (up to exact ties). Returns the asset indices in their new order.
    """
    L = _laplacian(similarity)
    _, vecs = np.linalg.eigh(L)
    order = list(np.argsort(vecs[:, 1], kind="stable"))
    p = _order_to_positions(order).astype(float)
    diag = np.diag(L)
    pair_curv = diag[:, None] + diag[None, :] - 2.0 * L
    for _ in range(max_passes):
        g = L @ p
        delta = p[None, :] - p[:, None]  # p_b - p_a for swapping assets a and b
        change = 4.0 * delta * (g[:, None] - g[None, :]) + 2.0 * delta * delta * pair_curv
        a, b = np.unravel_index(int(np.argmin(change)), change.shape)
        if not change[a, b] < -1e-12 * max(1.0, float(p @ g)):
            break
        p[a], p[b] = p[b], p[a]
    return [int(i) for i in np.argsort(p)]


def _brute_force_permutation(similarity: np.ndarray) -> np.ndarray:
    """Exact positions minimising W(A) by enumeration. Feasible for N <= 10.

    Returns ``positions`` with ``positions[i]`` the slot of asset i.
    """
    import itertools

    n = similarity.shape[0]
    if n > 10:
        raise ValueError(f"Brute-force permutation infeasible for n={n}. Use 'heuristic'.")

    L = _laplacian(similarity)
    best_perm = np.arange(n)
    best_obj = float("inf")
    perms = itertools.permutations(range(n))
    while True:
        chunk = np.array(list(itertools.islice(perms, 50_000)), dtype=float)
        if chunk.size == 0:
            break
        objs = 2.0 * np.einsum("ij,jk,ik->i", chunk, L, chunk)
        k = int(np.argmin(objs))
        if objs[k] < best_obj - 1e-12:
            best_obj = float(objs[k])
            best_perm = chunk[k].astype(int)

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
    """Decode asset positions from the N^2 binary QUBO solution vector.

    ``x[i * n + p] = 1`` places asset i at position p; the result has
    ``positions[i]`` = slot of asset i (greedy repair if the sample is not a
    permutation). Use ``np.argsort(positions)`` for the asset order.
    """
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

    # Coupling of split k is the block sum |S|[:k, k:]; a 2D prefix sum gives
    # every split at once instead of an O(n^2) Python loop per split.
    block = np.abs(np.asarray(similarity, dtype=float)[np.ix_(indices, indices)])
    prefix = block.cumsum(axis=0).cumsum(axis=1)
    k = np.arange(1, n)
    cross = prefix[k - 1, n - 1] - prefix[k - 1, k - 1]
    avg_cost = cross / (k * (n - k))
    best_split = int(k[np.argmin(avg_cost)])  # first minimum, as before
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
        1. **Quasi-block-diagonalisation**: find positions pi that minimise
           W(A) = sum_ij |C_ij| * (pi(i) - pi(j))^2, then order the assets by
           position (``info["permutation"]`` lists asset indices in that order)
        2. **Recursive bisection**: split reordered assets at the point that
           minimises off-diagonal coupling
        3. **Inverse-variance weight allocation**: each side of a split gets
           ``1 - V_side / (V_left + V_right)`` of its parent's weight, as in HRP

    Args:
        returns: ``{asset_id: [daily_returns]}``.
        backend: ``"heuristic"`` (spectral seriation plus pairwise-swap local
                 search, default; works on dense correlation matrices),
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

    qubo_feasible: Optional[bool] = None
    if solver is not None or backend in ("simulated_annealing", "braket"):
        if solver is None:
            from .quantum_backends import get_solver
            solver = get_solver(backend)
            backend_label = backend
        else:
            backend_label = "custom_solver"
        x = np.asarray(solver.solve(build_permutation_qubo(similarity)), dtype=float)
        assignment = x.reshape(n, n)
        qubo_feasible = bool(np.all(assignment.sum(axis=0) == 1) and np.all(assignment.sum(axis=1) == 1))
        # Positions -> order: the solver says where each asset goes.
        sorted_idx = [int(i) for i in np.argsort(_decode_permutation_from_qubo(x, n), kind="stable")]
    elif backend == "heuristic":
        sorted_idx = _spectral_seriation(similarity)
        backend_label = backend
    elif backend == "brute_force":
        sorted_idx = [int(i) for i in np.argsort(_brute_force_permutation(similarity), kind="stable")]
        backend_label = backend
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Use: 'heuristic', 'brute_force', 'simulated_annealing', or 'braket'."
        )

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

                alpha = _inverse_variance_split(_cluster_variance(cov_mat, left_orig),
                                                _cluster_variance(cov_mat, right_orig))

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
    r.info["backend"] = backend_label
    r.info["permutation"] = sorted_idx
    r.info["objective_value"] = _quasi_block_diag_objective(
        similarity, _order_to_positions(sorted_idx))
    if qubo_feasible is not None:
        r.info["qubo_feasible"] = qubo_feasible
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
