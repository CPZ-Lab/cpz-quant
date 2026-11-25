"""Convex-optimization backend (cvxpy): exact convex programs with rich constraints.

Optional dependency — install with ``pip install cpz-quant[cvx]`` (and
``cpz-quant[cvx-mip]`` for cardinality constraints, which need a
mixed-integer solver).

This backend complements the native scipy optimisers in
:mod:`cpz_quant.portfolio.optimization` with formulations that only a
disciplined convex solver can express exactly:

- cardinality-constrained portfolios (mixed-integer, ``max_assets``)
- robust mean-variance with box or ellipsoidal uncertainty sets on
  expected returns
- CVaR minimisation as the Rockafellar-Uryasev linear program with
  turnover and exposure constraints
- L2 (ridge) weight regularisation and turnover limits on any objective

Fail-loud policy: if cvxpy or a required mixed-integer solver is
missing, functions raise ``ImportError``/``RuntimeError`` with install
instructions — they never silently substitute an approximation.

All functions are pure: data in, results out. No DB access, no API calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from cpz_quant.frames import frame_friendly

from .optimization import (
    EPSILON,
    TRADING_DAYS,
    Constraints,
    OptResult,
    _build_matrices,
    _check_limits,
    _metrics,
)

try:  # optional dependency
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on minimal installs
    CVXPY_AVAILABLE = False


def has_cvxpy() -> bool:
    """``True`` when the optional cvxpy backend is installed."""
    return CVXPY_AVAILABLE


def _require_cvxpy() -> None:
    if not CVXPY_AVAILABLE:
        raise ImportError(
            "This optimiser requires the convex backend. "
            "Install it with: pip install cpz-quant[cvx]"
        )


def _base_constraints(
    w: "cp.Variable",
    constraints: Optional[Constraints],
    prev_weights: Optional[np.ndarray],
    label: str,
) -> "List[Any]":
    """Full investment, bounds, gross exposure and turnover as hard constraints.

    Sector limits, factor limits and a tracking-error limit need data that
    :class:`Constraints` does not carry (a sector map, factor loadings,
    benchmark weights), so they raise ``NotImplementedError`` rather than
    being dropped, exactly as in the native optimizers.
    """
    c = constraints or Constraints()
    _check_limits(c, label, allow_turnover=True)
    cons: List[Any] = [cp.sum(w) == 1]
    # Same effective floor as the native optimizers: a positive min_weight
    # still applies when long_only=True.
    cons.append(w >= (max(0.0, c.min_weight) if c.long_only else c.min_weight))
    cons.append(w <= c.max_weight)
    if c.max_gross_exposure is not None:
        cons.append(cp.norm1(w) <= c.max_gross_exposure)
    if c.max_turnover is not None:
        if prev_weights is None:
            raise ValueError(
                "Constraints.max_turnover requires prev_weights so turnover "
                "can be measured against the current portfolio."
            )
        cons.append(cp.norm1(w - prev_weights) <= c.max_turnover)
    return cons


def _prev_vector(
    ids: List[str], prev_weights: Optional[Dict[str, float]]
) -> Optional[np.ndarray]:
    if prev_weights is None:
        return None
    return np.array([float(prev_weights.get(i, 0.0)) for i in ids])


def _solve_or_raise(problem: "cp.Problem", label: str) -> None:
    problem.solve()
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"{label}: solver returned status '{problem.status}'")


@frame_friendly
def mean_variance_cvx(
    returns: Dict[str, List[float]],
    *,
    risk_aversion: float = 1.0,
    target_return: Optional[float] = None,
    risk_free_rate: float = 0.0,
    l2_reg: float = 0.0,
    constraints: Optional[Constraints] = None,
    prev_weights: Optional[Dict[str, float]] = None,
) -> OptResult:
    """Markowitz mean-variance as an exact quadratic program.

    Maximises ``mu @ w - risk_aversion * w' cov w - l2_reg * ||w||^2``,
    or minimises variance subject to ``mu @ w >= target_return`` when
    *target_return* (annualised, decimal) is given.

    Gross-exposure and turnover limits are hard constraints of the program,
    not penalties. The native core optimizers support gross exposure but
    reject turnover limits; this backend accepts explicit previous weights.
    """
    _require_cvxpy()
    ids, mu, cov = _build_matrices(returns, risk_free_rate)
    n = len(ids)
    w = cp.Variable(n)
    prev = _prev_vector(ids, prev_weights)
    cons = _base_constraints(w, constraints, prev, "mean_variance_cvx")
    risk = cp.quad_form(w, cp.psd_wrap(cov))
    objective: Any
    if target_return is not None:
        cons.append(mu @ w >= target_return)
        objective = cp.Minimize(risk + l2_reg * cp.sum_squares(w))
    else:
        objective = cp.Maximize(
            mu @ w - risk_aversion * risk - l2_reg * cp.sum_squares(w)
        )
    _solve_or_raise(cp.Problem(objective, cons), "mean_variance_cvx")
    result = _metrics(np.asarray(w.value).ravel(), mu, cov, risk_free_rate, ids)
    result.method = "mean_variance_cvx"
    result.info = {"backend": "cvxpy", "l2_reg": l2_reg}
    return result


@frame_friendly
def mean_cvar_cvx(
    returns: Dict[str, List[float]],
    *,
    alpha: float = 0.95,
    target_return: Optional[float] = None,
    risk_free_rate: float = 0.0,
    constraints: Optional[Constraints] = None,
    prev_weights: Optional[Dict[str, float]] = None,
) -> OptResult:
    """Minimise portfolio CVaR via the Rockafellar-Uryasev linear program.

    *alpha* is the confidence level (0.95 = expected loss in the worst
    5% of scenarios). *target_return* (annualised, decimal), exposure,
    and turnover limits become hard LP constraints.
    """
    _require_cvxpy()
    if not 0.5 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0.5, 1), got {alpha}")
    ids, mu, cov = _build_matrices(returns, risk_free_rate)
    min_len = min(len(returns[i]) for i in ids)
    R = np.column_stack(
        [np.array(returns[i][-min_len:], dtype=np.float64) for i in ids]
    )  # finite: _build_matrices rejected missing or non-finite returns
    n_scen, n = R.shape
    w = cp.Variable(n)
    zeta = cp.Variable()  # VaR level
    z = cp.Variable(n_scen, nonneg=True)  # scenario excess losses
    prev = _prev_vector(ids, prev_weights)
    cons = _base_constraints(w, constraints, prev, "mean_cvar_cvx")
    cons.append(z >= -R @ w - zeta)
    if target_return is not None:
        cons.append(mu @ w >= target_return)
    cvar = zeta + cp.sum(z) / (n_scen * (1.0 - alpha))
    _solve_or_raise(cp.Problem(cp.Minimize(cvar), cons), "mean_cvar_cvx")
    result = _metrics(np.asarray(w.value).ravel(), mu, cov, risk_free_rate, ids)
    result.method = "mean_cvar_cvx"
    result.info = {
        "backend": "cvxpy",
        "alpha": alpha,
        "cvar_daily": round(float(cvar.value), 6),  # type: ignore[arg-type]
        "var_daily": round(float(zeta.value), 6),  # type: ignore[arg-type]
    }
    return result


@frame_friendly
def robust_mean_variance_cvx(
    returns: Dict[str, List[float]],
    *,
    uncertainty: str = "ellipsoidal",
    kappa: float = 1.0,
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
    constraints: Optional[Constraints] = None,
    prev_weights: Optional[Dict[str, float]] = None,
) -> OptResult:
    """Robust mean-variance under an uncertainty set on expected returns.

    Guards against estimation error in ``mu`` — the classic cause of
    extreme, unstable Markowitz weights — by optimising against the
    worst-case expected return inside the set:

    - ``"ellipsoidal"``: worst case over ``{m : ||S^-1/2 (m - mu)|| <= kappa}``
      where ``S`` is the estimation-error covariance of the annualised mean.
      Yields the second-order-cone penalty ``kappa * ||S^1/2 w||``.
    - ``"box"``: worst case over per-asset intervals ``mu_i +/- kappa * se_i``
      (``se`` = standard error of the annualised mean). Yields ``kappa * se @ |w|``.

    ``mu`` is the daily mean times 252 and ``cov`` the daily covariance times
    252, so ``S = 252^2 * cov_daily / T = 252 * cov / T``: with ``kappa = 1``
    the penalty is one standard error of the portfolio's annualised mean
    return, ``252 * sd(daily portfolio return) / sqrt(T)``.

    *kappa* scales the set (0 recovers plain mean-variance; 1-2 is typical).
    """
    _require_cvxpy()
    if uncertainty not in ("ellipsoidal", "box"):
        raise ValueError(f"uncertainty must be 'ellipsoidal' or 'box', got {uncertainty!r}")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError(f"kappa must be finite and >= 0, got {kappa}")
    ids, mu, cov = _build_matrices(returns, risk_free_rate)
    n = len(ids)
    n_obs = min(len(returns[i]) for i in ids)
    w = cp.Variable(n)
    prev = _prev_vector(ids, prev_weights)
    cons = _base_constraints(w, constraints, prev, "robust_mean_variance_cvx")
    # Covariance of the annualised mean estimate (mu = 252 * daily mean).
    est_cov = cov * TRADING_DAYS / n_obs
    if uncertainty == "ellipsoidal":
        # sqrtm via eigendecomposition; est_cov is symmetric PSD.
        vals, vecs = np.linalg.eigh(est_cov)
        sqrt_s = vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T
        penalty = kappa * cp.norm2(sqrt_s @ w)
    else:
        se = np.sqrt(np.clip(np.diag(est_cov), 0.0, None))
        penalty = kappa * (se @ cp.abs(w))
    risk = cp.quad_form(w, cp.psd_wrap(cov))
    objective = cp.Maximize(mu @ w - penalty - risk_aversion * risk)
    _solve_or_raise(cp.Problem(objective, cons), "robust_mean_variance_cvx")
    result = _metrics(np.asarray(w.value).ravel(), mu, cov, risk_free_rate, ids)
    result.method = "robust_mean_variance_cvx"
    result.info = {
        "backend": "cvxpy",
        "uncertainty": uncertainty,
        "kappa": kappa,
        "worst_case_penalty": round(float(penalty.value), 6),
    }
    return result


_MIP_SOLVERS = ("SCIP", "GUROBI", "MOSEK", "CPLEX", "XPRESS")


def _mip_solver() -> str:
    installed = cp.installed_solvers()
    for s in _MIP_SOLVERS:
        if s in installed:
            return s
    raise RuntimeError(
        "Cardinality constraints need a mixed-integer-capable solver; none of "
        f"{_MIP_SOLVERS} is installed (found: {installed}). "
        "Install the open-source option with: pip install cpz-quant[cvx-mip]"
    )


@frame_friendly
def cardinality_constrained_cvx(
    returns: Dict[str, List[float]],
    *,
    max_assets: int,
    min_position: float = 0.01,
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
    constraints: Optional[Constraints] = None,
) -> OptResult:
    """Mean-variance with a hard limit on the number of holdings.

    Solves the exact mixed-integer quadratic program: at most
    *max_assets* names, and every selected name gets at least
    *min_position* weight (semi-continuous — no dust positions).
    Long-only by construction.

    Requires a mixed-integer solver (``pip install cpz-quant[cvx-mip]``
    for SCIP). Raises ``RuntimeError`` if none is available — this
    function never falls back to a heuristic.
    """
    _require_cvxpy()
    ids, mu, cov = _build_matrices(returns, risk_free_rate)
    n = len(ids)
    if not 1 <= max_assets <= n:
        raise ValueError(f"max_assets must be in [1, {n}], got {max_assets}")
    c = constraints or Constraints()
    for name in ("sector_limits", "factor_limits", "max_turnover", "max_tracking_error"):
        if getattr(c, name) is not None:
            raise NotImplementedError(
                f"cardinality_constrained_cvx: {name} is not supported by this optimizer"
            )
    if c.min_weight > 0:
        raise ValueError(
            "cardinality_constrained_cvx: a positive min_weight would force every asset "
            "into the portfolio; use min_position for the floor of selected names"
        )
    max_w = min(c.max_weight, 1.0)
    if min_position > max_w:
        raise ValueError(f"min_position {min_position} exceeds max_weight {max_w}")
    solver = _mip_solver()
    w = cp.Variable(n)
    y = cp.Variable(n, boolean=True)
    cons: List[Any] = [
        cp.sum(w) == 1,
        w >= 0,
        w <= cp.multiply(max_w, y),
        w >= cp.multiply(min_position, y),
        cp.sum(y) <= max_assets,
    ]
    risk = cp.quad_form(w, cp.psd_wrap(cov))
    problem = cp.Problem(cp.Maximize(mu @ w - risk_aversion * risk), cons)
    problem.solve(solver=solver)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"cardinality_constrained_cvx: solver {solver} returned status "
            f"'{problem.status}'"
        )
    wv = np.asarray(w.value).ravel()
    wv[wv < EPSILON] = 0.0
    result = _metrics(wv, mu, cov, risk_free_rate, ids)
    result.method = "cardinality_constrained_cvx"
    result.info = {
        "backend": "cvxpy",
        "solver": solver,
        "max_assets": max_assets,
        "n_selected": int(np.sum(wv > EPSILON)),
        "min_position": min_position,
    }
    return result
