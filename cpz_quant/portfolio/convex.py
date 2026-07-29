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
    Constraints,
    OptResult,
    _build_matrices,
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
) -> "List[Any]":
    c = constraints or Constraints()
    cons: List[Any] = [cp.sum(w) == 1]
    if c.long_only:
        cons.append(w >= 0)
    else:
        cons.append(w >= c.min_weight)
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

    Unlike the scipy path, gross-exposure and turnover limits are hard
    constraints of the program, not penalties.
    """
    _require_cvxpy()
    ids, mu, cov = _build_matrices(returns, risk_free_rate)
    n = len(ids)
    w = cp.Variable(n)
    prev = _prev_vector(ids, prev_weights)
    cons = _base_constraints(w, constraints, prev)
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
    )
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    n_scen, n = R.shape
    w = cp.Variable(n)
    zeta = cp.Variable()  # VaR level
    z = cp.Variable(n_scen, nonneg=True)  # scenario excess losses
    prev = _prev_vector(ids, prev_weights)
    cons = _base_constraints(w, constraints, prev)
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
      where ``S`` is the estimation-error covariance ``cov / T``. Yields the
      second-order-cone penalty ``kappa * ||S^1/2 w||``.
    - ``"box"``: worst case over per-asset intervals ``mu_i ± kappa * se_i``
      (``se`` = standard error of the mean). Yields ``kappa * se @ |w|``.

    *kappa* scales the set (0 recovers plain mean-variance; 1-2 is typical).
    """
    _require_cvxpy()
    if uncertainty not in ("ellipsoidal", "box"):
        raise ValueError(f"uncertainty must be 'ellipsoidal' or 'box', got {uncertainty!r}")
    ids, mu, cov = _build_matrices(returns, risk_free_rate)
    n = len(ids)
    n_obs = min(len(returns[i]) for i in ids)
    w = cp.Variable(n)
    prev = _prev_vector(ids, prev_weights)
    cons = _base_constraints(w, constraints, prev)
    est_cov = cov / max(n_obs, 1)
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
