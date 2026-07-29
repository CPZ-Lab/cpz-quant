"""Risk-measure library + mean-risk portfolio optimization.

A native, dependency-free (numpy/scipy only) implementation of the full menu of
convex/coherent risk measures used as optimization objectives — the breadth an
institutional allocator expects. Every measure is computed from a realized
return series (scenario matrix), so the same code powers both reporting and
optimization.

Measures implemented (all on losses = -return, so larger = riskier):
  dispersion : VARIANCE, STANDARD_DEVIATION, SEMI_VARIANCE, MAD,
               FIRST_LOWER_PARTIAL_MOMENT, GINI_MEAN_DIFFERENCE
  tail       : VALUE_AT_RISK, CVAR (Expected Shortfall), EVAR (Entropic VaR),
               WORST_REALIZATION, ENTROPIC_RISK_MEASURE
  drawdown   : MAX_DRAWDOWN, AVERAGE_DRAWDOWN, DRAWDOWN_AT_RISK, CDAR, EDAR,
               ULCER_INDEX

References: Rockafellar & Uryasev (CVaR); Ahmadi-Javid (EVaR/EDaR);
Chekhlov, Uryasev & Zabarankin (CDaR); Yitzhaki (Gini mean difference).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
from scipy import optimize as sp_opt

EPSILON = 1e-12


class RiskMeasure(str, Enum):
    VARIANCE = "variance"
    STANDARD_DEVIATION = "standard_deviation"
    SEMI_VARIANCE = "semi_variance"
    MAD = "mean_absolute_deviation"
    FIRST_LOWER_PARTIAL_MOMENT = "first_lower_partial_moment"
    GINI_MEAN_DIFFERENCE = "gini_mean_difference"
    VALUE_AT_RISK = "value_at_risk"
    CVAR = "cvar"
    EVAR = "evar"
    WORST_REALIZATION = "worst_realization"
    ENTROPIC_RISK_MEASURE = "entropic_risk_measure"
    MAX_DRAWDOWN = "max_drawdown"
    AVERAGE_DRAWDOWN = "average_drawdown"
    DRAWDOWN_AT_RISK = "drawdown_at_risk"
    CDAR = "cdar"
    EDAR = "edar"
    ULCER_INDEX = "ulcer_index"


_DRAWDOWN_MEASURES = {
    RiskMeasure.MAX_DRAWDOWN, RiskMeasure.AVERAGE_DRAWDOWN, RiskMeasure.DRAWDOWN_AT_RISK,
    RiskMeasure.CDAR, RiskMeasure.EDAR, RiskMeasure.ULCER_INDEX,
}


def _drawdowns(returns: np.ndarray, compounded: bool = True) -> np.ndarray:
    """Drawdown series (<= 0) from a return vector."""
    if compounded:
        wealth = np.cumprod(1.0 + returns)
    else:
        wealth = 1.0 + np.cumsum(returns)
    peak = np.maximum.accumulate(wealth)
    return wealth / peak - 1.0


def _evar_from_losses(losses: np.ndarray, beta: float) -> float:
    """Entropic Value at Risk of a loss vector at confidence ``beta``.

    EVaR_beta = inf_{z>0} z * ln( (1/T) sum exp(losses/z) / (1-beta) ).
    Solved as a 1-D convex minimization over ln(z).
    """
    losses = np.asarray(losses, dtype=float)
    if losses.size == 0:
        return 0.0
    one_minus = max(1.0 - beta, EPSILON)

    def obj(log_z: float) -> float:
        z = np.exp(log_z)
        m = losses.max()
        # log-sum-exp for numerical stability
        lse = m / z + np.log(np.mean(np.exp((losses - m) / z)))
        return z * (lse - np.log(one_minus))

    res = sp_opt.minimize_scalar(obj, bounds=(-20, 20), method="bounded")
    return float(res.fun)


@dataclass
class RiskMeasureValues:
    values: dict = field(default_factory=dict)

    def __getitem__(self, k):  # convenience
        return self.values[k]


def compute_risk(
    returns,
    measure: RiskMeasure = RiskMeasure.VARIANCE,
    *,
    beta: float = 0.95,
    theta: float = 1.0,
    compounded: bool = True,
) -> float:
    """Compute a single risk measure from a return vector.

    Args:
        returns: 1-D array of (portfolio) returns.
        measure: which :class:`RiskMeasure` to compute.
        beta: confidence level for tail/drawdown-at-risk measures.
        theta: risk-aversion for the entropic risk measure.
        compounded: compound the wealth path for drawdown measures.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    m = RiskMeasure(measure)
    one_minus = max(1.0 - beta, EPSILON)

    if m == RiskMeasure.VARIANCE:
        return float(r.var(ddof=1)) if r.size > 1 else 0.0
    if m == RiskMeasure.STANDARD_DEVIATION:
        return float(r.std(ddof=1)) if r.size > 1 else 0.0
    if m == RiskMeasure.SEMI_VARIANCE:
        d = np.minimum(r - r.mean(), 0.0)
        return float(np.mean(d * d))
    if m == RiskMeasure.MAD:
        return float(np.mean(np.abs(r - r.mean())))
    if m == RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT:
        return float(np.mean(np.maximum(-r, 0.0)))
    if m == RiskMeasure.GINI_MEAN_DIFFERENCE:
        x = np.sort(r)
        n = x.size
        idx = np.arange(1, n + 1)
        return float((2.0 / (n * n)) * np.sum((2 * idx - n - 1) * x))
    if m == RiskMeasure.VALUE_AT_RISK:
        return float(-np.percentile(r, one_minus * 100.0, method="linear"))
    if m == RiskMeasure.CVAR:
        var = np.percentile(r, one_minus * 100.0, method="linear")
        tail = r[r <= var]
        return float(-tail.mean()) if tail.size else float(-var)
    if m == RiskMeasure.WORST_REALIZATION:
        return float(-r.min())
    if m == RiskMeasure.EVAR:
        return _evar_from_losses(-r, beta)
    if m == RiskMeasure.ENTROPIC_RISK_MEASURE:
        # rho_theta = (1/theta) ln E[e^{-theta r}], via stable log-sum-exp.
        a = -theta * r
        amax = float(a.max())
        lse = amax + np.log(np.mean(np.exp(a - amax)))
        return float(lse / theta)

    # ── drawdown-based ────────────────────────────────────────────────────
    dd = _drawdowns(r, compounded=compounded)  # <= 0
    depth = -dd  # >= 0
    if m == RiskMeasure.MAX_DRAWDOWN:
        return float(depth.max())
    if m == RiskMeasure.AVERAGE_DRAWDOWN:
        return float(depth.mean())
    if m == RiskMeasure.ULCER_INDEX:
        return float(np.sqrt(np.mean(depth * depth)))
    if m == RiskMeasure.DRAWDOWN_AT_RISK:
        return float(np.percentile(depth, beta * 100.0, method="linear"))
    if m == RiskMeasure.CDAR:
        dar = np.percentile(depth, beta * 100.0, method="linear")
        tail = depth[depth >= dar]
        return float(tail.mean()) if tail.size else float(dar)
    if m == RiskMeasure.EDAR:
        return _evar_from_losses(depth, beta)

    raise ValueError(f"Unknown risk measure {measure!r}")


def all_risk_measures(returns, *, beta: float = 0.95, theta: float = 1.0) -> dict:
    """Compute every risk measure for a return vector (reporting helper)."""
    return {m.value: compute_risk(returns, m, beta=beta, theta=theta) for m in RiskMeasure}


# ── mean-risk optimization ───────────────────────────────────────────────────
@dataclass
class MeanRiskResult:
    weights: dict
    risk_measure: str
    objective: str
    risk: float
    expected_return: float
    ratio: float
    success: bool
    method: str = "mean_risk"


def mean_risk_optimize(
    asset_returns,
    *,
    risk_measure: RiskMeasure = RiskMeasure.CVAR,
    objective: str = "min_risk",  # 'min_risk' | 'max_ratio' | 'max_return' | 'max_utility'
    beta: float = 0.95,
    risk_aversion: float = 1.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    budget: float = 1.0,
    asset_ids: Optional[List[str]] = None,
    n_restarts: int = 3,
    rf: float = 0.0,
) -> MeanRiskResult:
    """Optimize portfolio weights against any risk measure.

    Args:
        asset_returns: (T, N) matrix of per-period asset returns.
        risk_measure: objective risk measure (from :class:`RiskMeasure`).
        objective: ``min_risk`` (minimize the risk measure), ``max_ratio``
            (maximize mean-excess / risk), ``max_return`` (subject to risk),
            or ``max_utility`` (mean - risk_aversion * risk).
        beta: confidence for tail/drawdown measures.
        min_weight/max_weight: per-asset bounds (long-only by default).
        budget: sum of weights (1.0 = fully invested).
    """
    R = np.asarray(asset_returns, dtype=float)
    if R.ndim != 2:
        raise ValueError("asset_returns must be 2-D (T observations x N assets)")
    T, N = R.shape
    mu = R.mean(axis=0)
    ids = asset_ids or [f"asset_{i}" for i in range(N)]

    def port_risk(w: np.ndarray) -> float:
        return compute_risk(R @ w, risk_measure, beta=beta)

    def neg_objective(w: np.ndarray) -> float:
        pr = port_risk(w)
        pm = float(mu @ w)
        if objective == "min_risk":
            return pr
        if objective == "max_return":
            return -pm
        if objective == "max_utility":
            return -(pm - risk_aversion * pr)
        if objective == "max_ratio":
            return -((pm - rf) / (pr + EPSILON))
        raise ValueError(f"Unknown objective {objective!r}")

    bounds = [(min_weight, max_weight)] * N
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - budget}]

    best = None
    rng = np.random.default_rng(42)
    for k in range(max(1, n_restarts)):
        x0 = np.full(N, budget / N) if k == 0 else rng.dirichlet(np.ones(N)) * budget
        try:
            res = sp_opt.minimize(neg_objective, x0, method="SLSQP", bounds=bounds,
                                  constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
            if res.success and (best is None or res.fun < best.fun):
                best = res
        except Exception:
            continue
    if best is None:
        w = np.full(N, budget / N)
        success = False
    else:
        w = best.x
        success = True

    w = np.where(np.abs(w) < 1e-8, 0.0, w)
    pr = port_risk(w)
    pm = float(mu @ w)
    return MeanRiskResult(
        weights={ids[i]: float(round(w[i], 8)) for i in range(N)},
        risk_measure=RiskMeasure(risk_measure).value,
        objective=objective,
        risk=float(pr),
        expected_return=pm,
        ratio=float((pm - rf) / (pr + EPSILON)),
        success=success,
    )
