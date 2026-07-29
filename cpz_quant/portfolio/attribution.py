"""Performance attribution: Brinson-Fachler, factor, risk, alpha/beta decomposition.

All functions are pure: data in, results out.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

EPSILON: float = 1e-15
TRADING_DAYS: int = 252


class BrinsonResult(BaseModel):
    total_active_return: float = 0.0
    allocation_effect: float = 0.0
    selection_effect: float = 0.0
    interaction_effect: float = 0.0
    per_sector: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class FactorAttrResult(BaseModel):
    factor_contributions: Dict[str, float] = Field(default_factory=dict)
    alpha: float = 0.0
    residual: float = 0.0
    r_squared: float = 0.0
    total_return: float = 0.0


class RiskAttrResult(BaseModel):
    total_risk: float = 0.0
    factor_risk: float = 0.0
    specific_risk: float = 0.0
    factor_risk_pct: float = 0.0
    specific_risk_pct: float = 0.0
    per_factor: Dict[str, float] = Field(default_factory=dict)


class AlphaBetaResult(BaseModel):
    alpha: float = 0.0
    beta: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    r_squared: float = 0.0


def brinson_fachler(
    portfolio_weights: Dict[str, float],
    portfolio_returns: Dict[str, float],
    benchmark_weights: Dict[str, float],
    benchmark_returns: Dict[str, float],
    sectors: Dict[str, str],
) -> BrinsonResult:
    """Brinson-Fachler (1985) attribution.

    Decomposes active return into allocation, selection, and interaction
    effects at the sector level.

    Args:
        portfolio_weights: ``{asset: weight}`` in portfolio.
        portfolio_returns: ``{asset: period_return}`` in portfolio.
        benchmark_weights: ``{asset: weight}`` in benchmark.
        benchmark_returns: ``{asset: period_return}`` in benchmark.
        sectors: ``{asset: sector_name}`` mapping.
    """
    all_sectors = set(sectors.values())

    sector_pw: Dict[str, float] = {}
    sector_pr: Dict[str, float] = {}
    sector_bw: Dict[str, float] = {}
    sector_br: Dict[str, float] = {}

    for sec in all_sectors:
        p_assets = [a for a in portfolio_weights if sectors.get(a) == sec]
        b_assets = [a for a in benchmark_weights if sectors.get(a) == sec]

        pw = sum(portfolio_weights.get(a, 0) for a in p_assets)
        bw = sum(benchmark_weights.get(a, 0) for a in b_assets)

        sector_pw[sec] = pw
        sector_bw[sec] = bw
        sector_pr[sec] = (
            sum(portfolio_weights.get(a, 0) * portfolio_returns.get(a, 0) for a in p_assets) / max(pw, EPSILON)
            if pw > EPSILON else 0.0
        )
        sector_br[sec] = (
            sum(benchmark_weights.get(a, 0) * benchmark_returns.get(a, 0) for a in b_assets) / max(bw, EPSILON)
            if bw > EPSILON else 0.0
        )

    total_br = sum(sector_bw[s] * sector_br[s] for s in all_sectors)

    total_alloc = 0.0
    total_select = 0.0
    total_interact = 0.0
    per_sector: Dict[str, Dict[str, float]] = {}

    for sec in all_sectors:
        dw = sector_pw[sec] - sector_bw[sec]
        dr = sector_pr[sec] - sector_br[sec]

        alloc = dw * (sector_br[sec] - total_br)
        select = sector_bw[sec] * dr
        interact = dw * dr

        total_alloc += alloc
        total_select += select
        total_interact += interact

        per_sector[sec] = {
            "allocation": round(alloc * 100, 4),
            "selection": round(select * 100, 4),
            "interaction": round(interact * 100, 4),
            "total": round((alloc + select + interact) * 100, 4),
        }

    total_active = total_alloc + total_select + total_interact

    return BrinsonResult(
        total_active_return=round(total_active * 100, 4),
        allocation_effect=round(total_alloc * 100, 4),
        selection_effect=round(total_select * 100, 4),
        interaction_effect=round(total_interact * 100, 4),
        per_sector=per_sector,
    )


def factor_attribution(
    portfolio_returns: List[float],
    factor_returns: Dict[str, List[float]],
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
) -> FactorAttrResult:
    """Factor-based return attribution via multivariate OLS.

    return = sum(beta_k * factor_k) + alpha + epsilon

    Args:
        portfolio_returns: Portfolio daily return series.
        factor_returns: ``{factor_name: [daily_returns]}`` mapping.
        annualize: Annualise alpha.
        trading_days: Trading days per year.
    """
    factor_names = list(factor_returns.keys())
    min_len = min(len(portfolio_returns), *(len(factor_returns[f]) for f in factor_names))

    y = np.array(portfolio_returns[-min_len:], dtype=np.float64)
    X = np.column_stack([
        np.array(factor_returns[f][-min_len:], dtype=np.float64) for f in factor_names
    ])
    X_with_const = np.column_stack([np.ones(min_len), X])

    try:
        betas, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
    except np.linalg.LinAlgError:
        return FactorAttrResult()

    alpha_daily = betas[0]
    factor_betas = betas[1:]

    predicted = X_with_const @ betas
    residual = y - predicted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, EPSILON)

    contributions: Dict[str, float] = {}
    for j, f in enumerate(factor_names):
        contrib = float(factor_betas[j] * np.mean(X[:, j]))
        if annualize:
            contrib *= trading_days
        contributions[f] = round(contrib * 100, 4)

    alpha = alpha_daily * (trading_days if annualize else 1.0)

    return FactorAttrResult(
        factor_contributions=contributions,
        alpha=round(alpha * 100, 4),
        residual=round(float(np.std(residual, ddof=1)) * math.sqrt(trading_days) * 100, 4),
        r_squared=round(r2, 4),
        total_return=round(float(np.mean(y)) * trading_days * 100, 4),
    )


def risk_attribution(
    weights: Dict[str, float],
    cov: np.ndarray,
    *,
    factor_loadings: Optional[np.ndarray] = None,
    factor_cov: Optional[np.ndarray] = None,
    specific_var: Optional[np.ndarray] = None,
    asset_ids: Optional[List[str]] = None,
    factor_names: Optional[List[str]] = None,
) -> RiskAttrResult:
    """Decompose portfolio risk into factor and specific components.

    If factor model matrices are not provided, uses total covariance only.

    Args:
        weights: ``{asset_id: weight}``.
        cov: ``(n, n)`` covariance matrix.
        factor_loadings: ``(n, k)`` factor loading matrix B.
        factor_cov: ``(k, k)`` factor covariance F.
        specific_var: ``(n,)`` specific variance diagonal.
        asset_ids: Asset ordering matching matrices.
        factor_names: Factor name ordering.
    """
    if asset_ids is None:
        asset_ids = list(weights.keys())

    w = np.array([weights.get(a, 0.0) for a in asset_ids], dtype=np.float64)
    total_var = float(w @ cov @ w)
    total_risk = math.sqrt(total_var) if total_var > 0 else 0.0

    if factor_loadings is not None and factor_cov is not None and specific_var is not None:
        B = np.asarray(factor_loadings)
        F = np.asarray(factor_cov)
        D = np.diag(np.asarray(specific_var))
        factor_var = float(w @ B @ F @ B.T @ w)
        spec_var_val = float(w @ D @ w)
    else:
        factor_var = total_var
        spec_var_val = 0.0

    per_factor: Dict[str, float] = {}
    if factor_names and factor_loadings is not None and factor_cov is not None:
        Bw = factor_loadings.T @ w
        for j, fname in enumerate(factor_names):
            per_factor[fname] = round(float(Bw[j] ** 2 * factor_cov[j, j]) / max(total_var, EPSILON) * 100, 2)

    return RiskAttrResult(
        total_risk=round(total_risk * 100, 4),
        factor_risk=round(math.sqrt(max(factor_var, 0)) * 100, 4),
        specific_risk=round(math.sqrt(max(spec_var_val, 0)) * 100, 4),
        factor_risk_pct=round(factor_var / max(total_var, EPSILON) * 100, 2),
        specific_risk_pct=round(spec_var_val / max(total_var, EPSILON) * 100, 2),
        per_factor=per_factor,
    )


def alpha_beta_decomposition(
    portfolio_returns: List[float],
    benchmark_returns: List[float],
    *,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
) -> AlphaBetaResult:
    """CAPM alpha/beta decomposition: ``R_p = alpha + beta * R_b + epsilon``.

    Args:
        portfolio_returns: Portfolio daily return series.
        benchmark_returns: Benchmark daily return series.
        annualize: Annualise alpha and tracking error.
        trading_days: Trading days per year.
    """
    min_len = min(len(portfolio_returns), len(benchmark_returns))
    y = np.array(portfolio_returns[-min_len:], dtype=np.float64)
    x = np.array(benchmark_returns[-min_len:], dtype=np.float64)

    cov_xb = np.cov(x, y, ddof=1)
    var_x = cov_xb[0, 0]
    beta = cov_xb[0, 1] / max(var_x, EPSILON)
    alpha_daily = np.mean(y) - beta * np.mean(x)

    excess = y - x
    te_daily = float(np.std(excess, ddof=1))

    ss_res = float(np.sum((y - (alpha_daily + beta * x)) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, EPSILON)

    scale = math.sqrt(trading_days) if annualize else 1.0
    alpha_ann = alpha_daily * (trading_days if annualize else 1.0)
    te_ann = te_daily * scale
    ir = alpha_ann / max(te_ann, EPSILON) if te_ann > EPSILON else 0.0

    return AlphaBetaResult(
        alpha=round(alpha_ann * 100, 4),
        beta=round(beta, 4),
        tracking_error=round(te_ann * 100, 4),
        information_ratio=round(ir, 4),
        r_squared=round(r2, 4),
    )


def rolling_attribution(
    portfolio_returns: List[float],
    benchmark_returns: List[float],
    *,
    window: int = 60,
    trading_days: int = TRADING_DAYS,
) -> Dict[str, List[float]]:
    """Rolling alpha, beta, and IR over time.

    Args:
        portfolio_returns: Portfolio daily return series.
        benchmark_returns: Benchmark daily return series.
        window: Rolling window size.
        trading_days: Trading days per year.

    Returns:
        ``{"alpha": [...], "beta": [...], "ir": [...]}``
    """
    min_len = min(len(portfolio_returns), len(benchmark_returns))
    y = np.array(portfolio_returns[-min_len:], dtype=np.float64)
    x = np.array(benchmark_returns[-min_len:], dtype=np.float64)

    alphas: List[float] = []
    betas: List[float] = []
    irs: List[float] = []

    for i in range(window, min_len):
        ywin = y[i - window : i]
        xwin = x[i - window : i]

        cov_xb = np.cov(xwin, ywin, ddof=1)
        var_x = cov_xb[0, 0]
        b = cov_xb[0, 1] / max(var_x, EPSILON)
        a = np.mean(ywin) - b * np.mean(xwin)

        excess = ywin - xwin
        te = float(np.std(excess, ddof=1)) * math.sqrt(trading_days)
        a_ann = a * trading_days
        ir = a_ann / max(te, EPSILON)

        alphas.append(round(a_ann * 100, 4))
        betas.append(round(b, 4))
        irs.append(round(ir, 4))

    return {"alpha": alphas, "beta": betas, "ir": irs}
