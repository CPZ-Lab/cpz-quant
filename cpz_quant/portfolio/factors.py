"""Factor models: statistical PCA, fundamental, exposure, risk decomposition.

All functions are pure: data in, results out. No DB access, no API calls.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from cpz_quant.frames import frame_friendly

TRADING_DAYS: int = 252
EPSILON: float = 1e-15


class StatFactorModel(BaseModel):
    """Statistical (PCA) factor model output."""
    factor_loadings: List[List[float]] = Field(default_factory=list)
    factor_returns: List[List[float]] = Field(default_factory=list)
    specific_variance: List[float] = Field(default_factory=list)
    explained_variance_ratio: List[float] = Field(default_factory=list)
    factor_covariance: List[List[float]] = Field(default_factory=list)
    n_factors: int = 0
    asset_ids: List[str] = Field(default_factory=list)


class FundFactorModel(BaseModel):
    """Fundamental factor model output."""
    factor_returns: Dict[str, List[float]] = Field(default_factory=dict)
    factor_loadings: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    factor_premiums: Dict[str, float] = Field(default_factory=dict)
    t_stats: Dict[str, float] = Field(default_factory=dict)
    r_squared: float = 0.0


class FactorRiskDecomp(BaseModel):
    """Factor vs specific risk decomposition."""
    total_variance: float = 0.0
    factor_variance: float = 0.0
    specific_variance: float = 0.0
    factor_risk_pct: float = 0.0
    specific_risk_pct: float = 0.0
    per_factor_contribution: Dict[str, float] = Field(default_factory=dict)


@frame_friendly
def statistical_factor_model(
    returns: Dict[str, List[float]],
    *,
    n_factors: int = 5,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
) -> StatFactorModel:
    """PCA-based statistical factor model.

    Args:
        returns: ``{asset_id: [daily_returns]}`` mapping.
        n_factors: Number of principal components to extract.
        annualize: Annualise factor covariance and specific variance.
        trading_days: Trading days per year.
    """
    ids = list(returns.keys())
    min_len = min(len(returns[i]) for i in ids)
    R = np.column_stack([np.array(returns[i][-min_len:], dtype=np.float64) for i in ids])
    t, n = R.shape
    n_factors = min(n_factors, n, t - 1)

    X = R - R.mean(axis=0)
    cov_mat = X.T @ X / (t - 1)

    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total_var = np.sum(eigvals)
    explained = eigvals[:n_factors] / total_var if total_var > EPSILON else np.zeros(n_factors)

    B = eigvecs[:, :n_factors]
    factor_ret = X @ B
    F = np.cov(factor_ret.T, ddof=1)
    if F.ndim == 0:
        F = np.array([[float(F)]])

    residuals = X - factor_ret @ B.T
    spec_var = np.var(residuals, axis=0, ddof=1)

    scale = trading_days if annualize else 1.0

    return StatFactorModel(
        factor_loadings=B.tolist(),
        factor_returns=factor_ret.tolist(),
        specific_variance=(spec_var * scale).tolist(),
        explained_variance_ratio=explained.tolist(),
        factor_covariance=(F * scale).tolist(),
        n_factors=n_factors,
        asset_ids=ids,
    )


@frame_friendly
def fundamental_factors(
    returns: Dict[str, List[float]],
    market_caps: Dict[str, float],
    *,
    book_values: Optional[Dict[str, float]] = None,
    momentum_window: int = 252,
    trading_days: int = TRADING_DAYS,
) -> FundFactorModel:
    """Cross-sectional fundamental factor model (Fama-MacBeth style).

    Builds factor exposures for Market, Size (SMB), and optionally
    Value (HML) and Momentum (UMD), then runs cross-sectional
    regressions to estimate factor returns.

    Args:
        returns: ``{asset_id: [daily_returns]}`` mapping.
        market_caps: ``{asset_id: market_cap}`` mapping.
        book_values: ``{asset_id: book_value}`` (optional, enables HML).
        momentum_window: Look-back for momentum factor.
        trading_days: Trading days per year.
    """
    ids = list(returns.keys())
    min_len = min(len(returns[i]) for i in ids)
    n = len(ids)

    log_caps = {s: np.log(max(market_caps.get(s, 1.0), EPSILON)) for s in ids}
    mean_log_cap = np.mean(list(log_caps.values()))
    size_exposure = {s: -(log_caps[s] - mean_log_cap) for s in ids}

    factor_names = ["market", "size"]
    loadings: Dict[str, Dict[str, float]] = {}
    for s in ids:
        loadings[s] = {"market": 1.0, "size": size_exposure[s]}

    if book_values:
        bm = {s: book_values.get(s, 0) / max(market_caps.get(s, 1), EPSILON) for s in ids}
        mean_bm = np.mean(list(bm.values()))
        for s in ids:
            loadings[s]["value"] = bm[s] - mean_bm
        factor_names.append("value")

    if min_len >= momentum_window:
        for s in ids:
            r = np.array(returns[s], dtype=np.float64)
            cum = float(np.prod(1.0 + r[-momentum_window:-21]) - 1.0) if len(r) > momentum_window else 0.0
            loadings[s]["momentum"] = cum
        mean_mom = np.mean([loadings[s]["momentum"] for s in ids])
        for s in ids:
            loadings[s]["momentum"] -= mean_mom
        factor_names.append("momentum")

    B = np.zeros((n, len(factor_names)))
    for i, s in enumerate(ids):
        for j, f in enumerate(factor_names):
            B[i, j] = loadings[s].get(f, 0.0)

    R = np.column_stack([np.array(returns[i][-min_len:], dtype=np.float64) for i in ids])
    t = R.shape[0]

    factor_ret = np.zeros((t, len(factor_names)))
    for k in range(t):
        try:
            betas, _, _, _ = np.linalg.lstsq(B, R[k, :], rcond=None)
            factor_ret[k, :] = betas
        except np.linalg.LinAlgError:
            pass

    premiums = {f: float(np.mean(factor_ret[:, j]) * trading_days) for j, f in enumerate(factor_names)}
    t_stats_out = {}
    for j, f in enumerate(factor_names):
        mean_f = np.mean(factor_ret[:, j])
        std_f = np.std(factor_ret[:, j], ddof=1)
        t_stats_out[f] = float(mean_f / max(std_f, EPSILON) * np.sqrt(t))

    predicted = R @ np.linalg.pinv(B.T)
    ss_res = np.sum((R - predicted @ B.T) ** 2)
    ss_tot = np.sum((R - R.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, EPSILON)

    return FundFactorModel(
        factor_returns={f: factor_ret[:, j].tolist() for j, f in enumerate(factor_names)},
        factor_loadings=loadings,
        factor_premiums=premiums,
        t_stats=t_stats_out,
        r_squared=round(float(r2), 4),
    )


def factor_exposure(
    weights: Dict[str, float],
    factor_loadings: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Portfolio factor exposure: ``w' @ B``.

    Args:
        weights: ``{asset_id: weight}``.
        factor_loadings: ``{asset_id: {factor: loading}}``.

    Returns:
        ``{factor_name: portfolio_exposure}``.
    """
    factors: Dict[str, float] = {}
    for asset, w in weights.items():
        for f, loading in factor_loadings.get(asset, {}).items():
            factors[f] = factors.get(f, 0.0) + w * loading
    return {f: round(v, 6) for f, v in factors.items()}


def factor_risk_decomposition(
    weights: Dict[str, float],
    factor_loadings: np.ndarray,
    factor_cov: np.ndarray,
    specific_var: np.ndarray,
    *,
    asset_ids: Optional[List[str]] = None,
    factor_names: Optional[List[str]] = None,
) -> FactorRiskDecomp:
    """Decompose portfolio variance into factor risk and specific risk.

    factor_risk = w' B F B' w
    specific_risk = w' D w
    total = factor_risk + specific_risk

    Args:
        weights: ``{asset_id: weight}``.
        factor_loadings: ``(n_assets, n_factors)`` loading matrix B.
        factor_cov: ``(n_factors, n_factors)`` factor covariance F.
        specific_var: ``(n_assets,)`` diagonal of specific variance D.
        asset_ids: Asset ID ordering matching the matrices.
        factor_names: Factor name ordering.
    """
    if asset_ids is None:
        asset_ids = list(weights.keys())

    w = np.array([weights.get(a, 0.0) for a in asset_ids], dtype=np.float64)
    B = np.asarray(factor_loadings, dtype=np.float64)
    F = np.asarray(factor_cov, dtype=np.float64)
    D = np.diag(np.asarray(specific_var, dtype=np.float64))

    factor_var = float(w @ B @ F @ B.T @ w)
    spec_var = float(w @ D @ w)
    total_var = factor_var + spec_var

    per_factor: Dict[str, float] = {}
    if factor_names and B.shape[1] == len(factor_names):
        Bw = B.T @ w
        for j, fname in enumerate(factor_names):
            contrib = float(Bw[j] ** 2 * F[j, j])
            per_factor[fname] = round(contrib, 8)

    return FactorRiskDecomp(
        total_variance=round(total_var, 8),
        factor_variance=round(factor_var, 8),
        specific_variance=round(spec_var, 8),
        factor_risk_pct=round(factor_var / max(total_var, EPSILON) * 100, 2),
        specific_risk_pct=round(spec_var / max(total_var, EPSILON) * 100, 2),
        per_factor_contribution=per_factor,
    )


@frame_friendly
def rolling_factor_exposure(
    returns: Dict[str, List[float]],
    factor_returns: Dict[str, List[float]],
    *,
    window: int = 60,
) -> Dict[str, Dict[str, List[float]]]:
    """Rolling factor betas over time for each asset. Detects style drift.

    Args:
        returns: ``{asset_id: [daily_returns]}`` mapping.
        factor_returns: ``{factor_name: [daily_factor_returns]}`` mapping.
        window: Rolling OLS window.

    Returns:
        ``{asset_id: {factor_name: [rolling_beta]}}``.
    """
    factor_names = list(factor_returns.keys())
    min_len = min(len(v) for v in factor_returns.values())

    F = np.column_stack([
        np.array(factor_returns[f][-min_len:], dtype=np.float64) for f in factor_names
    ])

    result: Dict[str, Dict[str, List[float]]] = {}
    for asset_id, rets in returns.items():
        r = np.array(rets[-min_len:], dtype=np.float64)
        t = len(r)
        betas: Dict[str, List[float]] = {f: [] for f in factor_names}
        for i in range(window, t):
            y = r[i - window : i]
            X = F[i - window : i]
            try:
                b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                for j, f in enumerate(factor_names):
                    betas[f].append(round(float(b[j]), 6))
            except np.linalg.LinAlgError:
                for f in factor_names:
                    betas[f].append(0.0)
        result[asset_id] = betas

    return result
