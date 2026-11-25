"""Factor attribution — isolate idiosyncratic alpha from repackaged risk premia.

A top quant risk desk never accepts a raw return: they neutralize it against a
factor model (market, size, value, momentum, quality, low-vol, ...) and ask what
alpha survives. This runs an OLS of strategy returns on the supplied factor
returns and reports the annualized alpha, per-factor betas, R^2, and the
idiosyncratic (residual) Sharpe — the part of the edge that is not explained by
known factors.

Needs a factor-returns dataset, so it lives in the SDK / backtest engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np

TRADING_DAYS = 252.0


@dataclass
class FactorAttribution:
    """OLS factor attribution.

    ``alpha_tstat``, the beta t-stats and ``idiosyncratic_sharpe`` are NaN
    when the factors explain the returns exactly (residuals at floating-point
    noise): each would be rounding error divided by rounding error.
    """
    alpha_daily: float
    alpha_annualized: float
    alpha_tstat: float
    betas: Dict[str, float]
    beta_tstats: Dict[str, float]
    r_squared: float
    idiosyncratic_sharpe: float   # annualized Sharpe of the residual (unexplained) return
    explained_fraction: float     # share of return variance explained by the factors
    observations: int = field(default=0)


def factor_attribution(
    strategy_returns: Sequence[float],
    factor_returns: Mapping[str, Sequence[float]],
) -> FactorAttribution:
    """Attribute strategy returns to a set of factor return series via OLS.

    Args:
        strategy_returns: (T,) per-period strategy returns.
        factor_returns: mapping of factor name -> (T,) per-period factor returns
            (e.g. {"MKT": ..., "SMB": ..., "HML": ..., "MOM": ...}).

    Returns a :class:`FactorAttribution` with annualized alpha and residual Sharpe.
    """
    y = np.asarray(strategy_returns, dtype=float)
    T = y.size
    names = list(factor_returns.keys())
    if not names:
        raise ValueError("provide at least one factor series")
    F = np.column_stack([np.asarray(factor_returns[n], dtype=float) for n in names])
    if F.shape[0] != T:
        raise ValueError("all factor series must align with strategy_returns length")
    if T <= len(names) + 1:
        raise ValueError("not enough observations for the number of factors")

    # Design matrix with intercept.
    X = np.column_stack([np.ones(T), F])
    beta, _resid, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residuals = y - fitted

    # R^2.
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # An exact fit leaves residuals at floating-point noise relative to the
    # returns. Their ratio to anything else is noise too, so the residual-based
    # statistics are undefined rather than "strongly significant".
    scale = float(np.sum(y**2))
    exact_fit = ss_res <= 1e-20 * scale if scale > 0 else True

    # Standard errors -> t-stats (classical OLS).
    dof = T - X.shape[1]
    sigma2 = ss_res / dof if dof > 0 else float("nan")
    if exact_fit:
        tstats = np.full_like(beta, np.nan)
    else:
        xtx_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * xtx_inv))
        with np.errstate(divide="ignore", invalid="ignore"):
            tstats = np.where(se > 0, beta / se, np.nan)

    alpha_daily = float(beta[0])
    # Appraisal ratio (Treynor-Black): the idiosyncratic return each period is
    # alpha (the intercept) plus mean-zero residual noise, so its risk-adjusted
    # value is alpha / residual-vol, annualized. Using residuals.mean() here is
    # wrong — OLS residuals are mean-zero by construction.
    res_sd = float(residuals.std(ddof=1))
    idio_sharpe = (alpha_daily / res_sd) * np.sqrt(TRADING_DAYS) if not exact_fit and res_sd > 0 \
        else float("nan")

    return FactorAttribution(
        alpha_daily=round(alpha_daily, 6),
        alpha_annualized=round(alpha_daily * TRADING_DAYS, 4),
        alpha_tstat=round(float(tstats[0]), 3),  # NaN survives round()
        betas={n: round(float(b), 4) for n, b in zip(names, beta[1:])},
        beta_tstats={n: round(float(t), 3) for n, t in zip(names, tstats[1:])},
        r_squared=round(float(r2), 4),
        idiosyncratic_sharpe=round(float(idio_sharpe), 3),
        explained_fraction=round(float(max(0.0, min(1.0, r2))), 4),
        observations=int(T),
    )
