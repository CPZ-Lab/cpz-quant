"""Factor models: statistical PCA, fundamental, exposure, risk decomposition.

All functions are pure: data in, results out. No DB access, no API calls.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Union

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
    """Fundamental factor model output.

    ``factor_returns`` holds one value per regression day, the first being
    day ``start_index`` of the aligned return series.
    """
    factor_returns: Dict[str, List[float]] = Field(default_factory=dict)
    factor_loadings: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    factor_premiums: Dict[str, float] = Field(default_factory=dict)
    t_stats: Dict[str, float] = Field(default_factory=dict)
    r_squared: float = 0.0
    start_index: int = 0


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
    market_caps: Mapping[str, Union[float, Sequence[float]]],
    *,
    book_values: Optional[Dict[str, float]] = None,
    momentum_window: int = 252,
    momentum_skip: int = 21,
    caps_at_start: bool = False,
    trading_days: int = TRADING_DAYS,
) -> FundFactorModel:
    """Cross-sectional fundamental factor model (Fama-MacBeth, point in time).

    For every day ``t`` the cross-section of returns ``R[t]`` is regressed
    on exposures that were known at the close of day ``t - 1``:

    - market: 1 for every asset;
    - size: ``-(log cap_i(t-1) - mean)``;
    - value (with *book_values*): ``book_i / cap_i(t-1) - mean``;
    - momentum: compounded return over ``R[t - momentum_window : t -
      momentum_skip]`` (the classic 12-1 month window by default), demeaned.

    Building the exposures from the same returns they are regressed on (for
    example momentum over the last year of the sample, applied to every day
    of that year) fits part of each day's return with itself and reports a
    premium of roughly ``252 / days`` with large t-stats on pure noise.

    Args:
        returns: ``{asset_id: [daily_returns]}`` mapping (aligned at the end).
        market_caps: per asset a time series aligned with *returns* (the
            cap at the close of each day; day ``t`` uses day ``t - 1``), or,
            with ``caps_at_start=True``, one number: the cap at the close
            *before the first return*, rolled forward with the asset's own
            returns so day ``t`` only uses returns before ``t``.
        caps_at_start: confirm that single-number caps are start-of-sample
            values. Without it a single number raises ``ValueError``: today's
            caps (the usual snapshot) already contain the sample's returns,
            and using them for every day shows size and value premia in pure
            noise.
        book_values: ``{asset_id: book_value}`` (optional, enables value),
            held fixed over the sample.
        momentum_window: Look-back for momentum (days). Momentum is used
            when there are at least ``momentum_window + 2`` days; the
            regressions then start on day ``momentum_window``.
        momentum_skip: Most recent days left out of the momentum window.
        trading_days: Trading days per year.

    Returns:
        :class:`FundFactorModel`. ``factor_returns`` has one value per
        regression day, starting at ``start_index`` of the aligned return
        series. ``factor_loadings`` are the exposures known at the end of
        the sample (the ones that apply to the next day). Premiums and
        t-stats are Fama-MacBeth averages over the regression days.

    Raises:
        KeyError: an asset has no market cap (or no book value when
            *book_values* is given).
        ValueError: a non-positive or non-finite market cap, a cap series
            that does not match the returns, or too few days.
    """
    ids = list(returns.keys())
    min_len = min(len(returns[i]) for i in ids)
    n = len(ids)
    R = np.column_stack([np.array(returns[i][-min_len:], dtype=np.float64) for i in ids])
    t_len = R.shape[0]

    missing_caps = [s for s in ids if s not in market_caps]
    if missing_caps:
        raise KeyError(f"market_caps missing for {missing_caps}")
    single = [s for s in ids if np.ndim(market_caps[s]) == 0]
    if single and not caps_at_start:
        raise ValueError(
            f"market_caps for {single[:5]}{' ...' if len(single) > 5 else ''} are single numbers. "
            "Pass each cap as a series aligned with the returns, or pass caps_at_start=True if "
            "the numbers are caps at the close before the first return. Today's caps already "
            "contain the sample's returns and would put look-ahead into every day's size and "
            "value exposure."
        )
    # caps[t, i] = cap of asset i at the close of day t; row 0 is the close
    # before the first return, so day t's exposures read row t.
    caps = np.empty((t_len + 1, n))
    for i, s in enumerate(ids):
        c = market_caps[s]
        if np.ndim(c) == 0:
            caps[0, i] = float(c)  # type: ignore[arg-type]
            caps[1:, i] = caps[0, i] * np.cumprod(1.0 + R[:, i])
        else:
            series = np.asarray(c, dtype=np.float64)
            if series.size < t_len:
                raise ValueError(f"market_caps[{s!r}] has {series.size} values for {t_len} returns")
            caps[1:, i] = series[-t_len:]
            caps[0, i] = np.nan  # unknown: day 0 cannot be regressed
    if np.any(caps[1:] <= 0) or np.any(np.isinf(caps)) or np.any(np.isnan(caps[1:])):
        raise ValueError("market caps must be positive and finite")
    if not np.isnan(caps[0]).any() and np.any(caps[0] <= 0):
        raise ValueError("market caps must be positive and finite")

    factor_names = ["market", "size"]
    if book_values:
        missing_bv = [s for s in ids if s not in book_values]
        if missing_bv:
            raise KeyError(f"book_values missing for {missing_bv}")
        book = np.array([float(book_values[s]) for s in ids])
        factor_names.append("value")
    use_momentum = t_len >= momentum_window + 2
    if use_momentum:
        if not 0 <= momentum_skip < momentum_window:
            raise ValueError("momentum_skip must be >= 0 and < momentum_window")
        factor_names.append("momentum")
        log_growth = np.vstack([np.zeros((1, n)), np.cumsum(np.log1p(R), axis=0)])

    def exposures(t: int) -> np.ndarray:
        """Exposure matrix for day ``t``, from data up to the close of ``t - 1``."""
        cols = [np.ones(n)]
        log_cap = np.log(caps[t])
        cols.append(-(log_cap - log_cap.mean()))
        if book_values:
            bm = book / caps[t]
            cols.append(bm - bm.mean())
        if use_momentum:
            lo, hi = t - momentum_window, t - momentum_skip
            mom = np.expm1(log_growth[hi] - log_growth[lo])
            cols.append(mom - mom.mean())
        return np.column_stack(cols)

    first = momentum_window if use_momentum else (0 if not np.isnan(caps[0]).any() else 1)
    days = range(first, t_len)
    if len(days) < 2:
        raise ValueError(
            f"need at least two regression days; {t_len} days of returns start regressions "
            f"on day {first}"
        )

    k_f = len(factor_names)
    factor_ret = np.empty((len(days), k_f))
    ss_res = 0.0
    for row, t in enumerate(days):
        B_t = exposures(t)
        f_t, _, _, _ = np.linalg.lstsq(B_t, R[t, :], rcond=None)
        factor_ret[row, :] = f_t
        ss_res += float(np.sum((R[t, :] - B_t @ f_t) ** 2))

    t_obs = len(days)
    premiums = {f: float(np.mean(factor_ret[:, j]) * trading_days) for j, f in enumerate(factor_names)}
    t_stats_out = {}
    for j, f in enumerate(factor_names):
        mean_f = float(np.mean(factor_ret[:, j]))
        std_f = float(np.std(factor_ret[:, j], ddof=1))
        t_stats_out[f] = mean_f / max(std_f, EPSILON) * math.sqrt(t_obs)

    R_used = R[first:]
    ss_tot = float(np.sum((R_used - R_used.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, EPSILON)

    # Exposures known after the last close: what applies to the next day.
    if use_momentum:
        latest = exposures(t_len) if t_len - momentum_window >= 0 else None
    else:
        latest = exposures(t_len)
    loadings: Dict[str, Dict[str, float]] = {}
    if latest is not None:
        for i, s in enumerate(ids):
            loadings[s] = {f: float(latest[i, j]) for j, f in enumerate(factor_names)}

    return FundFactorModel(
        factor_returns={f: factor_ret[:, j].tolist() for j, f in enumerate(factor_names)},
        factor_loadings=loadings,
        factor_premiums=premiums,
        t_stats=t_stats_out,
        r_squared=round(float(r2), 4),
        start_index=int(first),
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

    ``per_factor_contribution[j] = (B'w)_j * (F B'w)_j`` (Euler allocation),
    which sums to ``factor_variance``.

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
        # Euler split: (B'w)_j * (F B'w)_j includes factor j's covariance with
        # the others, so the parts sum to factor_variance (one can be negative).
        Bw = B.T @ w
        FBw = F @ Bw
        for j, fname in enumerate(factor_names):
            contrib = float(Bw[j] * FBw[j])
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
