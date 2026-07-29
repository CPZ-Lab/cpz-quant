"""Pre-selection transformers — prune the asset universe before optimization.

Composable, pure-numpy filters that reduce the universe to a tradable, de-noised
set. Each takes ``{asset_id: [returns]}`` and returns a filtered mapping (plus a
``selected`` id list), so they chain cleanly ahead of any allocator.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np

EPSILON = 1e-15
TRADING_DAYS = 252


def _matrix(returns: Mapping[str, Sequence[float]]):
    ids = list(returns.keys())
    min_len = min((len(returns[i]) for i in ids), default=0)
    R = np.column_stack([np.asarray(returns[i][-min_len:], dtype=float) for i in ids]) if ids else np.zeros((0, 0))
    return ids, R


def _subset(returns: Mapping[str, Sequence[float]], keep: List[str]) -> Dict[str, list]:
    return {k: list(returns[k]) for k in keep}


def drop_zero_variance(returns, *, tol: float = 1e-12) -> Dict[str, list]:
    """Drop assets whose return variance is (near) zero."""
    keep = [i for i in returns if np.var(np.asarray(returns[i], dtype=float)) > tol]
    return _subset(returns, keep)


def select_complete_assets(returns, *, min_obs: int = None) -> Dict[str, list]:  # type: ignore[assignment]
    """Keep only assets with a full (longest common) history — drops names with
    short samples from inception/delisting. If ``min_obs`` is given, keep assets
    with at least that many observations."""
    lengths = {i: len(returns[i]) for i in returns}
    if not lengths:
        return {}
    target = max(lengths.values()) if min_obs is None else min_obs
    keep = [i for i in returns if lengths[i] >= target]
    return _subset(returns, keep)


def drop_highly_correlated(returns, *, threshold: float = 0.95) -> Dict[str, list]:
    """Greedily drop one asset from every pair with ``|corr| > threshold``.

    Of each offending pair, the asset with the higher mean absolute correlation
    to the rest of the universe is removed (keep the more independent one).
    """
    ids, R = _matrix(returns)
    if len(ids) < 2:
        return _subset(returns, ids)
    corr = np.corrcoef(R.T)
    corr = np.nan_to_num(corr, nan=0.0)
    absc = np.abs(corr)
    np.fill_diagonal(absc, 0.0)
    removed = set()
    while True:
        # find the worst remaining pair
        best = None
        for i in range(len(ids)):
            if i in removed:
                continue
            for j in range(i + 1, len(ids)):
                if j in removed:
                    continue
                if absc[i, j] > threshold and (best is None or absc[i, j] > best[2]):
                    best = (i, j, absc[i, j])
        if best is None:
            break
        i, j, _ = best
        mean_i = absc[i, [k for k in range(len(ids)) if k not in removed and k != i]].mean()
        mean_j = absc[j, [k for k in range(len(ids)) if k not in removed and k != j]].mean()
        removed.add(i if mean_i >= mean_j else j)
    keep = [ids[k] for k in range(len(ids)) if k not in removed]
    return _subset(returns, keep)


def select_k_extremes(returns, *, k: int, measure: str = "sharpe", highest: bool = True) -> Dict[str, list]:
    """Keep the ``k`` assets with the highest (or lowest) ranking metric.

    Args:
        measure: ``"sharpe"`` (mean/std), ``"return"`` (mean), or
            ``"volatility"`` (std).
        highest: keep the top-k (True) or bottom-k (False).
    """
    ids, R = _matrix(returns)
    if not ids:
        return {}
    mu = R.mean(axis=0)
    sd = R.std(axis=0, ddof=1) if R.shape[0] > 1 else np.ones(R.shape[1])
    if measure == "sharpe":
        score = np.where(sd > 0, mu / sd, 0.0)
    elif measure == "return":
        score = mu
    elif measure == "volatility":
        score = sd
    else:
        raise ValueError(f"Unknown measure {measure!r}. Use sharpe|return|volatility.")
    order = np.argsort(score)
    picked = order[-k:] if highest else order[:k]
    keep = [ids[i] for i in sorted(picked)]
    return _subset(returns, keep)


def select_non_dominated(returns, *, risk_measure: str = "volatility") -> Dict[str, list]:
    """Keep the Pareto-efficient assets: no other asset has both a higher mean
    return AND a lower risk. Removes strictly dominated names."""
    ids, R = _matrix(returns)
    if len(ids) < 2:
        return _subset(returns, ids)
    ret = R.mean(axis=0)
    if risk_measure == "volatility":
        risk = R.std(axis=0, ddof=1)
    elif risk_measure == "cvar":
        var = np.percentile(R, 5, axis=0)
        risk = np.array([-R[R[:, j] <= var[j], j].mean() for j in range(R.shape[1])])
    else:
        raise ValueError(f"Unknown risk_measure {risk_measure!r}. Use volatility|cvar.")
    n = len(ids)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if ret[j] >= ret[i] and risk[j] <= risk[i] and (ret[j] > ret[i] or risk[j] < risk[i]):
                dominated[i] = True
                break
    keep = [ids[i] for i in range(n) if not dominated[i]]
    return _subset(returns, keep)
