"""Regime-conditional performance.

The final institutional lens: a strategy's edge is only trustworthy if you know
*when* it works. This decomposes performance by regime — volatility terciles
(calm / normal / stressed) and, when a benchmark is supplied, trend (bull / bear)
— so an allocator can see whether the Sharpe is earned across regimes or is a
single-regime artifact.

Regimes are derived from the series itself (rolling realized volatility and a
benchmark trend filter), so no external macro dataset is required; the analysis
is self-contained and reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence

import numpy as np

TRADING_DAYS = 252.0


@dataclass
class RegimeStats:
    name: str
    share_of_days: float          # fraction of the sample in this regime
    n_days: int
    annualized_return: Optional[float]
    annualized_vol: Optional[float]
    sharpe: Optional[float]
    hit_rate: Optional[float]     # fraction of positive days


@dataclass
class RegimeBreakdown:
    vol_regimes: List[RegimeStats] = field(default_factory=list)
    trend_regimes: List[RegimeStats] = field(default_factory=list)
    worst_vol_regime: Optional[str] = None
    consistent_across_regimes: Optional[bool] = None  # positive Sharpe in every vol regime
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _to_returns(eq: np.ndarray) -> np.ndarray:
    prev, cur = eq[:-1], eq[1:]
    mask = (prev > 0) & np.isfinite(cur) & np.isfinite(prev)
    return (cur[mask] / prev[mask]) - 1.0


def _stats(name: str, rets: np.ndarray, total: int) -> RegimeStats:
    n = rets.size
    if n == 0:
        return RegimeStats(name, 0.0, 0, None, None, None, None)
    mu = float(rets.mean())
    sd = float(rets.std(ddof=1)) if n > 1 else 0.0
    ann_ret = float((np.prod(1 + rets)) ** (TRADING_DAYS / n) - 1.0) if n > 0 else None
    ann_vol = sd * np.sqrt(TRADING_DAYS)
    sharpe = (mu / sd) * np.sqrt(TRADING_DAYS) if sd > 0 else None
    hit = float(np.mean(rets > 0))
    return RegimeStats(
        name=name,
        share_of_days=round(n / total, 4) if total else 0.0,
        n_days=int(n),
        annualized_return=round(ann_ret, 4) if ann_ret is not None else None,
        annualized_vol=round(ann_vol, 4),
        sharpe=round(sharpe, 3) if sharpe is not None else None,
        hit_rate=round(hit, 4),
    )


def regime_conditional_performance(
    equity: Sequence[float],
    benchmark: Optional[Sequence[float]] = None,
    *,
    vol_window: int = 21,
    trend_window: int = 50,
) -> Optional[RegimeBreakdown]:
    """Decompose strategy performance by volatility and trend regime.

    Args:
        equity: daily strategy equity curve.
        benchmark: optional aligned benchmark equity (enables bull/bear trend).
        vol_window: rolling window (days) for the realized-volatility regime.
        trend_window: rolling window (days) for the benchmark trend filter.

    Returns ``None`` when the series is too short.
    """
    eq = np.asarray(equity, dtype=float)
    eq = eq[np.isfinite(eq) & (eq > 0)]
    if eq.size < max(vol_window + 20, 40):
        return None
    rets = _to_returns(eq)
    n = rets.size
    if n < 30:
        return None

    # ── Volatility regimes: rolling realized vol -> terciles ──────────────────
    roll_vol = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - vol_window + 1)
        w = rets[lo : i + 1]
        if w.size >= 2:
            roll_vol[i] = w.std(ddof=1)
    valid = np.isfinite(roll_vol)
    breakdown = RegimeBreakdown()
    if valid.sum() >= 30:
        q1, q2 = np.quantile(roll_vol[valid], [1 / 3, 2 / 3])
        labels = ["calm", "normal", "stressed"]
        buckets = {k: [] for k in labels}  # type: ignore[var-annotated]
        for i in range(n):
            if not valid[i]:
                continue
            rv = roll_vol[i]
            key = "calm" if rv <= q1 else ("normal" if rv <= q2 else "stressed")
            buckets[key].append(rets[i])
        total_valid = int(valid.sum())
        breakdown.vol_regimes = [_stats(k, np.asarray(buckets[k]), total_valid) for k in labels]
        sharpes = [r.sharpe for r in breakdown.vol_regimes if r.sharpe is not None]
        breakdown.consistent_across_regimes = bool(sharpes) and all(s > 0 for s in sharpes)
        ranked = [r for r in breakdown.vol_regimes if r.sharpe is not None]
        if ranked:
            breakdown.worst_vol_regime = min(ranked, key=lambda r: r.sharpe).name  # type: ignore[arg-type,return-value]

    # ── Trend regimes: benchmark above/below its rolling mean ────────────────
    if benchmark is not None and len(benchmark) == len(equity):
        bench = np.asarray(benchmark, dtype=float)
        bench = bench[np.isfinite(np.asarray(equity, dtype=float)) & (np.asarray(equity, dtype=float) > 0)]
        if bench.size == eq.size and np.all(bench > 0):
            # Align a trend flag to each return day (index i corresponds to eq[i+1]).
            bull_mask = []
            for i in range(1, eq.size):
                lo = max(0, i - trend_window + 1)
                ma = bench[lo : i + 1].mean()
                bull_mask.append(bench[i] >= ma)
            bull_mask = np.asarray(bull_mask, dtype=bool)  # type: ignore[assignment]
            if bull_mask.size == n:  # type: ignore[attr-defined]
                breakdown.trend_regimes = [
                    _stats("bull", rets[bull_mask], n),
                    _stats("bear", rets[~bull_mask], n),  # type: ignore[operator]
                ]

    if not breakdown.vol_regimes and not breakdown.trend_regimes:
        return None
    return breakdown
