"""Probability of Backtest Overfitting (PBO) via Combinatorially-Symmetric
Cross-Validation (CSCV).

Bailey, Borwein, Lopez de Prado & Zhu (2015), "The Probability of Backtest
Overfitting". This is the definitive multiple-testing / selection-bias test a
top quant shop runs before trusting an optimized strategy: across every
in-sample / out-of-sample split of the trial matrix, how often does the
configuration that looked best in-sample fall below the median out-of-sample?

Requires the per-trial return matrix (T observations x N configurations), so it
lives in the SDK / backtest engine, not the edge certification layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional

import numpy as np


@dataclass
class PBOResult:
    pbo: float                    # P(best-IS config is below-median OOS), in [0, 1]
    n_trials: int
    n_splits: int
    n_combinations: int
    mean_logit: float             # mean logit of OOS relative rank; < 0 => overfit-leaning
    performance_degradation: Optional[float]  # slope of OOS vs IS performance (negative = overfit)


def _sharpe(x: np.ndarray) -> np.ndarray:
    """Column-wise Sharpe (mean/std, ddof=1) of a (rows x cols) block."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(sd > 0, mu / sd, 0.0)
    return s


def probability_of_backtest_overfitting(
    returns_matrix, n_splits: int = 16,
) -> PBOResult:
    """Compute PBO via CSCV.

    Args:
        returns_matrix: array-like of shape (T, N) — per-period returns for each
            of N strategy configurations (trials).
        n_splits: number of disjoint sub-samples S (even). C(S, S/2) IS/OOS
            splits are evaluated.

    Returns a :class:`PBOResult`. A PBO above ~0.5 means the optimization is more
    likely than not overfit.
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T observations x N trials)")
    T, N = M.shape
    if N < 2:
        raise ValueError("need at least 2 trials/configurations to assess overfitting")
    if n_splits % 2 != 0:
        n_splits += 1
    if T < n_splits * 2:
        raise ValueError(f"need at least {n_splits * 2} observations for {n_splits} splits")

    # Disjoint, (near-)equal contiguous blocks.
    blocks = np.array_split(np.arange(T), n_splits)
    idx = list(range(n_splits))
    half = n_splits // 2

    logits: List[float] = []
    is_perf_all: List[float] = []
    oos_perf_all: List[float] = []

    for is_blocks in combinations(idx, half):
        is_set = set(is_blocks)
        is_rows = np.concatenate([blocks[b] for b in idx if b in is_set])
        oos_rows = np.concatenate([blocks[b] for b in idx if b not in is_set])

        is_perf = _sharpe(M[is_rows])
        oos_perf = _sharpe(M[oos_rows])

        n_star = int(np.argmax(is_perf))  # best config in-sample
        # Relative OOS rank of the IS-best config, in (0, 1).
        order = np.argsort(oos_perf)  # ascending
        rank = int(np.where(order == n_star)[0][0]) + 1
        omega = rank / (N + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(omega / (1 - omega))))

        is_perf_all.append(float(is_perf[n_star]))
        oos_perf_all.append(float(oos_perf[n_star]))

    logits_arr = np.asarray(logits)
    pbo = float(np.mean(logits_arr <= 0.0))

    # Performance degradation: OLS slope of OOS-best-config perf vs its IS perf.
    degradation: Optional[float] = None
    x = np.asarray(is_perf_all)
    y = np.asarray(oos_perf_all)
    if x.size >= 2 and x.std() > 0:
        degradation = float(np.polyfit(x, y, 1)[0])

    return PBOResult(
        pbo=round(pbo, 4),
        n_trials=N,
        n_splits=n_splits,
        n_combinations=len(logits),
        mean_logit=round(float(logits_arr.mean()), 4),
        performance_degradation=None if degradation is None else round(degradation, 4),
    )
