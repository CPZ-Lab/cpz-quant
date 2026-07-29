"""Model selection for portfolio allocators — native, leakage-safe.

Time-series cross-validation and hyperparameter search built on the same
combinatorially-purged machinery as the certification PBO test, so tuning does
not leak future information into the allocation. Pure numpy/itertools — no
scikit-learn required.

- :class:`WalkForward` — expanding/rolling train window, forward test window.
- :class:`CombinatorialPurgedCV` — all C(N, k) train/test partitions with purge
  + embargo (Lopez de Prado); yields a *distribution* of out-of-sample paths.
- :func:`cross_validate` — fit an allocator on each train fold, score its
  out-of-sample portfolio returns.
- :func:`grid_search` — pick the hyperparameters with the best OOS score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

TRADING_DAYS = 252.0
Allocator = Callable[[Dict[str, list]], Dict[str, float]]


class WalkForward:
    """Rolling/expanding walk-forward splits over ``n_samples`` time steps."""

    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None, expanding: bool = True):
        self.n_splits = n_splits
        self.test_size = test_size
        self.expanding = expanding

    def split(self, n_samples: int):
        ts = self.test_size or max(1, n_samples // (self.n_splits + 1))
        first = n_samples - self.n_splits * ts
        if first <= 0:
            raise ValueError("series too short for the requested n_splits/test_size")
        for k in range(self.n_splits):
            test_start = first + k * ts
            test_end = test_start + ts
            train_start = 0 if self.expanding else max(0, test_start - first)
            yield np.arange(train_start, test_start), np.arange(test_start, min(test_end, n_samples))


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (Lopez de Prado 2018).

    Splits the series into ``n_splits`` contiguous groups and, for every
    combination of ``n_test_splits`` groups held out as test, purges ``purge``
    observations around each test boundary and embargoes ``embargo`` after it.
    Produces ``C(n_splits, n_test_splits)`` train/test folds — a distribution of
    backtest paths rather than a single one.
    """

    def __init__(self, n_splits: int = 6, n_test_splits: int = 2, purge: int = 1, embargo: int = 1):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge = purge
        self.embargo = embargo

    def split(self, n_samples: int):
        groups = np.array_split(np.arange(n_samples), self.n_splits)
        for test_combo in combinations(range(self.n_splits), self.n_test_splits):
            test_idx = np.concatenate([groups[g] for g in test_combo])
            test_set = set(test_idx.tolist())
            # purge + embargo around each test index
            blocked = set(test_set)
            for t in test_idx:
                for d in range(1, self.purge + 1):
                    blocked.add(t - d)
                    blocked.add(t + d)
                for d in range(1, self.embargo + 1):
                    blocked.add(t + d)
            train_idx = np.array([i for i in range(n_samples) if i not in blocked], dtype=int)
            yield train_idx, np.sort(test_idx)

    def n_paths(self) -> int:
        from math import comb
        return comb(self.n_splits, self.n_test_splits)


@dataclass
class CVResult:
    oos_sharpe: float
    oos_return_ann: float
    oos_vol_ann: float
    per_split_sharpe: List[float] = field(default_factory=list)
    n_splits: int = 0

    def stability(self) -> float:
        """Fraction of folds with a positive OOS Sharpe."""
        if not self.per_split_sharpe:
            return 0.0
        return float(np.mean([s > 0 for s in self.per_split_sharpe]))


def _as_matrix(returns: Mapping[str, Sequence[float]]) -> Tuple[List[str], np.ndarray]:
    ids = list(returns.keys())
    min_len = min(len(returns[i]) for i in ids)
    R = np.column_stack([np.asarray(returns[i][-min_len:], dtype=float) for i in ids])
    return ids, R


def cross_validate(allocator: Allocator, returns: Mapping[str, Sequence[float]], cv) -> CVResult:
    """Fit ``allocator`` on each train fold and score its OOS portfolio returns.

    Args:
        allocator: ``train_returns_dict -> {asset_id: weight}``.
        returns: ``{asset_id: [returns]}``.
        cv: a splitter with ``.split(n_samples)`` (WalkForward / CombinatorialPurgedCV).
    """
    ids, R = _as_matrix(returns)
    T = R.shape[0]
    oos_chunks: List[np.ndarray] = []
    per_split: List[float] = []
    for train_idx, test_idx in cv.split(T):
        if train_idx.size < 2 or test_idx.size < 2:
            continue
        train = {ids[j]: R[train_idx, j].tolist() for j in range(len(ids))}
        w = allocator(train)
        wv = np.array([float(w.get(i, 0.0)) for i in ids])
        port = R[test_idx, :] @ wv
        oos_chunks.append(port)
        sd = port.std(ddof=1) if port.size > 1 else 0.0
        per_split.append(float((port.mean() / sd) * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0)
    if not oos_chunks:
        return CVResult(0.0, 0.0, 0.0, [], 0)
    oos = np.concatenate(oos_chunks)
    sd = oos.std(ddof=1) if oos.size > 1 else 0.0
    return CVResult(
        oos_sharpe=float((oos.mean() / sd) * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
        oos_return_ann=float(oos.mean() * TRADING_DAYS),
        oos_vol_ann=float(sd * np.sqrt(TRADING_DAYS)),
        per_split_sharpe=per_split,
        n_splits=len(per_split),
    )


@dataclass
class GridSearchResult:
    best_params: dict
    best_score: float
    results: List[dict] = field(default_factory=list)


def grid_search(
    allocator_factory: Callable[..., Allocator],
    param_grid: Mapping[str, Sequence],
    returns: Mapping[str, Sequence[float]],
    cv,
    *,
    scorer: Callable[[CVResult], float] = lambda r: r.oos_sharpe,
) -> GridSearchResult:
    """Grid-search allocator hyperparameters by out-of-sample score.

    Args:
        allocator_factory: ``**params -> allocator`` (an allocator is
            ``train_returns -> weights``).
        param_grid: ``{param: [values]}``.
        scorer: maps a :class:`CVResult` to a scalar to maximize (default OOS Sharpe).
    """
    keys = list(param_grid.keys())
    best = None
    results = []
    for combo in product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        cvres = cross_validate(allocator_factory(**params), returns, cv)
        score = scorer(cvres)
        results.append({"params": params, "score": score, "oos_sharpe": cvres.oos_sharpe})
        if best is None or score > best[1]:
            best = (params, score)
    return GridSearchResult(best_params=best[0] if best else {},
                            best_score=best[1] if best else 0.0, results=results)
