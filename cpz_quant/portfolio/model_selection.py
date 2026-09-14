"""Model selection for portfolio allocators with explicit fold/path semantics.

Fixed observation-count purge/embargo gaps are not a guarantee against leakage.
Choose gaps for the feature/label horizon and fit preprocessing on training data
only. CPCV is distinct from the certification PBO test's CSCV procedure.

- :class:`WalkForward` — expanding/rolling train window, forward test window.
- :class:`CombinatorialPurgedCV` — all C(N, k) train/test partitions with purge
  + embargo; reconstructs C(N-1, k-1) complete out-of-sample paths.
- :func:`cross_validate` — fit an allocator on each train fold, score its
  out-of-sample portfolio returns.
- :func:`grid_search` — pick the hyperparameters with the best OOS score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from math import comb
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

TRADING_DAYS = 252.0
Allocator = Callable[[Dict[str, list]], Dict[str, float]]


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class WalkForward:
    """Rolling/expanding walk-forward splits over ``n_samples`` time steps."""

    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None, expanding: bool = True):
        self.n_splits = _integer(n_splits, "n_splits", 1)
        self.test_size = None if test_size is None else _integer(test_size, "test_size", 1)
        self.expanding = expanding

    def split(self, n_samples: int):
        n_samples = _integer(n_samples, "n_samples", 1)
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
    Produces ``C(N, k)`` train/test folds, not full paths. Each group appears in
    ``C(N-1, k-1)`` test folds; :meth:`reconstruct_paths` combines their outputs
    into that many complete chronological paths. Embargo starts immediately
    after each test group, so it overlaps (rather than adds to) the post-test
    purge. This positional gap is not event-label-overlap purging.
    """

    def __init__(self, n_splits: int = 6, n_test_splits: int = 2, purge: int = 1, embargo: int = 1):
        self.n_splits = _integer(n_splits, "n_splits", 2)
        self.n_test_splits = _integer(n_test_splits, "n_test_splits", 1)
        if self.n_test_splits >= self.n_splits:
            raise ValueError("n_test_splits must be smaller than n_splits")
        self.purge = _integer(purge, "purge", 0)
        self.embargo = _integer(embargo, "embargo", 0)

    def _groups(self, n_samples):
        n_samples = _integer(n_samples, "n_samples", self.n_splits)
        return np.array_split(np.arange(n_samples), self.n_splits)

    def split(self, n_samples: int):
        groups = self._groups(n_samples)
        for test_combo in combinations(range(self.n_splits), self.n_test_splits):
            test_idx = np.concatenate([groups[g] for g in test_combo])
            blocked = np.zeros(n_samples, dtype=bool)
            for g in test_combo:
                start = max(0, int(groups[g][0]) - self.purge)
                end = min(n_samples, int(groups[g][-1]) + 1 + max(self.purge, self.embargo))
                blocked[start:end] = True
            train_idx = np.flatnonzero(~blocked)
            if train_idx.size < 2:
                raise ValueError("purge/embargo leave fewer than two training observations")
            yield train_idx, np.sort(test_idx)

    def n_folds(self) -> int:
        """Number of train/test fits: C(N, k)."""
        return comb(self.n_splits, self.n_test_splits)

    def n_paths(self) -> int:
        """Number of complete reconstructed paths: C(N-1, k-1)."""
        return comb(self.n_splits - 1, self.n_test_splits - 1)

    def reconstruct_paths(self, fold_returns: Sequence[Sequence[float]], n_samples: int) -> np.ndarray:
        """Return shape ``(n_paths(), n_samples)`` from held-out fold returns.

        Inputs must follow ``split(n_samples)`` order, each vector aligned to
        that fold's sorted test indices. The r-th occurrence of each group is
        assigned to path r. Every held-out value is used exactly once across
        paths, and every observation appears exactly once within each path.
        Paths share original observations and are not independent histories.
        """
        groups = self._groups(n_samples)
        if len(fold_returns) != self.n_folds():
            raise ValueError("fold_returns must contain every train/test fold")
        paths = np.empty((self.n_paths(), n_samples), dtype=float)
        occurrence = np.zeros(self.n_splits, dtype=int)
        for values, combo in zip(fold_returns, combinations(range(self.n_splits), self.n_test_splits)):
            fold = np.asarray(values, dtype=float)
            expected = sum(len(groups[g]) for g in combo)
            if fold.shape != (expected,) or not np.all(np.isfinite(fold)):
                raise ValueError("each fold must have finite returns aligned to all test indices")
            offset = 0
            for g in combo:
                group = groups[g]
                paths[occurrence[g], group] = fold[offset:offset + len(group)]
                occurrence[g] += 1
                offset += len(group)
        return paths


@dataclass
class CVResult:
    """OOS summaries: arithmetic means of complete-path metrics for CPCV.

    Walk-forward has one time-ordered held-out path. ``n_splits`` and
    ``per_split_sharpe`` always describe fits/folds, not full paths.
    """
    oos_sharpe: float
    oos_return_ann: float
    oos_vol_ann: float
    per_split_sharpe: List[float] = field(default_factory=list)
    n_splits: int = 0
    per_path_sharpe: List[float] = field(default_factory=list)
    per_path_return_ann: List[float] = field(default_factory=list)
    per_path_vol_ann: List[float] = field(default_factory=list)
    n_paths: int = 0

    def stability(self) -> float:
        """Fraction of folds with a positive OOS Sharpe."""
        if not self.per_split_sharpe:
            return 0.0
        return float(np.mean([s > 0 for s in self.per_split_sharpe]))


def _as_matrix(returns: Mapping[str, Sequence[float]]) -> Tuple[List[str], np.ndarray]:
    ids = list(returns.keys())
    if not ids:
        raise ValueError("returns must contain at least one asset")
    for aid in ids:
        values = np.asarray(returns[aid], dtype=float)
        if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"returns for {aid!r} require at least two finite observations")
    min_len = min(len(returns[i]) for i in ids)
    R = np.column_stack([np.asarray(returns[i][-min_len:], dtype=float) for i in ids])
    return ids, R


def _indices(values, n, label):
    indices = np.asarray(values)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= n)
            or np.unique(indices).size != indices.size):
        raise ValueError(f"{label} indices must be unique integers within the return series")
    if indices.size < 2:
        raise ValueError(f"each {label} fold requires at least two observations")
    return indices


def _annualized_metrics(values):
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(values.mean() * TRADING_DAYS)
        vol = float(values.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if not np.isfinite(mean) or not np.isfinite(vol):
        raise ValueError("held-out returns must produce finite metrics")
    # Preserve the established zero-volatility score convention.
    sharpe = mean / vol if vol > 0 else 0.0
    if not np.isfinite(sharpe):
        raise ValueError("held-out returns must produce a finite Sharpe score")
    return sharpe, mean, vol


def cross_validate(allocator: Allocator, returns: Mapping[str, Sequence[float]], cv) -> CVResult:
    """Fit ``allocator`` on each train fold and score its OOS portfolio returns.

    Args:
        allocator: ``train_returns_dict -> {asset_id: weight}``.
        returns: ``{asset_id: [returns]}``.
        cv: a splitter with ``.split(n_samples)`` (WalkForward / CombinatorialPurgedCV).

    CPCV summaries average complete reconstructed path metrics, not repeated
    observations pooled as one history. Other splitters must have disjoint test
    indices; their held-out observations form one time-ordered path. Invalid or
    insufficient folds and non-finite allocator outputs raise, never skip.
    """
    ids, R = _as_matrix(returns)
    T = R.shape[0]
    oos_chunks: List[np.ndarray] = []
    test_chunks: List[np.ndarray] = []
    per_split: List[float] = []
    seen = np.zeros(T, dtype=bool)
    cpcv = isinstance(cv, CombinatorialPurgedCV)
    for train_idx, test_idx in cv.split(T):
        train_idx = _indices(train_idx, T, "train")
        test_idx = _indices(test_idx, T, "test")
        if np.intersect1d(train_idx, test_idx).size:
            raise ValueError("train and test indices overlap")
        if not cpcv and np.any(seen[test_idx]):
            raise ValueError("overlapping test folds require CombinatorialPurgedCV path reconstruction")
        seen[test_idx] = True
        train = {ids[j]: R[train_idx, j].tolist() for j in range(len(ids))}
        w = allocator(train)
        wv = np.array([float(w.get(i, 0.0)) for i in ids])
        if not np.all(np.isfinite(wv)):
            raise ValueError("allocator weights must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            port = R[test_idx, :] @ wv
        if not np.all(np.isfinite(port)):
            raise ValueError("held-out portfolio returns must be finite")
        oos_chunks.append(port)
        test_chunks.append(test_idx)
        per_split.append(_annualized_metrics(port)[0])
    if not oos_chunks:
        raise ValueError("cross-validation produced no folds")
    if cpcv:
        paths = cv.reconstruct_paths(oos_chunks, T)
    else:
        order = np.argsort(np.concatenate(test_chunks))
        paths = np.concatenate(oos_chunks)[order][None, :]
    path_metrics = np.asarray([_annualized_metrics(path) for path in paths])
    return CVResult(
        oos_sharpe=float(path_metrics[:, 0].mean()),
        oos_return_ann=float(path_metrics[:, 1].mean()),
        oos_vol_ann=float(path_metrics[:, 2].mean()),
        per_split_sharpe=per_split,
        n_splits=len(per_split),
        per_path_sharpe=path_metrics[:, 0].tolist(),
        per_path_return_ann=path_metrics[:, 1].tolist(),
        per_path_vol_ann=path_metrics[:, 2].tolist(),
        n_paths=len(paths),
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
