"""scikit-learn-compatible portfolio estimators.

Optional integration: wraps the native cpz_quant.portfolio allocators as scikit-learn
estimators so they compose in ``Pipeline`` and tune under ``GridSearchCV`` /
``RandomizedSearchCV``. This is the ONLY part of cpz_quant.portfolio that uses
scikit-learn, and it is an optional extra — install with
``pip install 'cpz-ai[sklearn]'``. The base package has no scikit-learn
dependency; the pure-numpy allocators and :mod:`cpz_quant.portfolio.model_selection`
cover the same ground without it.

Convention: ``X`` is a (T observations, N assets) return matrix. ``fit`` learns
``weights_``; ``predict``/``transform`` return the portfolio return series;
``score`` returns the annualized Sharpe (higher is better).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - exercised only without sklearn
    _HAS_SKLEARN = False

    class BaseEstimator:  # type: ignore[no-redef]  # minimal fallback so the module imports
        def get_params(self, deep=True):
            return {k: getattr(self, k) for k in getattr(self, "_param_names", [])}

        def set_params(self, **params):
            for k, v in params.items():
                setattr(self, k, v)
            return self

    class TransformerMixin:  # type: ignore[no-redef]  # noqa: D401
        pass

from .optimization import (
    hierarchical_equal_risk_contribution,
    hierarchical_risk_parity,
    nested_clustered_optimization,
)
from .risk_measures import RiskMeasure, mean_risk_optimize

TRADING_DAYS = 252.0


def _to_dict(X: np.ndarray):
    X = np.asarray(X, dtype=float)
    return {f"asset_{i}": X[:, i].tolist() for i in range(X.shape[1])}, X.shape[1]


class _BasePortfolio(BaseEstimator, TransformerMixin):
    """Shared fit/predict/score for portfolio estimators."""

    def _allocate(self, X):  # -> weights array aligned to columns
        raise NotImplementedError

    def fit(self, X, y=None):
        self.weights_ = np.asarray(self._allocate(np.asarray(X, dtype=float)), dtype=float)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.weights_

    def transform(self, X):
        return self.predict(X).reshape(-1, 1)

    def score(self, X, y=None) -> float:
        port = self.predict(X)
        sd = port.std(ddof=1) if port.size > 1 else 0.0
        return float((port.mean() / sd) * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


class MeanRiskEstimator(_BasePortfolio):
    """Mean-risk optimizer as a scikit-learn estimator."""

    _param_names = ["risk_measure", "objective", "beta", "min_weight", "max_weight"]

    def __init__(self, risk_measure: str = "cvar", objective: str = "min_risk",
                 beta: float = 0.95, min_weight: float = 0.0, max_weight: float = 1.0):
        self.risk_measure = risk_measure
        self.objective = objective
        self.beta = beta
        self.min_weight = min_weight
        self.max_weight = max_weight

    def _allocate(self, X):
        res = mean_risk_optimize(
            X, risk_measure=RiskMeasure(self.risk_measure), objective=self.objective,
            beta=self.beta, min_weight=self.min_weight, max_weight=self.max_weight,
        )
        return list(res.weights.values())


class HRPEstimator(_BasePortfolio):
    """Hierarchical Risk Parity as a scikit-learn estimator."""

    _param_names = ["linkage_method"]

    def __init__(self, linkage_method: str = "single"):
        self.linkage_method = linkage_method

    def _allocate(self, X):
        d, n = _to_dict(X)
        w = hierarchical_risk_parity(d, linkage_method=self.linkage_method).weights
        return [w[f"asset_{i}"] for i in range(n)]


class HERCEstimator(_BasePortfolio):
    """Hierarchical Equal Risk Contribution as a scikit-learn estimator."""

    _param_names = ["linkage_method", "n_clusters"]

    def __init__(self, linkage_method: str = "ward", n_clusters: Optional[int] = None):
        self.linkage_method = linkage_method
        self.n_clusters = n_clusters

    def _allocate(self, X):
        d, n = _to_dict(X)
        w = hierarchical_equal_risk_contribution(
            d, linkage_method=self.linkage_method, n_clusters=self.n_clusters).weights
        return [w[f"asset_{i}"] for i in range(n)]


class NCOEstimator(_BasePortfolio):
    """Nested Clustered Optimization as a scikit-learn estimator."""

    _param_names = ["objective", "linkage_method", "n_clusters"]

    def __init__(self, objective: str = "min_variance", linkage_method: str = "ward",
                 n_clusters: Optional[int] = None):
        self.objective = objective
        self.linkage_method = linkage_method
        self.n_clusters = n_clusters

    def _allocate(self, X):
        d, n = _to_dict(X)
        w = nested_clustered_optimization(
            d, objective=self.objective, linkage_method=self.linkage_method,
            n_clusters=self.n_clusters).weights
        return [w[f"asset_{i}"] for i in range(n)]


def has_sklearn() -> bool:
    """True when scikit-learn is available for full Pipeline/GridSearchCV support."""
    return _HAS_SKLEARN
