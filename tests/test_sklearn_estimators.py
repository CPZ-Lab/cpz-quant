"""Tests for the optional scikit-learn portfolio estimators."""

from __future__ import annotations

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")  # skip entirely if the extra isn't installed

from cpz_quant.portfolio.sklearn_estimators import (
    HERCEstimator,
    HRPEstimator,
    MeanRiskEstimator,
    NCOEstimator,
    has_sklearn,
)


def _X(n=800, k=4, seed=1):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.008, n)
    return np.column_stack([0.5 * f + rng.normal(0.0004, 0.01 + 0.004 * j, n) for j in range(k)])


@pytest.mark.parametrize("Est", [MeanRiskEstimator, HRPEstimator, HERCEstimator, NCOEstimator])
def test_fit_learns_valid_weights(Est):
    X = _X()
    est = Est().fit(X)
    assert est.weights_.shape == (X.shape[1],)
    assert est.weights_.sum() == pytest.approx(1.0, abs=1e-5)


def test_predict_and_score(Est=HRPEstimator):
    X = _X()
    est = HRPEstimator().fit(X)
    port = est.predict(X)
    assert port.shape == (X.shape[0],)
    assert np.isfinite(est.score(X))


def test_get_set_params_roundtrip():
    est = MeanRiskEstimator(risk_measure="cvar", objective="min_risk")
    est.set_params(risk_measure="variance", objective="max_ratio")
    assert est.get_params()["risk_measure"] == "variance"
    assert est.get_params()["objective"] == "max_ratio"


def test_gridsearchcv_integration():
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    X = _X()
    gs = GridSearchCV(
        HRPEstimator(),
        {"linkage_method": ["single", "average", "ward"]},
        cv=TimeSeriesSplit(n_splits=3),
    )
    gs.fit(X)
    assert gs.best_params_["linkage_method"] in {"single", "average", "ward"}


def test_pipeline_composition():
    from sklearn.pipeline import Pipeline
    X = _X()
    pipe = Pipeline([("alloc", NCOEstimator(objective="min_variance"))])
    pipe.fit(X)
    assert has_sklearn()
    assert pipe.predict(X).shape == (X.shape[0],)
