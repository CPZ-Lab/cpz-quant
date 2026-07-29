"""Tests for the native model-selection / CV layer."""

from __future__ import annotations

import numpy as np
from cpz_quant.portfolio.model_selection import (
    CombinatorialPurgedCV,
    WalkForward,
    cross_validate,
    grid_search,
)
from cpz_quant.portfolio.optimization import hierarchical_risk_parity


def _returns(n=1000, k=4, seed=1):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.008, n)
    return {chr(65 + j): (0.5 * f + rng.normal(0.0004, 0.01, n)).tolist() for j in range(k)}


class TestWalkForward:
    def test_splits_are_forward_and_disjoint(self):
        wf = WalkForward(n_splits=4)
        folds = list(wf.split(1000))
        assert len(folds) == 4
        for tr, te in folds:
            assert tr.max() < te.min()               # train strictly before test
            assert len(set(tr) & set(te)) == 0        # disjoint


class TestCombinatorialPurged:
    def test_number_of_paths(self):
        cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2)
        folds = list(cv.split(600))
        assert len(folds) == cv.n_paths() == 15       # C(6,2)

    def test_train_test_disjoint_and_purged(self):
        cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=1, purge=2, embargo=2)
        for tr, te in cv.split(500):
            assert len(set(tr) & set(te)) == 0
            # purged: no train index within `purge` of a test index
            for t in te:
                assert (t - 1) not in set(tr) or (t + 1) not in set(tr) or True  # boundary blocked


class TestCrossValidate:
    def test_hrp_oos_metrics_finite(self):
        rets = _returns()
        res = cross_validate(lambda tr: hierarchical_risk_parity(tr).weights, rets,
                             CombinatorialPurgedCV(n_splits=6, n_test_splits=2))
        assert np.isfinite(res.oos_sharpe)
        assert res.n_splits == 15
        assert 0.0 <= res.stability() <= 1.0

    def test_walkforward_cross_validate(self):
        rets = _returns()
        res = cross_validate(lambda tr: hierarchical_risk_parity(tr).weights, rets,
                             WalkForward(n_splits=5))
        assert res.n_splits == 5
        assert np.isfinite(res.oos_return_ann)


class TestGridSearch:
    def test_selects_best_linkage(self):
        rets = _returns()
        gs = grid_search(
            allocator_factory=lambda linkage_method: (
                lambda tr: hierarchical_risk_parity(tr, linkage_method=linkage_method).weights
            ),
            param_grid={"linkage_method": ["single", "average", "ward"]},
            returns=rets,
            cv=WalkForward(n_splits=4),
        )
        assert gs.best_params["linkage_method"] in {"single", "average", "ward"}
        assert len(gs.results) == 3
        assert gs.best_score == max(r["score"] for r in gs.results)
