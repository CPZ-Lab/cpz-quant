"""grid_search never selects a setting its scorer could not score (CPZ-Lab/cpzai#3175).

Fails on origin/main 7abc99b: an unscored (NaN) first setting stayed "best"
because NaN compares False against every real score.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from cpz_quant.portfolio.model_selection import WalkForward, grid_search


def _returns(n=600, seed=1):
    rng = np.random.default_rng(seed)
    return {k: rng.normal(0.0005, 0.01, n).tolist() for k in "ABC"}


def _factory(cash):
    def allocate(train):
        return {k: (0.0 if cash else 1.0 / len(train)) for k in train}

    return allocate


def _sharpe_unless_flat(result):
    # The all-cash book has no volatility, so it has no Sharpe ratio.
    return float("nan") if result.oos_vol_ann == 0 else result.oos_sharpe


@pytest.mark.parametrize("order", [[True, False], [False, True]])
def test_unscored_setting_is_never_selected_whatever_the_grid_order(order):
    rets = _returns()
    gs = grid_search(_factory, {"cash": order}, rets, WalkForward(n_splits=3),
                     scorer=_sharpe_unless_flat)
    assert gs.best_params == {"cash": False}
    scored = [r["score"] for r in gs.results if not math.isnan(r["score"])]
    assert gs.best_score == max(scored)
    # The unscored setting is still reported, with its blank score.
    assert any(r["params"] == {"cash": True} and math.isnan(r["score"]) for r in gs.results)


@pytest.mark.parametrize("blank", [float("nan"), None, float("inf")])
def test_every_kind_of_blank_score_is_skipped(blank):
    gs = grid_search(_factory, {"cash": [True, False]}, _returns(), WalkForward(n_splits=3),
                     scorer=lambda r: blank if r.oos_vol_ann == 0 else r.oos_sharpe)
    assert gs.best_params == {"cash": False}


def test_no_finite_score_raises_instead_of_returning_a_blank_best():
    with pytest.raises(ValueError, match="none of the 2 settings has a finite score"):
        grid_search(_factory, {"cash": [True, False]}, _returns(), WalkForward(n_splits=3),
                    scorer=lambda r: float("nan"))


def test_empty_grid_raises_instead_of_a_score_of_zero():
    with pytest.raises(ValueError, match="no settings"):
        grid_search(_factory, {"cash": []}, _returns(), WalkForward(n_splits=3))


@pytest.mark.parametrize("bad", ["0.5", np.array([0.1, 0.2]), True])
def test_a_score_that_is_not_a_real_number_is_a_scorer_bug(bad):
    with pytest.raises(TypeError, match="scorer must return a real number or None"):
        grid_search(_factory, {"cash": [True, False]}, _returns(), WalkForward(n_splits=3),
                    scorer=lambda r: bad)


def test_numpy_float_scores_are_real_numbers():
    gs = grid_search(_factory, {"cash": [True, False]}, _returns(), WalkForward(n_splits=3),
                     scorer=lambda r: np.float64("nan") if r.oos_vol_ann == 0 else np.float64(r.oos_sharpe))
    assert gs.best_params == {"cash": False}
