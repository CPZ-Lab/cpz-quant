"""Path reconstruction is distinct from concatenating overlapping test folds."""

from itertools import combinations

import numpy as np
import pytest
from cpz_quant.portfolio.model_selection import CombinatorialPurgedCV, WalkForward, cross_validate


@pytest.mark.parametrize("groups,test_groups,folds,paths", [
    (6, 2, 15, 5), (5, 1, 5, 1), (4, 3, 4, 3),
])
def test_distinct_fold_and_path_counts(groups, test_groups, folds, paths):
    cv = CombinatorialPurgedCV(groups, test_groups)
    assert cv.n_paths() == paths
    assert cv.n_folds() == folds


def test_each_path_covers_every_observation_once_and_uses_every_held_out_value():
    cv = CombinatorialPurgedCV(6, 2, purge=0, embargo=0)
    n = 61  # Unequal group sizes must not drop the tail or shift observations.
    folds = list(cv.split(n))
    values = [1000 * split + test for split, (_, test) in enumerate(folds)]
    paths = cv.reconstruct_paths(values, n)
    assert paths.shape == (5, n)
    np.testing.assert_array_equal(paths % 1000, np.tile(np.arange(n), (5, 1)))
    np.testing.assert_array_equal(np.sort(paths.ravel()), np.sort(np.concatenate(values)))
    # Canonical first path from the paper's six-group/two-test-group example.
    groups = np.array_split(np.arange(n), 6)
    for group, split in zip(groups, [0, 0, 1, 2, 3, 4]):
        np.testing.assert_array_equal(paths[0, group], 1000 * split + group)


def test_path_metrics_are_computed_on_complete_paths_not_pooled_folds():
    cv = CombinatorialPurgedCV(4, 2, purge=0, embargo=0)
    n = 24
    returns = {"a": np.linspace(-0.03, 0.04, n).tolist()}
    fits = []

    def allocator(train):
        scale = len(fits) + 1
        fits.append(scale)
        return {"a": scale}

    result = cross_validate(allocator, returns, cv)
    groups = np.array_split(np.arange(n), 4)
    expected = np.empty((3, n))
    occurrences = np.zeros(4, dtype=int)
    for fold, combo in enumerate(combinations(range(4), 2)):
        for group in combo:
            expected[occurrences[group], groups[group]] = (
                np.asarray(returns["a"])[groups[group]] * (fold + 1)
            )
            occurrences[group] += 1
    means = expected.mean(axis=1) * 252
    vols = expected.std(axis=1, ddof=1) * np.sqrt(252)
    sharpes = means / vols
    assert result.n_splits == 6
    assert result.n_paths == 3
    np.testing.assert_allclose(result.per_path_sharpe, sharpes)
    assert result.oos_sharpe == pytest.approx(sharpes.mean())
    assert result.oos_return_ann == pytest.approx(means.mean())
    assert result.oos_vol_ann == pytest.approx(vols.mean())
    pooled_sharpe = expected.mean() / expected.std(ddof=1) * np.sqrt(252)
    assert abs(result.oos_sharpe - pooled_sharpe) > 1e-4


@pytest.mark.parametrize("kwargs", [
    {"n_splits": 1}, {"n_test_splits": 0}, {"n_test_splits": 6},
    {"n_splits": 3.5}, {"purge": -1}, {"embargo": -1},
])
def test_invalid_cpcv_config_raises(kwargs):
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(**kwargs)


def test_nonempty_groups_are_required():
    with pytest.raises(ValueError):
        list(CombinatorialPurgedCV(6, 2).split(5))


def test_insufficient_training_data_is_not_silently_skipped():
    with pytest.raises(ValueError, match="train"):
        cross_validate(lambda train: {"a": 1}, {"a": [0.1] * 12},
                       CombinatorialPurgedCV(3, 1, purge=100))


def test_walkforward_aggregates_one_chronological_path():
    returns = {"a": np.linspace(-0.01, 0.02, 30).tolist()}
    result = cross_validate(lambda train: {"a": 1}, returns, WalkForward(3, test_size=5))
    expected = np.asarray(returns["a"])[15:]
    assert result.n_paths == 1
    assert result.oos_sharpe == pytest.approx(expected.mean() / expected.std(ddof=1) * np.sqrt(252))
    assert result.per_path_sharpe == [result.oos_sharpe]


def test_custom_overlapping_splitter_is_rejected():
    class Overlapping:
        def split(self, n):
            yield np.arange(4), np.arange(4, 8)
            yield np.arange(4), np.arange(6, 10)

    with pytest.raises(ValueError, match="overlap"):
        cross_validate(lambda train: {"a": 1}, {"a": [0.1] * 10}, Overlapping())


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_allocator_output_raises(bad):
    with pytest.raises(ValueError, match="finite"):
        cross_validate(lambda train: {"a": bad}, {"a": [0.1] * 30}, WalkForward(3))


def test_reconstruction_rejects_incomplete_fold_outputs():
    cv = CombinatorialPurgedCV(4, 2)
    with pytest.raises(ValueError, match="fold"):
        cv.reconstruct_paths([np.ones(10)], 40)


@pytest.mark.parametrize("purge,embargo", [(2, 0), (0, 3), (2, 5)])
def test_real_purge_and_post_test_embargo_boundaries(purge, embargo):
    cv = CombinatorialPurgedCV(5, 1, purge=purge, embargo=embargo)
    for train, test in cv.split(50):
        start, end = test[0], test[-1]
        blocked = set(range(max(0, start - purge), min(50, end + max(purge, embargo) + 1)))
        assert not (set(train) & blocked)
        assert set(train) == set(range(50)) - blocked
