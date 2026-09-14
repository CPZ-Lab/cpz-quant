"""An infeasible research allocation must never become an equal-weight result."""
from types import SimpleNamespace

import numpy as np
import pytest
from cpz_quant.portfolio.optimization import Constraints, min_variance


def returns():
    rng = np.random.default_rng(42)
    return {name: rng.normal(0, 0.01, 100).tolist() for name in ("a", "b", "c")}


def test_infeasible_weight_caps_raise():
    # Three assets capped at 10% cannot sum to 100%. This previously returned
    # one third per asset under the label min_variance.
    with pytest.raises(RuntimeError, match="min_variance"):
        min_variance(returns(), constraints=Constraints(long_only=True, max_weight=0.1))


@pytest.mark.parametrize("weights", [[np.nan, 0, 1], [0.3, 0.3, 0.3], [-0.1, 0.5, 0.6]])
def test_invalid_solver_output_is_not_presented_as_a_portfolio(monkeypatch, weights):
    monkeypatch.setattr(
        "cpz_quant.portfolio.optimization.sp_opt.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=True, x=np.array(weights), message="ok"),
    )
    with pytest.raises(RuntimeError, match="min_variance"):
        min_variance(returns(), constraints=Constraints(long_only=True))


def test_solver_failure_with_feasible_iterate_still_raises(monkeypatch):
    monkeypatch.setattr(
        "cpz_quant.portfolio.optimization.sp_opt.minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=False, x=np.ones(3) / 3, message="Iteration limit reached"
        ),
    )
    with pytest.raises(RuntimeError, match="Iteration limit reached"):
        min_variance(returns())


def test_feasible_long_only_allocation_respects_caps():
    result = min_variance(returns(), constraints=Constraints(long_only=True, max_weight=0.4))
    assert sum(result.weights.values()) == pytest.approx(1, abs=2e-6)
    assert all(0 <= w <= 0.4 + 1e-6 for w in result.weights.values())


def test_short_weights_match_closed_form_minimum_variance():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 0.01, (1000, 2))
    sample = np.column_stack([x[:, 0], 2 * x[:, 0] + x[:, 1]])
    inv_ones = np.linalg.solve(np.cov(sample.T), np.ones(2))
    expected = inv_ones / inv_ones.sum()
    result = min_variance(
        {"a": sample[:, 0].tolist(), "b": sample[:, 1].tolist()},
        constraints=Constraints(long_only=False, min_weight=-2, max_weight=2, max_gross_exposure=4),
    )
    assert result.weights["b"] < 0
    np.testing.assert_allclose(list(result.weights.values()), expected, atol=2e-6)
