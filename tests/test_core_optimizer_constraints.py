"""Synthetic regression cases for the three core fully invested optimizers."""

from types import SimpleNamespace

import numpy as np
import pytest
from cpz_quant.portfolio.optimization import Constraints, max_sharpe, mean_variance, min_variance

CORE = [mean_variance, min_variance, max_sharpe]


def sample():
    rng = np.random.default_rng(19)
    return {name: rng.normal(0.001, 0.01 * scale, 500).tolist()
            for name, scale in zip(("a", "b", "c"), (1, 3, 8))}


@pytest.mark.parametrize("allocator", [mean_variance, max_sharpe])
def test_infeasible_caps_raise(allocator):
    with pytest.raises(RuntimeError, match=allocator.__name__):
        allocator(sample(), constraints=Constraints(long_only=True, max_weight=0.1))


@pytest.mark.parametrize("allocator", CORE)
def test_positive_long_only_minimum_is_enforced(allocator):
    result = allocator(sample(), constraints=Constraints(long_only=True, min_weight=0.2))
    assert min(result.weights.values()) >= 0.2 - 1e-7
    assert sum(result.weights.values()) == pytest.approx(1, abs=1e-7)


@pytest.mark.parametrize("allocator", [mean_variance, max_sharpe])
@pytest.mark.parametrize("success,weights", [
    (False, [0.2, 0.3, 0.5]),
    (True, [float("nan"), 0, 1]),
    (True, [0.1, 0.2, 0.3]),
    (True, [-0.1, 0.5, 0.6]),
])
def test_invalid_solver_results_raise(monkeypatch, allocator, success, weights):
    monkeypatch.setattr(
        "cpz_quant.portfolio.optimization.sp_opt.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=success, x=np.array(weights), message="test"),
    )
    with pytest.raises(RuntimeError, match=allocator.__name__):
        allocator(sample(), constraints=Constraints(long_only=True))


def test_unattainable_target_return_raises():
    with pytest.raises(RuntimeError, match="mean_variance"):
        mean_variance(sample(), target_return=100, constraints=Constraints(long_only=True))


def test_success_flag_cannot_hide_missed_target(monkeypatch):
    monkeypatch.setattr(
        "cpz_quant.portfolio.optimization.sp_opt.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=True, x=np.ones(3) / 3, message="ok"),
    )
    with pytest.raises(RuntimeError, match="target_return"):
        mean_variance(sample(), target_return=0.02, constraints=Constraints(long_only=True))


def test_feasible_target_is_enforced():
    returns = sample()
    mu = np.array([np.mean(r) * 252 for r in returns.values()])
    target = float(mu @ np.array([0.2, 0.3, 0.5]))
    result = mean_variance(returns, target_return=target, constraints=Constraints(long_only=True))
    assert np.array(list(result.weights.values())) @ mu == pytest.approx(target, abs=1e-7)


@pytest.mark.parametrize("allocator", CORE)
@pytest.mark.parametrize("name,value", [
    ("sector_limits", {"technology": 0.3}),
    ("factor_limits", {"market": (-0.5, 0.5)}),
    ("max_turnover", 0.1),
    ("max_tracking_error", 0.1),
])
def test_unsupported_constraints_are_not_ignored(allocator, name, value):
    with pytest.raises(NotImplementedError, match=name):
        allocator(sample(), constraints=Constraints(**{name: value}))


@pytest.mark.parametrize("allocator", CORE)
def test_net_exposure_cap_must_allow_full_investment(allocator):
    with pytest.raises(ValueError, match="max_net_exposure"):
        allocator(sample(), constraints=Constraints(max_net_exposure=0.5))


@pytest.mark.parametrize("allocator", CORE)
def test_success_flag_cannot_hide_gross_exposure_violation(monkeypatch, allocator):
    monkeypatch.setattr(
        "cpz_quant.portfolio.optimization.sp_opt.minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=True, x=np.array([1.2, 0.3, -0.5]), message="ok"
        ),
    )
    with pytest.raises(RuntimeError, match="max_gross_exposure"):
        allocator(sample(), constraints=Constraints(max_weight=2, max_gross_exposure=1.2))


def test_gross_cap_changes_real_short_allocation():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 0.01, (1000, 2))
    returns = {"a": x[:, 0].tolist(), "b": (2 * x[:, 0] + x[:, 1]).tolist()}
    result = min_variance(returns, constraints=Constraints(
        min_weight=-2, max_weight=2, max_gross_exposure=1.2,
    ))
    assert sum(abs(w) for w in result.weights.values()) <= 1.2 + 1e-7
    np.testing.assert_allclose(list(result.weights.values()), [1.1, -0.1], atol=1e-6)


@pytest.mark.parametrize("allocator", CORE)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_returns_are_not_fabricated_as_zero(allocator, bad):
    returns = sample()
    returns["a"][10] = bad
    with pytest.raises(ValueError, match="finite"):
        allocator(returns)


@pytest.mark.parametrize("allocator", CORE)
def test_reported_weights_keep_solver_precision(allocator):
    result = allocator(sample(), constraints=Constraints(
        long_only=True, min_weight=0.2000004, max_weight=0.5,
    ))
    assert min(result.weights.values()) >= 0.2000004 - 1e-7
    assert sum(result.weights.values()) == pytest.approx(1, abs=1e-7)
