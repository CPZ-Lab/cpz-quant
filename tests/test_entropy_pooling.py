"""Tests for entropy pooling (Meucci fully-flexible views)."""

from __future__ import annotations

import numpy as np
import pytest
from cpz_quant.portfolio.entropy_pooling import (
    entropy_pooling,
    mean_view_rows,
    posterior_moments,
)


@pytest.fixture
def R():
    rng = np.random.default_rng(3)
    return rng.normal([0.0004, 0.0006], [0.01, 0.015], size=(2000, 2))


def test_no_views_returns_prior(R):
    res = entropy_pooling(n_scenarios=R.shape[0])
    assert np.allclose(res.posterior, 1.0 / R.shape[0])
    assert res.relative_entropy == pytest.approx(0.0, abs=1e-12)
    assert res.effective_ratio() == pytest.approx(1.0, abs=1e-9)


def test_mean_view_is_satisfied(R):
    target = 0.0020  # bullish view on asset 0, well above the ~0.0004 sample mean
    A, b = mean_view_rows(R, {0: target})
    res = entropy_pooling(n_scenarios=R.shape[0], a_eq=A, b_eq=b)
    posterior_mean0 = res.posterior @ R[:, 0]
    assert posterior_mean0 == pytest.approx(target, abs=1e-4)
    assert res.converged


def test_stronger_view_distorts_more(R):
    _, _ = mean_view_rows(R, {0: 0.0004})
    A1, b1 = mean_view_rows(R, {0: 0.0010})
    A2, b2 = mean_view_rows(R, {0: 0.0030})  # extreme view
    r1 = entropy_pooling(n_scenarios=R.shape[0], a_eq=A1, b_eq=b1)
    r2 = entropy_pooling(n_scenarios=R.shape[0], a_eq=A2, b_eq=b2)
    # A more extreme view moves further from the prior and uses fewer effective scenarios.
    assert r2.relative_entropy > r1.relative_entropy
    assert r2.effective_number_of_scenarios < r1.effective_number_of_scenarios


def test_effective_scenarios_bounded(R):
    A, b = mean_view_rows(R, {0: 0.0015})
    res = entropy_pooling(n_scenarios=R.shape[0], a_eq=A, b_eq=b)
    assert 0 < res.effective_number_of_scenarios <= R.shape[0]


def test_inequality_view(R):
    # View: E_q[asset 1] <= 0 (bearish cap). Posterior mean should be <= 0 (+tol).
    row = R[:, 1][None, :]
    res = entropy_pooling(n_scenarios=R.shape[0], a_ineq=row, b_ineq=np.array([0.0]))
    assert res.posterior @ R[:, 1] <= 1e-4


def test_posterior_moments_shift(R):
    A, b = mean_view_rows(R, {0: 0.0020})
    res = entropy_pooling(n_scenarios=R.shape[0], a_eq=A, b_eq=b)
    m = posterior_moments(R, res.posterior)
    assert m["mean"].shape == (2,)
    assert m["cov"].shape == (2, 2)
    assert m["mean"][0] == pytest.approx(0.0020, abs=1e-4)
