"""Tests for copulas + synthetic scenario generation."""

from __future__ import annotations

import numpy as np
import pytest
from cpz_quant.portfolio.copula import (
    ClaytonCopula,
    GaussianCopula,
    GumbelCopula,
    StudentTCopula,
    fit_copula,
    pseudo_observations,
    synthetic_returns,
)
from scipy import stats


def _correlated(n=3000, rho=0.7, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    # Non-normal marginals so marginal-preservation is a real test.
    x0 = 0.001 + 0.01 * z[:, 0]
    x1 = 0.002 + 0.02 * z[:, 1]
    return np.column_stack([x0, x1])


class TestPseudoObs:
    def test_in_unit_interval_and_uniform(self):
        X = _correlated()
        U = pseudo_observations(X)
        assert U.min() > 0 and U.max() < 1
        assert abs(U.mean() - 0.5) < 0.02


class TestGaussian:
    def test_fit_recovers_correlation(self):
        c = GaussianCopula.fit(_correlated(rho=0.7))
        assert c.correlation[0, 1] == pytest.approx(0.7, abs=0.05)

    def test_no_tail_dependence(self):
        c = GaussianCopula.fit(_correlated())
        assert c.tail_dependence() == (0.0, 0.0)

    def test_sample_preserves_dependence(self):
        c = GaussianCopula.fit(_correlated(rho=0.6))
        U = c.sample(5000, random_state=1)
        z = stats.norm.ppf(U)
        assert np.corrcoef(z.T)[0, 1] == pytest.approx(0.6, abs=0.06)


class TestStudentT:
    def test_fit_and_positive_tail_dependence(self):
        c = StudentTCopula.fit(_correlated(rho=0.7))
        assert c.dof >= 3
        lo, up = c.tail_dependence()
        assert lo > 0 and up > 0 and abs(lo - up) < 1e-9  # symmetric, non-zero

    def test_sample_shape_and_range(self):
        c = StudentTCopula.fit(_correlated())
        U = c.sample(2000, random_state=2)
        assert U.shape == (2000, 2)
        assert U.min() > 0 and U.max() < 1


class TestArchimedean:
    def test_clayton_lower_tail_only(self):
        c = ClaytonCopula.fit(_correlated(rho=0.6))
        lo, up = c.tail_dependence()
        assert c.theta > 0 and lo > 0 and up == 0.0

    def test_gumbel_upper_tail_only_and_samples_valid(self):
        c = GumbelCopula.fit(_correlated(rho=0.6))
        lo, up = c.tail_dependence()
        assert c.theta >= 1 and up > 0 and lo == 0.0
        U = c.sample(3000, random_state=3)
        assert np.all((U > 0) & (U < 1))
        # positive rank dependence is reproduced
        tau, _ = stats.kendalltau(U[:, 0], U[:, 1])
        assert tau > 0.2


class TestSynthetic:
    def test_preserves_marginals_and_dependence(self):
        X = _correlated(rho=0.7, seed=5)
        syn = synthetic_returns(X, n_samples=8000, family="student-t", random_state=9)
        assert syn.shape == (8000, 2)
        # Marginals preserved (inverse-ECDF): mean/std close to historical.
        assert syn[:, 0].mean() == pytest.approx(X[:, 0].mean(), abs=0.001)
        assert syn[:, 1].std() == pytest.approx(X[:, 1].std(), abs=0.004)
        # Dependence preserved (sign + rough magnitude).
        assert np.corrcoef(syn.T)[0, 1] == pytest.approx(0.7, abs=0.1)

    def test_bivariate_guard(self):
        with pytest.raises(ValueError):
            fit_copula(np.random.default_rng(0).normal(size=(100, 3)), family="clayton")
