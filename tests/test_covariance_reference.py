"""Covariance estimators against the papers' published code and sklearn.

#3019 (OAS, EWMA, Marchenko-Pastur, Ledoit-Wolf, missing values),
#3146 (Ledoit-Wolf units, EWMA min_periods), #3137 (Gerber).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from cpz_quant.portfolio.covariance import (
    denoise_mp,
    detone_cov,
    ewma_cov,
    factor_model_cov,
    gerber_cov,
    ledoit_wolf,
    oracle_approximating,
    sample_cov,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "allocator_reference.npz"


@pytest.fixture(scope="module")
def ref():
    with np.load(FIXTURE, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def _as_dict(R):
    return {f"a{i}": R[:, i].tolist() for i in range(R.shape[1])}


def _short_wide(seed=3, t=20, n=30):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.01, (t, n)) + rng.normal(0, 0.01, (t, 1))


# ── Oracle Approximating Shrinkage ───────────────────────────────────


def test_oas_matches_sklearn_with_more_assets_than_days():
    sk = pytest.importorskip("sklearn.covariance")
    R = _short_wide()
    ours = oracle_approximating(_as_dict(R), annualize=False)
    # The old eigenvalue formula divided by 1 + (N/T)(lambda - mu)/mu, which
    # nears zero: its largest eigenvalue was about 6e10 here, against 0.0028.
    np.testing.assert_allclose(ours, sk.OAS().fit(R).covariance_, rtol=1e-10, atol=1e-18)


def test_oas_formula_by_hand():
    # Chen et al. (2010) Eq. 23 as scikit-learn writes it, from the MLE S.
    R = _short_wide(seed=8, t=60, n=10)
    X = R - R.mean(axis=0)
    S = X.T @ X / len(R)
    mu, alpha = np.trace(S) / 10, np.mean(S ** 2)
    rho = min((alpha + mu ** 2) / ((60 + 1) * (alpha - mu ** 2 / 10)), 1.0)
    expected = (1 - rho) * S + rho * mu * np.eye(10)
    np.testing.assert_allclose(oracle_approximating(_as_dict(R), annualize=False), expected,
                               rtol=1e-12)


# ── Ledoit-Wolf ──────────────────────────────────────────────────────


def _cov_cor_m(x):
    """Ledoit & Wolf's published covCor.m (constant-correlation target)."""
    t, n = x.shape
    x = x - x.mean(axis=0)
    sample = x.T @ x / t
    var = np.diag(sample)
    sqrtvar = np.sqrt(var)
    r_bar = (np.sum(sample / np.outer(sqrtvar, sqrtvar)) - n) / (n * (n - 1))
    prior = r_bar * np.outer(sqrtvar, sqrtvar)
    prior[np.eye(n, dtype=bool)] = var
    y = x ** 2
    phi_mat = y.T @ y / t - 2 * (x.T @ x) * sample / t + sample ** 2
    phi = phi_mat.sum()
    term1 = (x ** 3).T @ x / t
    help_ = x.T @ x / t
    theta = term1 - np.tile(np.diag(help_), (n, 1)).T * sample - help_ * np.tile(var, (n, 1)).T \
        + np.tile(var, (n, 1)).T * sample
    theta[np.eye(n, dtype=bool)] = 0
    rho = np.diag(phi_mat).sum() + r_bar * np.sum(np.outer(1 / sqrtvar, sqrtvar) * theta)
    gamma = np.linalg.norm(sample - prior, "fro") ** 2
    shrinkage = max(0.0, min(1.0, (phi - rho) / gamma / t))
    return shrinkage * prior + (1 - shrinkage) * sample, shrinkage


def _cov_market_m(x):
    """Ledoit & Wolf's published covMarket.m (single-index target)."""
    t, n = x.shape
    x = x - x.mean(axis=0)
    xmkt = x.mean(axis=1)
    full = np.cov(np.column_stack([x, xmkt]).T) * (t - 1) / t
    covmkt, varmkt = full[:n, n], full[n, n]
    sample = full[:n, :n]
    prior = np.outer(covmkt, covmkt) / varmkt
    prior[np.eye(n, dtype=bool)] = np.diag(sample)
    c = np.linalg.norm(sample - prior, "fro") ** 2
    y = x ** 2
    p = np.sum(y.T @ y) / t - np.sum(sample ** 2)
    rdiag = np.sum(y ** 2) / t - np.sum(np.diag(sample) ** 2)
    z = x * xmkt[:, None]
    v1 = y.T @ z / t - np.tile(covmkt, (n, 1)).T * sample
    roff1 = np.sum(v1 * np.tile(covmkt, (n, 1))) / varmkt - np.sum(np.diag(v1) * covmkt) / varmkt
    v3 = z.T @ z / t - varmkt * sample
    roff3 = np.sum(v3 * np.outer(covmkt, covmkt)) / varmkt ** 2 \
        - np.sum(np.diag(v3) * covmkt ** 2) / varmkt ** 2
    r = rdiag + 2 * roff1 - roff3
    shrinkage = max(0.0, min(1.0, (p - r) / c / t))
    return shrinkage * prior + (1 - shrinkage) * sample, shrinkage


def test_ledoit_wolf_constant_correlation_matches_cov_cor_m(ref):
    R = ref["returns"][:250]
    expected, delta = _cov_cor_m(R)
    res = ledoit_wolf(_as_dict(R), annualize=False)
    # The identity-target intensity formula used before shrank 1.3x too hard here.
    assert res.shrinkage_intensity == pytest.approx(delta, abs=1e-6)
    np.testing.assert_allclose(np.array(res.covariance), expected, rtol=1e-5)


def test_ledoit_wolf_single_factor_matches_cov_market_m_and_pyportfolioopt(ref):
    R = ref["returns"]
    expected, delta = _cov_market_m(R)
    res = ledoit_wolf(_as_dict(R), annualize=False, target="single_factor")
    assert res.shrinkage_intensity == pytest.approx(delta, abs=1e-6)
    assert res.shrinkage_intensity == pytest.approx(float(ref["lw_single_factor_delta"][0]), abs=1e-6)
    np.testing.assert_allclose(np.array(res.covariance), expected, rtol=1e-5)


def test_ledoit_wolf_identity_matches_sklearn(ref):
    sk = pytest.importorskip("sklearn.covariance")
    R = ref["returns"][:250]
    cov, shrinkage = sk.ledoit_wolf(R)
    res = ledoit_wolf(_as_dict(R), annualize=False, target="identity")
    assert res.shrinkage_intensity == pytest.approx(shrinkage, abs=1e-6)
    np.testing.assert_allclose(np.array(res.covariance), cov, rtol=1e-5)


@pytest.mark.parametrize("target", ["constant_correlation", "identity", "single_factor"])
@pytest.mark.parametrize("scale", [0.01, 100.0])
def test_ledoit_wolf_intensity_does_not_depend_on_units(ref, target, scale):
    # A fixed 1e-15 floor on the dispersion term pulled the intensity toward 0
    # for returns scaled by 0.01 (0.0897 -> 0.007 in the report).
    R = ref["returns"][:250]
    base = ledoit_wolf(_as_dict(R), annualize=False, target=target)
    scaled = ledoit_wolf(_as_dict(R * scale), annualize=False, target=target)
    assert scaled.shrinkage_intensity == pytest.approx(base.shrinkage_intensity, abs=1e-6)
    np.testing.assert_allclose(np.array(scaled.covariance), np.array(base.covariance) * scale ** 2,
                               rtol=1e-6)


# ── EWMA ─────────────────────────────────────────────────────────────


def test_ewma_weights_are_normalised(ref):
    R = ref["returns"][:60]
    lam = 0.5 ** (1 / 60)
    w = lam ** np.arange(59, -1, -1)
    expected = np.cov(R.T, aweights=w, ddof=0)  # = pandas ewm(halflife=60).cov(bias=True)
    ours = ewma_cov(_as_dict(R), halflife=60, annualize=False)
    np.testing.assert_allclose(ours, expected, rtol=1e-10)
    # 60 days at a 60-day half-life carry half the weight; unnormalised, the
    # equal-weight book's variance came out at about half of this.
    eq = np.ones(R.shape[1]) / R.shape[1]
    assert eq @ ours @ eq == pytest.approx(eq @ expected @ eq, rel=1e-10)


def test_ewma_below_min_periods_raises(ref):
    R = ref["returns"]
    with pytest.raises(ValueError, match="at least 10 observations"):
        ewma_cov(_as_dict(R[:1]))
    with pytest.raises(ValueError, match="at least 30 observations"):
        ewma_cov(_as_dict(R[:29]), min_periods=30)
    assert ewma_cov(_as_dict(R[:30]), min_periods=30).shape == (8, 8)


# ── Marchenko-Pastur denoising ───────────────────────────────────────


def _ldp_denoised_cov(R, b_width=0.01):
    """Lopez de Prado (2020), MLAM Snippets 2.1-2.5 (pandas Series replaced by arrays)."""
    from scipy.optimize import minimize

    KernelDensity = pytest.importorskip("sklearn.neighbors").KernelDensity
    t, n = R.shape
    q = t / n

    def mp_pdf(var, pts=1000):
        e_min, e_max = var * (1 - (1.0 / q) ** 0.5) ** 2, var * (1 + (1.0 / q) ** 0.5) ** 2
        e_val = np.linspace(e_min, e_max, pts).ravel()
        pdf = (q / (2 * np.pi * var * e_val) * ((e_max - e_val) * (e_val - e_min)) ** 0.5).ravel()
        return e_val, pdf

    def err_pdfs(var, e_val):
        x, pdf0 = mp_pdf(var)
        kde = KernelDensity(kernel="gaussian", bandwidth=b_width).fit(e_val.reshape(-1, 1))
        pdf1 = np.exp(kde.score_samples(x.reshape(-1, 1)))
        return np.sum((pdf1 - pdf0) ** 2)

    corr = np.corrcoef(R, rowvar=False)
    e_val, e_vec = np.linalg.eigh(corr)
    idx = e_val.argsort()[::-1]
    e_val, e_vec = e_val[idx], e_vec[:, idx]
    out = minimize(lambda *x: err_pdfs(*x), 0.5, args=(e_val,), bounds=((1e-5, 1 - 1e-5),))
    var = out["x"][0] if out["success"] else 1
    e_max = var * (1 + (1.0 / q) ** 0.5) ** 2
    n_facts = n - e_val[::-1].searchsorted(e_max)
    e_val_ = e_val.copy()
    e_val_[n_facts:] = e_val_[n_facts:].sum() / float(n - n_facts)
    corr1 = e_vec @ np.diag(e_val_) @ e_vec.T
    sd1 = np.sqrt(np.diag(corr1))
    corr1 = np.clip(corr1 / np.outer(sd1, sd1), -1, 1)
    sd = np.std(R, axis=0, ddof=1)
    return corr1 * np.outer(sd, sd), n_facts


@pytest.mark.parametrize("rows", [126, 600])
def test_denoise_matches_lopez_de_prado(ref, rows):
    R = ref["returns"][:rows]
    expected, n_facts = _ldp_denoised_cov(R)
    assert 0 < n_facts < R.shape[1]  # some eigenvalues are noise
    ours = denoise_mp(_as_dict(R), annualize=False)
    np.testing.assert_allclose(ours, expected, rtol=1e-8, atol=1e-14)
    # Each asset keeps its sample variance (the old version moved it by ~4%),
    # and the noise eigenvalues are replaced, so the matrix is better conditioned.
    np.testing.assert_allclose(np.diag(ours), np.var(R, axis=0, ddof=1), rtol=1e-12)
    assert np.linalg.cond(ours) < np.linalg.cond(np.cov(R.T))


def test_denoise_needs_more_days_than_assets():
    with pytest.raises(ValueError, match="more observations than assets"):
        denoise_mp(_as_dict(_short_wide()))


# ── Gerber ───────────────────────────────────────────────────────────


def _gerber_2021_by_loop(X, c=0.5):
    """g_ij = (n_c - n_d) / (T - n_nn), counted day by day (Gerber et al. 2021)."""
    t, n = X.shape
    sd = X.std(axis=0)
    up, down = X >= c * sd, X <= -c * sd
    g = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            conc = np.sum((up[:, i] & up[:, j]) | (down[:, i] & down[:, j]))
            disc = np.sum((up[:, i] & down[:, j]) | (down[:, i] & up[:, j]))
            both_neutral = np.sum(~(up[:, i] | down[:, i]) & ~(up[:, j] | down[:, j]))
            g[i, j] = (conc - disc) / (t - both_neutral)
    return g * np.outer(sd, sd)


def test_gerber_matches_riskfolio_statistic_1(ref):
    names = [str(n) for n in ref["names"]]
    cols = [names.index(k) for k in ("EQ1", "EQ2", "EQ3", "BD1", "GLD", "CMD", "REIT")]
    W = ref["returns"][:90, cols]
    ours = gerber_cov(_as_dict(W), threshold=0.5, annualize=False)
    np.testing.assert_allclose(ours, ref["gerber_stat1_90d"], rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(ours, _gerber_2021_by_loop(W), rtol=1e-12, atol=1e-18)


def test_gerber_is_positive_semidefinite_on_rolling_windows(ref):
    # The (n_c - n_d) / (n_c + n_d) form failed PSD in every window of this
    # kind and let a long/short book show negative variance.
    R = ref["returns"]
    for start in range(0, R.shape[0] - 90, 15):
        g = gerber_cov(_as_dict(R[start:start + 90]), annualize=False)
        assert np.linalg.eigvalsh(g).min() > -1e-14


# ── Missing values ───────────────────────────────────────────────────


@pytest.mark.parametrize("estimator", [
    sample_cov, ledoit_wolf, oracle_approximating, ewma_cov, factor_model_cov,
    denoise_mp, gerber_cov, detone_cov,
])
def test_missing_value_raises_instead_of_spreading_nan(ref, estimator):
    d = _as_dict(ref["returns"][:200])
    d["a3"][17] = float("nan")
    with pytest.raises(ValueError, match="'a3' at row 17"):
        estimator(d)
