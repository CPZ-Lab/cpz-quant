//! Rust acceleration for CPZ Analytics.
//!
//! Provides high-performance implementations of compute-intensive
//! operations: indicators, covariance estimation, portfolio optimisation,
//! Monte Carlo simulation, and risk parity.
//!
//! Build with maturin: `maturin develop --release`

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;
use rand_distr::Normal;
use rayon::prelude::*;

mod analytics;
mod indicators;
mod covariance;
mod portfolio;

/// Monte Carlo VaR simulation with parallel execution.
///
/// Runs `num_simulations` paths of `horizon_days` length using
/// the given mean and standard deviation of daily returns.
/// Returns a dictionary with VaR, ES, and percentile statistics.
#[pyfunction]
#[pyo3(signature = (daily_returns, num_simulations, horizon_days, seed=None))]
fn monte_carlo_var(
    daily_returns: Vec<f64>,
    num_simulations: usize,
    horizon_days: usize,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    let n = daily_returns.len();
    if n < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 2 returns",
        ));
    }

    let mu: f64 = daily_returns.iter().sum::<f64>() / n as f64;
    let variance: f64 =
        daily_returns.iter().map(|r| (r - mu).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let sigma = variance.sqrt();

    let dist = Normal::new(mu, sigma).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid distribution params: {e}"))
    })?;

    // Parallel Monte Carlo simulation using rayon
    let terminal_values: Vec<f64> = (0..num_simulations)
        .into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(seed.unwrap_or(42) + i as u64);
            let mut cumulative = 1.0_f64;
            for _ in 0..horizon_days {
                let r: f64 = rng.sample(dist);
                cumulative *= 1.0 + r;
            }
            cumulative - 1.0
        })
        .collect();

    let mut sorted = terminal_values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let percentile = |p: f64| -> f64 {
        let idx = ((p / 100.0) * sorted.len() as f64) as usize;
        sorted[idx.min(sorted.len() - 1)]
    };

    let var_95 = percentile(5.0);
    let var_99 = percentile(1.0);

    let es_95: f64 = {
        let cutoff = (0.05 * sorted.len() as f64) as usize;
        let tail = &sorted[..cutoff.max(1)];
        tail.iter().sum::<f64>() / tail.len() as f64
    };

    let es_99: f64 = {
        let cutoff = (0.01 * sorted.len() as f64) as usize;
        let tail = &sorted[..cutoff.max(1)];
        tail.iter().sum::<f64>() / tail.len() as f64
    };

    let mean_ret: f64 = terminal_values.iter().sum::<f64>() / terminal_values.len() as f64;
    let worst = sorted[0];
    let best = sorted[sorted.len() - 1];

    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        dict.set_item("num_simulations", num_simulations)?;
        dict.set_item("horizon_days", horizon_days)?;
        dict.set_item("var_95", (var_95 * 100.0 * 100.0).round() / 100.0)?;
        dict.set_item("var_99", (var_99 * 100.0 * 100.0).round() / 100.0)?;
        dict.set_item("expected_shortfall_95", (es_95 * 100.0 * 100.0).round() / 100.0)?;
        dict.set_item("expected_shortfall_99", (es_99 * 100.0 * 100.0).round() / 100.0)?;
        dict.set_item("mean_return", (mean_ret * 100.0 * 100.0).round() / 100.0)?;
        dict.set_item("worst_case", (worst * 100.0 * 100.0).round() / 100.0)?;
        dict.set_item("best_case", (best * 100.0 * 100.0).round() / 100.0)?;
        Ok(dict.unbind().into_any())
    })
}

/// Compute covariance matrix from a flat returns matrix.
///
/// Takes a flat array of returns (row-major, n_assets x n_days) and
/// returns a flat covariance matrix (n_assets x n_assets).
#[pyfunction]
fn covariance_matrix(returns_flat: Vec<f64>, n_assets: usize, n_days: usize) -> PyResult<Vec<f64>> {
    if returns_flat.len() != n_assets * n_days {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "returns_flat length must equal n_assets * n_days",
        ));
    }

    let means: Vec<f64> = (0..n_assets)
        .map(|i| {
            let row = &returns_flat[i * n_days..(i + 1) * n_days];
            row.iter().sum::<f64>() / n_days as f64
        })
        .collect();

    // Annualized covariance matrix
    let mut cov = vec![0.0_f64; n_assets * n_assets];
    for i in 0..n_assets {
        for j in i..n_assets {
            let mut s = 0.0_f64;
            for k in 0..n_days {
                s += (returns_flat[i * n_days + k] - means[i])
                    * (returns_flat[j * n_days + k] - means[j]);
            }
            let c = s / (n_days as f64 - 1.0) * 252.0;
            cov[i * n_assets + j] = c;
            cov[j * n_assets + i] = c;
        }
    }

    Ok(cov)
}

/// Risk parity optimization using iterative gradient descent.
///
/// Finds weights where each asset contributes equally to total portfolio risk.
/// Returns optimal weights as a Vec<f64>.
#[pyfunction]
fn risk_parity_weights(cov_flat: Vec<f64>, n_assets: usize, max_iter: usize) -> PyResult<Vec<f64>> {
    if cov_flat.len() != n_assets * n_assets {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cov_flat length must equal n_assets^2",
        ));
    }

    let mut weights = vec![1.0 / n_assets as f64; n_assets];
    let target_rc = 1.0 / n_assets as f64;

    for _iter in 0..max_iter {
        let mut sigma_w = vec![0.0_f64; n_assets];
        for i in 0..n_assets {
            for j in 0..n_assets {
                sigma_w[i] += cov_flat[i * n_assets + j] * weights[j];
            }
        }

        let port_var: f64 = weights.iter().zip(sigma_w.iter()).map(|(w, sw)| w * sw).sum();
        let port_vol = port_var.sqrt();

        if port_vol < 1e-15 {
            break;
        }

        let rc: Vec<f64> = weights
            .iter()
            .zip(sigma_w.iter())
            .map(|(w, sw)| w * sw / port_vol)
            .collect();

        let mut new_weights = vec![0.0_f64; n_assets];
        for i in 0..n_assets {
            if rc[i] > 1e-15 {
                new_weights[i] = weights[i] * (target_rc * port_vol / rc[i]).sqrt();
            } else {
                new_weights[i] = weights[i];
            }
        }

        let sum: f64 = new_weights.iter().sum();
        for w in &mut new_weights {
            *w /= sum;
        }

        let max_diff: f64 = weights
            .iter()
            .zip(new_weights.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        weights = new_weights;

        if max_diff < 1e-10 {
            break;
        }
    }

    Ok(weights)
}

/// Python module definition.
#[pymodule]
fn cpz_risk_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // === Existing (untouched) ===
    m.add_function(wrap_pyfunction!(monte_carlo_var, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(risk_parity_weights, m)?)?;

    // === Indicators ===
    m.add_function(wrap_pyfunction!(indicators::rolling_sma, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::ema, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::wma, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::kama, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::rsi, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::macd, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::adx, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::atr, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::bollinger, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::garman_klass_vol, m)?)?;

    // === Covariance ===
    m.add_function(wrap_pyfunction!(covariance::ledoit_wolf, m)?)?;
    m.add_function(wrap_pyfunction!(covariance::ewma_cov, m)?)?;
    m.add_function(wrap_pyfunction!(covariance::denoise_mp, m)?)?;

    // === Portfolio ===
    m.add_function(wrap_pyfunction!(portfolio::hrp_weights, m)?)?;
    m.add_function(wrap_pyfunction!(portfolio::black_litterman_posterior, m)?)?;

    // === Certification analytics ===
    m.add_function(wrap_pyfunction!(analytics::certification_analytics, m)?)?;

    Ok(())
}
