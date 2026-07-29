//! Rust-accelerated portfolio optimisation.

use pyo3::prelude::*;

/// Hierarchical Risk Parity weights.
///
/// Takes a flat correlation matrix and per-asset volatilities,
/// performs quasi-diagonalisation and recursive bisection.
#[pyfunction]
pub fn hrp_weights(corr_flat: Vec<f64>, vol: Vec<f64>, n_assets: usize) -> PyResult<Vec<f64>> {
    if corr_flat.len() != n_assets * n_assets || vol.len() != n_assets {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Dimension mismatch: corr_flat must be n^2, vol must be n",
        ));
    }

    // Distance matrix: d_ij = sqrt(0.5 * (1 - corr_ij))
    let mut dist = vec![0.0_f64; n_assets * n_assets];
    for i in 0..n_assets {
        for j in 0..n_assets {
            let c = corr_flat[i * n_assets + j].clamp(-1.0, 1.0);
            dist[i * n_assets + j] = (0.5 * (1.0 - c)).max(0.0).sqrt();
        }
    }

    // Simple single-linkage clustering (greedy)
    let mut merged = vec![false; n_assets];
    let mut clusters: Vec<Vec<usize>> = (0..n_assets).map(|i| vec![i]).collect();

    for _ in 0..(n_assets - 1) {
        let mut best_dist = f64::INFINITY;
        let mut best_i = 0;
        let mut best_j = 0;

        for i in 0..clusters.len() {
            if merged[i] {
                continue;
            }
            for j in (i + 1)..clusters.len() {
                if merged[j] {
                    continue;
                }
                // Single linkage: minimum distance between clusters
                let mut min_d = f64::INFINITY;
                for &ai in &clusters[i] {
                    for &aj in &clusters[j] {
                        let d = dist[ai * n_assets + aj];
                        if d < min_d {
                            min_d = d;
                        }
                    }
                }
                if min_d < best_dist {
                    best_dist = min_d;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        // Merge best_j into best_i
        let j_cluster = clusters[best_j].clone();
        clusters[best_i].extend(j_cluster);
        merged[best_j] = true;
    }

    // Find the remaining cluster (the sorted order)
    let sorted_idx: Vec<usize> = clusters
        .into_iter()
        .zip(merged.iter())
        .filter(|(_, &m)| !m)
        .flat_map(|(c, _)| c)
        .collect();

    // Build covariance from correlation * vol
    let mut cov = vec![0.0_f64; n_assets * n_assets];
    for i in 0..n_assets {
        for j in 0..n_assets {
            cov[i * n_assets + j] = corr_flat[i * n_assets + j] * vol[i] * vol[j];
        }
    }

    // Recursive bisection
    let mut weights = vec![1.0_f64; n_assets];

    fn bisect(
        sorted: &[usize],
        cov: &[f64],
        n: usize,
        weights: &mut Vec<f64>,
    ) {
        if sorted.len() <= 1 {
            return;
        }
        let half = sorted.len() / 2;
        let left = &sorted[..half];
        let right = &sorted[half..];

        let var_left = cluster_variance(left, cov, n);
        let var_right = cluster_variance(right, cov, n);

        let inv_left = 1.0 / var_left.max(1e-15).sqrt();
        let inv_right = 1.0 / var_right.max(1e-15).sqrt();
        let alpha = inv_left / (inv_left + inv_right);

        for &i in left {
            weights[i] *= alpha;
        }
        for &i in right {
            weights[i] *= 1.0 - alpha;
        }

        bisect(left, cov, n, weights);
        bisect(right, cov, n, weights);
    }

    fn cluster_variance(items: &[usize], cov: &[f64], n: usize) -> f64 {
        let k = items.len();
        if k == 1 {
            return cov[items[0] * n + items[0]];
        }
        // Inverse-variance weights within cluster
        let inv_diag: Vec<f64> = items
            .iter()
            .map(|&i| 1.0 / cov[i * n + i].max(1e-15))
            .collect();
        let sum_inv: f64 = inv_diag.iter().sum();
        let w: Vec<f64> = inv_diag.iter().map(|v| v / sum_inv).collect();

        let mut var = 0.0_f64;
        for a in 0..k {
            for b in 0..k {
                var += w[a] * w[b] * cov[items[a] * n + items[b]];
            }
        }
        var
    }

    bisect(&sorted_idx, &cov, n_assets, &mut weights);

    // Normalize
    let total: f64 = weights.iter().sum();
    if total > 1e-15 {
        for w in weights.iter_mut() {
            *w /= total;
        }
    }

    Ok(weights)
}

/// Black-Litterman posterior returns and covariance.
///
/// Returns (posterior_returns, posterior_cov_flat).
#[pyfunction]
pub fn black_litterman_posterior(
    pi: Vec<f64>,          // equilibrium returns (n,)
    cov_flat: Vec<f64>,    // covariance matrix (n*n)
    p_flat: Vec<f64>,      // view matrix (k*n)
    q: Vec<f64>,           // view returns (k,)
    omega_flat: Vec<f64>,  // view uncertainty (k*k)
    tau: f64,
    n_assets: usize,
    n_views: usize,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if pi.len() != n_assets
        || cov_flat.len() != n_assets * n_assets
        || p_flat.len() != n_views * n_assets
        || q.len() != n_views
        || omega_flat.len() != n_views * n_views
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Dimension mismatch in Black-Litterman inputs",
        ));
    }

    // tau * Sigma
    let mut tau_sigma = vec![0.0_f64; n_assets * n_assets];
    for i in 0..(n_assets * n_assets) {
        tau_sigma[i] = tau * cov_flat[i];
    }

    // Simplified BL: posterior_mu = pi + tau*Sigma*P' * inv(P*tau*Sigma*P' + Omega) * (Q - P*pi)
    // For simplicity, compute via the direct formula

    // P * pi
    let mut p_pi = vec![0.0_f64; n_views];
    for k in 0..n_views {
        for j in 0..n_assets {
            p_pi[k] += p_flat[k * n_assets + j] * pi[j];
        }
    }

    // Q - P*pi
    let mut q_minus_ppi = vec![0.0_f64; n_views];
    for k in 0..n_views {
        q_minus_ppi[k] = q[k] - p_pi[k];
    }

    // P * tau_Sigma * P' + Omega (k x k)
    let mut m = vec![0.0_f64; n_views * n_views];
    for i in 0..n_views {
        for j in 0..n_views {
            let mut val = omega_flat[i * n_views + j];
            for a in 0..n_assets {
                for b in 0..n_assets {
                    val += p_flat[i * n_assets + a] * tau_sigma[a * n_assets + b] * p_flat[j * n_assets + b];
                }
            }
            m[i * n_views + j] = val;
        }
    }

    // Invert M (for small k, direct formula works)
    let m_inv = if n_views == 1 {
        if m[0].abs() > 1e-15 {
            vec![1.0 / m[0]]
        } else {
            vec![0.0]
        }
    } else {
        // Simple Gauss-Jordan for small matrices
        let mut aug = vec![0.0_f64; n_views * 2 * n_views];
        for i in 0..n_views {
            for j in 0..n_views {
                aug[i * 2 * n_views + j] = m[i * n_views + j];
            }
            aug[i * 2 * n_views + n_views + i] = 1.0;
        }

        for col in 0..n_views {
            let pivot = aug[col * 2 * n_views + col];
            if pivot.abs() < 1e-15 {
                continue;
            }
            for j in 0..(2 * n_views) {
                aug[col * 2 * n_views + j] /= pivot;
            }
            for row in 0..n_views {
                if row == col {
                    continue;
                }
                let factor = aug[row * 2 * n_views + col];
                for j in 0..(2 * n_views) {
                    aug[row * 2 * n_views + j] -= factor * aug[col * 2 * n_views + j];
                }
            }
        }

        let mut inv = vec![0.0_f64; n_views * n_views];
        for i in 0..n_views {
            for j in 0..n_views {
                inv[i * n_views + j] = aug[i * 2 * n_views + n_views + j];
            }
        }
        inv
    };

    // M_inv * (Q - P*pi)
    let mut m_inv_q = vec![0.0_f64; n_views];
    for i in 0..n_views {
        for j in 0..n_views {
            m_inv_q[i] += m_inv[i * n_views + j] * q_minus_ppi[j];
        }
    }

    // tau_Sigma * P' * M_inv_q
    let mut adjustment = vec![0.0_f64; n_assets];
    for i in 0..n_assets {
        for k in 0..n_views {
            let mut ts_p = 0.0_f64;
            for j in 0..n_assets {
                ts_p += tau_sigma[i * n_assets + j] * p_flat[k * n_assets + j];
            }
            adjustment[i] += ts_p * m_inv_q[k];
        }
    }

    // Posterior returns: pi + adjustment
    let mut posterior = vec![0.0_f64; n_assets];
    for i in 0..n_assets {
        posterior[i] = pi[i] + adjustment[i];
    }

    // Posterior covariance (simplified: Sigma + tau*Sigma)
    let mut post_cov = vec![0.0_f64; n_assets * n_assets];
    for i in 0..(n_assets * n_assets) {
        post_cov[i] = cov_flat[i] + tau_sigma[i];
    }

    Ok((posterior, post_cov))
}
