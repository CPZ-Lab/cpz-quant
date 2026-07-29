//! Certification analytics — institutional risk metrics from an equity curve.
//!
//! Fast path for `cpz.certification.analytics`. Computes downside-aware ratios,
//! tail risk, drawdown geometry, statistical significance (López de Prado
//! MinTRL) and benchmark-conditional behavior from a daily equity series (and an
//! optional aligned benchmark series). Values are returned RAW (unrounded); the
//! Python layer rounds so the Rust and pure-Python paths are bit-parity.
//!
//! Returns Python `None` when the series is too short to be meaningful (mirrors
//! the TypeScript engine's `computeRiskAnalytics`).

use pyo3::prelude::*;
use pyo3::types::PyDict;

const TRADING_DAYS: f64 = 252.0;

fn to_returns(eq: &[f64]) -> Vec<f64> {
    let mut r = Vec::with_capacity(eq.len().saturating_sub(1));
    for i in 1..eq.len() {
        let p = eq[i - 1];
        if p > 0.0 && eq[i].is_finite() && p.is_finite() {
            r.push(eq[i] / p - 1.0);
        }
    }
    r
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn std_dev(xs: &[f64], m: f64) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() as f64 - 1.0)).sqrt()
}

fn percentile(xs: &[f64], p: f64) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (s.len() as f64 - 1.0) * p;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        s[lo]
    } else {
        s[lo] + (s[hi] - s[lo]) * (idx - lo as f64)
    }
}

/// Core numeric result. `None` fields mirror "could not compute".
pub struct Analytics {
    pub observations: usize,
    pub annualized_return: Option<f64>,
    pub annualized_vol: Option<f64>,
    pub sharpe: Option<f64>,
    pub sortino: Option<f64>,
    pub calmar: Option<f64>,
    pub max_drawdown: Option<f64>,
    pub ulcer_index: Option<f64>,
    pub max_time_under_water_days: Option<i64>,
    pub downside_deviation: Option<f64>,
    pub var95: Option<f64>,
    pub cvar95: Option<f64>,
    pub cvar99: Option<f64>,
    pub tail_ratio: Option<f64>,
    pub skew: Option<f64>,
    pub excess_kurtosis: Option<f64>,
    pub min_trl_years: Option<f64>,
    pub track_years: Option<f64>,
    pub significant: Option<bool>,
    pub beta: Option<f64>,
    pub correlation: Option<f64>,
    pub up_capture: Option<f64>,
    pub down_capture: Option<f64>,
    pub crisis_return: Option<f64>,
    pub crisis_benchmark_return: Option<f64>,
}

/// Pure-Rust computation, unit-testable without Python.
pub fn compute(equity: &[f64], benchmark: Option<&[f64]>) -> Option<Analytics> {
    // Joint clean: keep indices where the portfolio value is finite and > 0.
    let mut eq: Vec<f64> = Vec::with_capacity(equity.len());
    let mut bench: Vec<f64> = Vec::new();
    let has_bench = benchmark.map_or(false, |b| b.len() == equity.len());
    for (i, &p) in equity.iter().enumerate() {
        if p.is_finite() && p > 0.0 {
            eq.push(p);
            if has_bench {
                bench.push(benchmark.unwrap()[i]);
            }
        }
    }
    if eq.len() < 30 {
        return None;
    }
    let rets = to_returns(&eq);
    let n = rets.len();
    if n < 20 {
        return None;
    }

    let mu = mean(&rets);
    let sd = std_dev(&rets, mu);
    let ann_return = (eq[eq.len() - 1] / eq[0]).powf(TRADING_DAYS / n as f64) - 1.0;
    let ann_vol = sd * TRADING_DAYS.sqrt();
    let sharpe = if sd > 0.0 { Some(mu / sd * TRADING_DAYS.sqrt()) } else { None };

    // Downside deviation (target 0) + Sortino.
    let down_dev = (rets.iter().map(|&r| if r < 0.0 { r * r } else { 0.0 }).sum::<f64>()
        / rets.len() as f64)
        .sqrt();
    let down_dev_ann = down_dev * TRADING_DAYS.sqrt();
    let sortino = if down_dev > 0.0 { Some(mu / down_dev * TRADING_DAYS.sqrt()) } else { None };

    // Drawdown geometry.
    let mut peak = eq[0];
    let mut max_dd = 0.0_f64;
    let mut sum_sq_dd = 0.0_f64;
    let mut cur_uw = 0_i64;
    let mut max_uw = 0_i64;
    for &e in &eq {
        if e > peak {
            peak = e;
            cur_uw = 0;
        } else {
            cur_uw += 1;
            if cur_uw > max_uw {
                max_uw = cur_uw;
            }
        }
        let dd = if peak > 0.0 { e / peak - 1.0 } else { 0.0 };
        if dd < max_dd {
            max_dd = dd;
        }
        sum_sq_dd += (dd * 100.0).powi(2);
    }
    let ulcer = (sum_sq_dd / eq.len() as f64).sqrt();
    let calmar = if max_dd < 0.0 { Some(ann_return / max_dd.abs()) } else { None };

    // Tail risk.
    let var95 = percentile(&rets, 0.05);
    let below95: Vec<f64> = rets.iter().cloned().filter(|&r| r <= var95).collect();
    let cvar95 = if below95.is_empty() { var95 } else { mean(&below95) };
    let var99 = percentile(&rets, 0.01);
    let below99: Vec<f64> = rets.iter().cloned().filter(|&r| r <= var99).collect();
    let cvar99 = if below99.is_empty() { var99 } else { mean(&below99) };
    let p95 = percentile(&rets, 0.95);
    let tail_ratio = if var95 != 0.0 { Some(p95.abs() / var95.abs()) } else { None };

    // Higher moments.
    let m3 = mean(&rets.iter().map(|&r| (r - mu).powi(3)).collect::<Vec<_>>());
    let m4 = mean(&rets.iter().map(|&r| (r - mu).powi(4)).collect::<Vec<_>>());
    let skew = if sd > 0.0 { Some(m3 / sd.powi(3)) } else { None };
    let kurt = if sd > 0.0 { Some(m4 / sd.powi(4)) } else { None };
    let excess_kurtosis = kurt.map(|k| k - 3.0);

    // MinTRL (López de Prado): min obs for PSR(0) = 95%.
    let mut min_trl_years: Option<f64> = None;
    let srp = if sd > 0.0 { Some(mu / sd) } else { None };
    if let (Some(srp), Some(skew), Some(kurt)) = (srp, skew, kurt) {
        if srp > 0.0 {
            let z = 1.645_f64;
            let min_obs =
                1.0 + (1.0 - skew * srp + ((kurt - 1.0) / 4.0) * srp * srp) * (z / srp).powi(2);
            if min_obs.is_finite() && min_obs > 0.0 {
                min_trl_years = Some(min_obs / TRADING_DAYS);
            }
        }
    }
    let track_years = n as f64 / TRADING_DAYS;
    let significant = min_trl_years.map(|m| track_years >= m);

    // Benchmark-conditional.
    let mut beta = None;
    let mut correlation = None;
    let mut up_capture = None;
    let mut down_capture = None;
    let mut crisis_return = None;
    let mut crisis_benchmark_return = None;
    if has_bench && bench.iter().all(|&b| b.is_finite() && b > 0.0) {
        let b_rets = to_returns(&bench);
        if b_rets.len() == n {
            let b_mu = mean(&b_rets);
            let b_sd = std_dev(&b_rets, b_mu);
            let mut cov = 0.0_f64;
            for i in 0..n {
                cov += (rets[i] - mu) * (b_rets[i] - b_mu);
            }
            cov /= n as f64 - 1.0;
            if b_sd > 0.0 {
                beta = Some(cov / (b_sd * b_sd));
            }
            if sd > 0.0 && b_sd > 0.0 {
                correlation = Some(cov / (sd * b_sd));
            }

            let (mut s_up, mut b_up, mut s_dn, mut b_dn) = (1.0, 1.0, 1.0, 1.0);
            let (mut n_up, mut n_dn) = (0, 0);
            for i in 0..n {
                if b_rets[i] > 0.0 {
                    s_up *= 1.0 + rets[i];
                    b_up *= 1.0 + b_rets[i];
                    n_up += 1;
                } else if b_rets[i] < 0.0 {
                    s_dn *= 1.0 + rets[i];
                    b_dn *= 1.0 + b_rets[i];
                    n_dn += 1;
                }
            }
            if n_up > 0 && b_up != 1.0 {
                up_capture = Some((s_up - 1.0) / (b_up - 1.0));
            }
            if n_dn > 0 && b_dn != 1.0 {
                down_capture = Some((s_dn - 1.0) / (b_dn - 1.0));
            }

            // Crisis window = benchmark's deepest peak-to-trough drawdown.
            let (mut b_peak, mut b_peak_idx) = (bench[0], 0usize);
            let (mut worst_dd, mut t_idx, mut p_for_t) = (0.0_f64, 0usize, 0usize);
            for (i, &b) in bench.iter().enumerate() {
                if b > b_peak {
                    b_peak = b;
                    b_peak_idx = i;
                }
                let dd = b / b_peak - 1.0;
                if dd < worst_dd {
                    worst_dd = dd;
                    t_idx = i;
                    p_for_t = b_peak_idx;
                }
            }
            if worst_dd < 0.0 && t_idx > p_for_t {
                crisis_benchmark_return = Some(bench[t_idx] / bench[p_for_t] - 1.0);
                crisis_return = Some(eq[t_idx] / eq[p_for_t] - 1.0);
            }
        }
    }

    Some(Analytics {
        observations: n,
        annualized_return: Some(ann_return),
        annualized_vol: Some(ann_vol),
        sharpe,
        sortino,
        calmar,
        max_drawdown: Some(max_dd),
        ulcer_index: Some(ulcer),
        max_time_under_water_days: Some(max_uw),
        downside_deviation: Some(down_dev_ann),
        var95: Some(var95),
        cvar95: Some(cvar95),
        cvar99: Some(cvar99),
        tail_ratio,
        skew,
        excess_kurtosis,
        min_trl_years,
        track_years: Some(track_years),
        significant,
        beta,
        correlation,
        up_capture,
        down_capture,
        crisis_return,
        crisis_benchmark_return,
    })
}

/// PyO3 entry: returns a dict of raw (unrounded) metrics, or `None` if the
/// series is too short. The Python layer rounds and adds date-derived fields.
#[pyfunction]
#[pyo3(signature = (equity, benchmark=None))]
pub fn certification_analytics(
    equity: Vec<f64>,
    benchmark: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let a = match compute(&equity, benchmark.as_deref()) {
        Some(a) => a,
        None => return Ok(Python::with_gil(|py| py.None())),
    };
    Python::with_gil(|py| {
        let d = PyDict::new_bound(py);
        d.set_item("observations", a.observations)?;
        d.set_item("annualized_return", a.annualized_return)?;
        d.set_item("annualized_vol", a.annualized_vol)?;
        d.set_item("sharpe", a.sharpe)?;
        d.set_item("sortino", a.sortino)?;
        d.set_item("calmar", a.calmar)?;
        d.set_item("max_drawdown", a.max_drawdown)?;
        d.set_item("ulcer_index", a.ulcer_index)?;
        d.set_item("max_time_under_water_days", a.max_time_under_water_days)?;
        d.set_item("downside_deviation", a.downside_deviation)?;
        d.set_item("var95", a.var95)?;
        d.set_item("cvar95", a.cvar95)?;
        d.set_item("cvar99", a.cvar99)?;
        d.set_item("tail_ratio", a.tail_ratio)?;
        d.set_item("skew", a.skew)?;
        d.set_item("excess_kurtosis", a.excess_kurtosis)?;
        d.set_item("min_trl_years", a.min_trl_years)?;
        d.set_item("track_years", a.track_years)?;
        d.set_item("significant", a.significant)?;
        d.set_item("beta", a.beta)?;
        d.set_item("correlation", a.correlation)?;
        d.set_item("up_capture", a.up_capture)?;
        d.set_item("down_capture", a.down_capture)?;
        d.set_item("crisis_return", a.crisis_return)?;
        d.set_item("crisis_benchmark_return", a.crisis_benchmark_return)?;
        Ok(d.unbind().into_any())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn equity_from_returns(rets: &[f64]) -> Vec<f64> {
        let mut eq = vec![100.0];
        for &r in rets {
            eq.push(eq.last().unwrap() * (1.0 + r));
        }
        eq
    }

    #[test]
    fn too_short_returns_none() {
        assert!(compute(&[100.0, 101.0, 102.0], None).is_none());
    }

    #[test]
    fn steady_growth_has_no_drawdown_or_sortino() {
        let eq = equity_from_returns(&vec![0.0004; 300]);
        let a = compute(&eq, None).unwrap();
        assert_eq!(a.max_drawdown, Some(0.0));
        assert!(a.sortino.is_none()); // no downside
        assert!(a.calmar.is_none()); // no drawdown
        assert!(a.annualized_return.unwrap() > 0.0);
    }

    #[test]
    fn oscillating_series_has_full_metrics() {
        let rets: Vec<f64> = (0..400).map(|i| 0.0005 + 0.01 * ((i as f64) / 3.0).sin()).collect();
        let a = compute(&equity_from_returns(&rets), None).unwrap();
        assert!(a.max_drawdown.unwrap() < 0.0);
        assert!(a.max_time_under_water_days.unwrap() > 0);
        assert!(a.cvar95.unwrap() <= a.var95.unwrap());
        assert!(a.cvar99.unwrap() <= a.cvar95.unwrap());
        assert!(a.min_trl_years.unwrap() > 0.0);
        assert!(a.sortino.is_some());
    }

    #[test]
    fn benchmark_identity_gives_unit_beta() {
        let rets: Vec<f64> = (0..400).map(|i| 0.0005 + 0.01 * ((i as f64) / 3.0).sin()).collect();
        let eq = equity_from_returns(&rets);
        let a = compute(&eq, Some(&eq)).unwrap();
        assert!((a.beta.unwrap() - 1.0).abs() < 0.01);
        assert!((a.correlation.unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn defensive_book_survives_crisis() {
        // Benchmark rises, crashes ~30%, recovers; portfolio flat during crash.
        let mut b_rets = vec![0.004; 50];
        b_rets.extend(vec![-0.018; 20]);
        b_rets.extend(vec![0.003; 60]);
        let p_rets: Vec<f64> = b_rets
            .iter()
            .enumerate()
            .map(|(i, _)| if (50..70).contains(&i) { 0.0002 } else { 0.001 })
            .collect();
        let a = compute(&equity_from_returns(&p_rets), Some(&equity_from_returns(&b_rets))).unwrap();
        assert!(a.crisis_benchmark_return.unwrap() < -0.15);
        assert!(a.crisis_return.unwrap() > a.crisis_benchmark_return.unwrap());
        assert!(a.down_capture.unwrap() < 0.5);
    }
}
