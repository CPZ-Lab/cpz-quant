//! Rust-accelerated technical indicators.
//!
//! Each function takes raw f64 vectors and returns f64 vectors.
//! PyO3 handles type conversion at the boundary.

use pyo3::prelude::*;

/// Simple Moving Average via cumulative-sum trick.
#[pyfunction]
pub fn rolling_sma(close: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    let n = close.len();
    if period == 0 || n < period {
        return Ok(vec![f64::NAN; n]);
    }
    let mut out = vec![f64::NAN; n];
    let mut sum: f64 = 0.0;
    for i in 0..period {
        sum += close[i];
    }
    out[period - 1] = sum / period as f64;
    for i in period..n {
        sum += close[i] - close[i - period];
        out[i] = sum / period as f64;
    }
    Ok(out)
}

/// Exponential Moving Average.
#[pyfunction]
pub fn ema(close: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    let n = close.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    let one_minus = 1.0 - alpha;
    let mut out = vec![0.0_f64; n];
    out[0] = close[0];
    for i in 1..n {
        out[i] = alpha * close[i] + one_minus * out[i - 1];
    }
    Ok(out)
}

/// Weighted Moving Average.
#[pyfunction]
pub fn wma(close: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    let n = close.len();
    if period == 0 || n < period {
        return Ok(vec![f64::NAN; n]);
    }
    let denom: f64 = (period * (period + 1)) as f64 / 2.0;
    let mut out = vec![f64::NAN; n];
    for i in (period - 1)..n {
        let mut sum = 0.0_f64;
        for j in 0..period {
            sum += close[i - period + 1 + j] * (j + 1) as f64;
        }
        out[i] = sum / denom;
    }
    Ok(out)
}

/// Kaufman Adaptive Moving Average.
#[pyfunction]
pub fn kama(close: Vec<f64>, period: usize, fast_sc: f64, slow_sc: f64) -> PyResult<Vec<f64>> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if n <= period {
        return Ok(out);
    }
    out[period - 1] = close[period - 1];
    for i in period..n {
        let direction = (close[i] - close[i - period]).abs();
        let mut volatility = 0.0_f64;
        for j in (i - period)..i {
            volatility += (close[j + 1] - close[j]).abs();
        }
        let er = if volatility > 1e-15 { direction / volatility } else { 0.0 };
        let sc = (er * (fast_sc - slow_sc) + slow_sc).powi(2);
        out[i] = out[i - 1] + sc * (close[i] - out[i - 1]);
    }
    Ok(out)
}

/// RSI via Wilder's smoothing.
#[pyfunction]
pub fn rsi(close: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if n < period + 1 {
        return Ok(out);
    }

    let mut gains = vec![0.0_f64; n - 1];
    let mut losses = vec![0.0_f64; n - 1];
    for i in 0..(n - 1) {
        let delta = close[i + 1] - close[i];
        if delta > 0.0 {
            gains[i] = delta;
        } else {
            losses[i] = -delta;
        }
    }

    let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

    let rs = if avg_loss > 1e-15 { avg_gain / avg_loss } else { f64::INFINITY };
    out[period] = 100.0 - 100.0 / (1.0 + rs);

    let alpha = 1.0 / period as f64;
    for i in period..(n - 1) {
        avg_gain = alpha * gains[i] + (1.0 - alpha) * avg_gain;
        avg_loss = alpha * losses[i] + (1.0 - alpha) * avg_loss;
        let rs = if avg_loss > 1e-15 { avg_gain / avg_loss } else { f64::INFINITY };
        out[i + 1] = 100.0 - 100.0 / (1.0 + rs);
    }
    Ok(out)
}

/// MACD: returns (macd_line, signal_line, histogram).
#[pyfunction]
pub fn macd(
    close: Vec<f64>,
    fast: usize,
    slow: usize,
    signal: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let fast_ema = ema(close.clone(), fast)?;
    let slow_ema = ema(close, slow)?;
    let n = fast_ema.len();
    let mut macd_line = vec![0.0_f64; n];
    for i in 0..n {
        macd_line[i] = fast_ema[i] - slow_ema[i];
    }
    let signal_line = ema(macd_line.clone(), signal)?;
    let mut histogram = vec![0.0_f64; n];
    for i in 0..n {
        histogram[i] = macd_line[i] - signal_line[i];
    }
    Ok((macd_line, signal_line, histogram))
}

/// Average Directional Index.
#[pyfunction]
pub fn adx(high: Vec<f64>, low: Vec<f64>, close: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if n < period + 1 {
        return Ok(out);
    }

    let mut tr = vec![0.0_f64; n];
    let mut plus_dm = vec![0.0_f64; n];
    let mut minus_dm = vec![0.0_f64; n];

    for i in 1..n {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);

        let h_diff = high[i] - high[i - 1];
        let l_diff = low[i - 1] - low[i];
        plus_dm[i] = if h_diff > l_diff && h_diff > 0.0 { h_diff } else { 0.0 };
        minus_dm[i] = if l_diff > h_diff && l_diff > 0.0 { l_diff } else { 0.0 };
    }

    let mut s_tr: f64 = tr[1..=period].iter().sum::<f64>() / period as f64;
    let mut s_plus: f64 = plus_dm[1..=period].iter().sum::<f64>() / period as f64;
    let mut s_minus: f64 = minus_dm[1..=period].iter().sum::<f64>() / period as f64;

    let mut dx_values: Vec<f64> = Vec::with_capacity(n);

    for i in period..n {
        if i > period {
            s_tr = (s_tr * (period as f64 - 1.0) + tr[i]) / period as f64;
            s_plus = (s_plus * (period as f64 - 1.0) + plus_dm[i]) / period as f64;
            s_minus = (s_minus * (period as f64 - 1.0) + minus_dm[i]) / period as f64;
        }

        let plus_di = if s_tr > 1e-15 { s_plus / s_tr * 100.0 } else { 0.0 };
        let minus_di = if s_tr > 1e-15 { s_minus / s_tr * 100.0 } else { 0.0 };
        let di_sum = plus_di + minus_di;
        let dx = if di_sum > 1e-15 { (plus_di - minus_di).abs() / di_sum * 100.0 } else { 0.0 };
        dx_values.push(dx);

        if dx_values.len() == period {
            out[i] = dx_values.iter().sum::<f64>() / period as f64;
        } else if dx_values.len() > period {
            let prev = out[i - 1];
            if !prev.is_nan() {
                out[i] = (prev * (period as f64 - 1.0) + dx) / period as f64;
            }
        }
    }
    Ok(out)
}

/// Average True Range via Wilder's smoothing.
#[pyfunction]
pub fn atr(high: Vec<f64>, low: Vec<f64>, close: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if n < period + 1 {
        return Ok(out);
    }

    let mut tr = vec![0.0_f64; n];
    tr[0] = high[0] - low[0];
    for i in 1..n {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    let mut avg: f64 = tr[..period].iter().sum::<f64>() / period as f64;
    out[period - 1] = avg;
    let alpha = 1.0 / period as f64;
    for i in period..n {
        avg = alpha * tr[i] + (1.0 - alpha) * avg;
        out[i] = avg;
    }
    Ok(out)
}

/// Bollinger Bands: returns (upper, middle, lower).
#[pyfunction]
pub fn bollinger(
    close: Vec<f64>,
    period: usize,
    num_std: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let n = close.len();
    let mid = rolling_sma(close.clone(), period)?;
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];

    for i in (period - 1)..n {
        let window = &close[(i + 1 - period)..=i];
        let mean = mid[i];
        if mean.is_nan() {
            continue;
        }
        let var: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std = var.sqrt();
        upper[i] = mean + num_std * std;
        lower[i] = mean - num_std * std;
    }
    Ok((upper, mid, lower))
}

/// Garman-Klass volatility estimator.
#[pyfunction]
pub fn garman_klass_vol(
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    period: usize,
) -> PyResult<Vec<f64>> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if n < period {
        return Ok(out);
    }

    let ln2_const = 2.0_f64 * 2.0_f64.ln() - 1.0;
    let mut gk = vec![0.0_f64; n];
    for i in 0..n {
        let hl = (high[i] / low[i].max(1e-15)).max(1e-15).ln();
        let co = (close[i] / open[i].max(1e-15)).max(1e-15).ln();
        gk[i] = 0.5 * hl * hl - ln2_const * co * co;
    }

    for i in (period - 1)..n {
        let mean: f64 = gk[(i + 1 - period)..=i].iter().sum::<f64>() / period as f64;
        out[i] = mean.max(0.0).sqrt() * (252.0_f64).sqrt();
    }
    Ok(out)
}
