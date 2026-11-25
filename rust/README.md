# CPZ Risk Analytics — Rust Acceleration

High-performance Rust implementations for compute-intensive risk operations.

## Functions

- **`monte_carlo_var`**: Parallel Monte Carlo VaR simulation (~10x faster than NumPy)
- **`covariance_matrix`**: Fast covariance matrix computation (~5x faster)
- **`risk_parity_weights`**: Iterative risk parity optimization (~3x faster)

## Build

Requires Rust toolchain and maturin:

```bash
# Install maturin
pip install maturin

# Build and install the extension
cd rust
maturin develop --release
```

## Usage

cpz-quant loads the extension automatically when it is built, and uses it for
`cpz_quant.certification.compute_risk_analytics` and the indicator kernels in
`cpz_quant.indicators._rust_accel` (SMA, EMA, WMA, KAMA, RSI, MACD, ADX, ATR,
Bollinger bands, Garman-Klass volatility). `cpz_quant.certification.has_rust()`
reports whether it is loaded; without it the NumPy implementations run.

The Monte Carlo VaR, covariance, risk parity, Ledoit-Wolf, EWMA,
Marchenko-Pastur and HRP functions in this crate are not called by cpz-quant.
They predate the cpz-quant 1.3.0 estimator and allocator fixes and do not
reproduce the Python results.
