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

The Python wrapper automatically uses Rust when available:

```python
from cpz.risk._rust_accel import has_rust, fast_monte_carlo_var

if has_rust():
    result = fast_monte_carlo_var(returns, num_simulations=100000)
```

Falls back to NumPy/SciPy transparently if Rust is not built.
