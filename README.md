<p align="center">
  <a href="https://ai.cpz-lab.com/">
    <img src="https://drive.google.com/uc?id=1JY-PoPj9GHmpq3bZLC7WyJLbGuT1L3hN" alt="CPZAI" width="180">
  </a>
</p>

<h1 align="center">cpz-quant</h1>

<p align="center"><b>Portfolio optimization, risk analytics, and strategy certification in Python</b></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-informational" alt="Python 3.9-3.12"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/typing-py.typed-informational" alt="Typed"></a>
  <a href="rust/"><img src="https://img.shields.io/badge/rust-accelerated-CE422B?logo=rust&logoColor=white" alt="Rust-accelerated"></a>
</p>

<p align="center">
  <a href="https://ai.cpz-lab.com/">CPZAI operating system</a> ·
  <a href="https://cpz-lab.github.io/cpz-quant/">Documentation</a> ·
  <a href="https://github.com/CPZ-Lab/cpz-quant/issues">Issues</a>
</p>

**cpz-quant** is the open-source quantitative research engine from [CPZ Lab](https://ai.cpz-lab.com): institutional-grade **portfolio optimization**, **covariance estimation**, **risk measures**, **walk-forward and combinatorial purged cross-validation**, **anti-overfitting strategy certification** (Probability of Backtest Overfitting, Deflated Sharpe Ratio), and **vectorised technical indicators**. Pure functions on NumPy arrays and plain dictionaries: data in, results out, no I/O, no hidden state, fully typed.

It is the research core of the CPZAI systematic trading operating system, and it is fully standalone: `pip install cpz-quant` and you have the complete library with an Apache-2.0 license.

```bash
pip install cpz-quant
```

## 60-second quickstart

```python
import polars as pl
from cpz_quant.portfolio import (
    hierarchical_risk_parity, black_litterman, mean_cvar,
    ledoit_wolf, WalkForward, cross_validate,
)

# Daily returns per asset. Every function accepts a Polars DataFrame,
# a pandas DataFrame, or a plain {asset: [returns]} dict — date/string
# columns are treated as labels and excluded automatically.
returns = pl.DataFrame({
    "AAPL": [0.012, -0.004, 0.007],
    "MSFT": [0.008,  0.002, -0.001],
    "TLT":  [-0.002, 0.005, 0.001],
})

# Hierarchical Risk Parity: clustering-based allocation, no matrix inversion
hrp = hierarchical_risk_parity(returns)
print(hrp.weights, hrp.sharpe_ratio)

# Mean-CVaR: optimize tail risk instead of variance
cvar = mean_cvar(returns, confidence=0.95)

# Walk-forward cross-validation of any allocator
cv = cross_validate(
    lambda train: hierarchical_risk_parity(train).weights,
    returns,
    cv=WalkForward(n_splits=4, test_size=63),
)
print(cv.oos_sharpe)
```

Certify a strategy before you trust the backtest:

```python
import numpy as np
from cpz_quant.certification import (
    probability_of_backtest_overfitting,   # CSCV / PBO
    compute_risk_analytics,                # Sortino, CVaR, tail ratio, MinTRL...
)

trials = np.column_stack([...])            # (T, N): returns of N tested configs
pbo = probability_of_backtest_overfitting(trials)
analytics = compute_risk_analytics(equity_curve)
print(pbo.pbo, pbo.performance_degradation, analytics.sortino)
```

## What is in the box

### Portfolio optimization (20+ allocators)

| Family | Methods |
|---|---|
| Classic convex | mean-variance (Markowitz), minimum variance, maximum Sharpe ratio, maximum diversification, minimum tracking error, turnover-penalized |
| Risk-based | risk parity / risk budgeting, equal weight, inverse volatility via mean-risk |
| Clustering | Hierarchical Risk Parity (HRP), Hierarchical Equal Risk Contribution (HERC), Nested Clustered Optimization (NCO), Schur complementary allocation |
| Views and priors | Black-Litterman, entropy pooling (fully flexible views) |
| Tail-risk | mean-CVaR, 17-measure mean-risk optimizer (CVaR, EVaR, CDaR, EDaR, drawdown-at-risk, Ulcer index, Gini mean difference, ...) |
| Robust | robust mean-variance (scipy native), box and ellipsoidal uncertainty sets on expected returns (convex backend) |
| Cardinality | exact mixed-integer cardinality-constrained portfolios with semi-continuous position bounds (convex backend) |
| Alpha-risk-cost | Grinold-Kahn style alpha-risk-cost optimizer with transfer coefficient |
| Quantum / QUBO | QUBO portfolio selection, quantum-inspired HRP, simulated annealing and D-Wave backends |

### Covariance estimation and factor models

Sample, exponentially weighted (EWMA), **Ledoit-Wolf shrinkage**, Oracle Approximating Shrinkage, **Marchenko-Pastur denoising**, detoning, Gerber statistic, statistical (PCA) and fundamental factor models, factor risk decomposition.

### Model selection that respects time

**WalkForward** and **CombinatorialPurgedCV** splitters, `cross_validate`, and `grid_search`, built for overlapping financial samples where naive K-fold leaks. Optional scikit-learn estimator wrappers (`MeanRiskEstimator`, `HRPEstimator`, `HERCEstimator`, `NCOEstimator`) plug into sklearn `Pipeline` and `GridSearchCV` (`pip install cpz-quant[sklearn]`).

### Strategy certification (the referee layer)

Most backtests are overfit. cpz-quant ships the math to prove whether yours is:

- **Probability of Backtest Overfitting (PBO)** via combinatorially symmetric cross-validation (CSCV)
- **Deflated Sharpe Ratio** and Probabilistic Sharpe Ratio gates that account for multiple testing
- Regime-conditional performance breakdowns
- A graded, reproducible certification score (`certify`) used by the CPZ Certification Standard

### Convex optimization backend

`pip install cpz-quant[cvx]` adds exact cvxpy programs: hard gross-exposure and turnover constraints, L2 regularization, CVaR linear programming, robust uncertainty sets, and mixed-integer cardinality constraints (`[cvx-mip]` for the open-source SCIP solver). If a required solver is missing the library raises with install instructions; it never silently substitutes an approximation.

### Technical indicators

Vectorised momentum, trend, volatility, volume, and statistical indicators on NumPy/Polars, with optional Rust acceleration and graceful pure-Python fallback.

### Quantum and quantum-inspired optimization

Portfolio selection as a QUBO problem with pluggable solvers: exact brute force, simulated annealing (`pip install cpz-quant[quantum]`, dwave-neal), and quantum-inspired HRP cluster ordering. `build_portfolio_qubo` exposes the raw QUBO matrix for any annealer. Real quantum hardware (IonQ, Rigetti, IQM via Amazon Braket) runs through the CPZAI operating system with cost gating; the local solvers are fully standalone.

### Rust-accelerated core

The `rust/` crate (`cpz_risk_rs`, PyO3 + rayon) ships in this repo and accelerates the hot paths: certification analytics, Monte Carlo VaR, Ledoit-Wolf and EWMA covariance, Marchenko-Pastur denoising, HRP weights, and the indicator kernels. Build it with `pip install maturin && cd rust && maturin develop --release`. Everything runs identically without it — pure NumPy fallbacks are parity-tested, and `has_rust()` tells you which path is active. No silent behavior differences, only speed.

### Visualization

`pip install cpz-quant[viz]` adds four Plotly figures in `cpz_quant.viz`: `plot_weights`, `plot_frontier` (efficient frontier with max-Sharpe and min-variance marked), `plot_drawdown` (equity + underwater panel), and `plot_corr_clusters` (correlation matrix ordered by HRP clustering). Plotly is never imported unless you use them.

### Transaction costs, capacity, and attribution

Almgren-Chriss market impact, linear and square-root impact, spread costs, turnover analysis, alpha-decay capacity estimation, Brinson-Fachler attribution, factor and risk attribution, alpha-beta decomposition.

## Design principles

1. **Pure functions.** Every public API is data in, results out. No database, no network, no global state. Trivially testable and reproducible.
1. **DataFrame-native, dependency-lean.** Polars and pandas DataFrames work everywhere returns go; neither library is imported unless you pass one, and pandas is never a dependency.
2. **Fail loudly.** No silent fallbacks, no fabricated defaults. Missing solver, degenerate covariance, or invalid input raises with an actionable message.
3. **Typed end to end.** `py.typed`, mypy-checked in CI, pydantic result models where structure matters.
4. **Certification is not optional.** The same anti-overfitting gates that certify strategies on the CPZAI operating system are open source here, so any grade can be independently reproduced.

## FAQ

**How do I do portfolio optimization in Python with cpz-quant?**
`pip install cpz-quant`, then call any allocator in `cpz_quant.portfolio` with your returns as a Polars DataFrame, pandas DataFrame, or dict of series (see quickstart above). All 20+ methods share the same input shape and return an `OptResult` with weights, expected return, volatility, and Sharpe ratio.

**Does cpz-quant work with Polars and pandas?**
Yes, natively: every allocator, covariance estimator, and pre-selection transformer accepts a Polars or pandas DataFrame directly (numeric columns become assets; date/string columns are excluded as labels). Polars is a core dependency; pandas is supported but never required.

**Does cpz-quant support Hierarchical Risk Parity (HRP) and HERC?**
Yes: `hierarchical_risk_parity`, `hierarchical_equal_risk_contribution`, plus NCO and Schur complementary allocation for nested and cluster-aware variants.

**Can I detect backtest overfitting?**
Yes: `cpz_quant.certification.probability_of_backtest_overfitting` implements CSCV/PBO, and `certify` grades a strategy with Deflated Sharpe Ratio gates.

**Is it compatible with scikit-learn?**
Yes, optionally: `pip install cpz-quant[sklearn]` provides estimator wrappers that work inside sklearn pipelines and grid search, while the core library stays dependency-light.

**How does cpz-quant relate to the cpz-ai SDK?**
cpz-quant is the open-source research core (Apache-2.0). The proprietary [cpz-ai SDK](https://pypi.org/project/cpz-ai/) builds on it and adds live multi-broker execution, FIX connectivity, market data access, and the CPZAI operating system integration. Research is open; execution is a product.

**Is AI used in developing cpz-quant?**
Yes, and it is disclosed: parts of the library are developed with Simons, the AI research partner of the CPZAI operating system, under CPZ Lab's review and maintainership. AI-authored commits carry the git identity `Simons <simons@cpz-lab.com>` so provenance is auditable, in line with the transparency expectations of the EU AI Act, the NIST AI Risk Management Framework, and ISO/IEC 42001. See [CONTRIBUTING.md](CONTRIBUTING.md).

**Is this investment advice?**
No. cpz-quant is a software library for quantitative research. Nothing in it constitutes investment advice.

## Documentation

Full documentation: **https://cpz-lab.github.io/cpz-quant/**

- [Quickstart](https://cpz-lab.github.io/cpz-quant/quickstart/)
- [Portfolio optimization guide](https://cpz-lab.github.io/cpz-quant/portfolio/)
- [Certification guide](https://cpz-lab.github.io/cpz-quant/certification/)
- [Cross-validation guide](https://cpz-lab.github.io/cpz-quant/model-selection/)
- [Examples](examples/)

## Contributing

Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md). The library is tested on Python 3.9 to 3.12 with lint, type-check, and branch-coverage gates enforced in CI.

## Citation

If you use cpz-quant in academic work, please cite it (see [CITATION.cff](CITATION.cff)):

```
CPZ Lab (2026). cpz-quant: quantitative portfolio optimization, risk analytics,
and strategy certification in Python. https://github.com/CPZ-Lab/cpz-quant
```

## License

Apache License 2.0. Copyright (c) 2024-2026 CPZ Capital Ltd.
