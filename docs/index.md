# cpz-quant

**cpz-quant** is the open-source quantitative research engine from [CPZ Lab](https://ai.cpz-lab.com): portfolio optimization, covariance estimation, risk measures, time-aware cross-validation, anti-overfitting strategy certification, and vectorised technical indicators for Python.

```bash
pip install cpz-quant
```

Everything is a pure function: data in, results out. No I/O, no hidden state, fully typed, Apache-2.0.

## Why cpz-quant

- **20+ portfolio allocators** sharing one input shape: mean-variance, HRP, HERC, NCO, Schur, Black-Litterman, entropy pooling, mean-CVaR, robust MVO, risk parity, alpha-risk-cost, QUBO, and more.
- **A certification layer.** Probability of Backtest Overfitting (CSCV), Deflated Sharpe Ratio gates, and regime-conditional analytics turn "trust me, it backtests well" into a reproducible grade.
- **Exact convex programs** when you need hard constraints: cardinality, turnover, gross exposure, and robust uncertainty sets via the optional cvxpy backend.
- **Battle-tested.** This is the research core that powers the CPZAI systematic trading operating system in production.

## Where to start

- [Quickstart](quickstart.md): install and run your first optimization in 60 seconds.
- [Portfolio optimization](portfolio.md): every allocator, covariance estimator, and cost model.
- [Convex backend](convex.md): cardinality constraints and robust uncertainty sets.
- [Model selection](model-selection.md): walk-forward and combinatorial purged cross-validation.
- [Certification](certification.md): prove your backtest is not overfit.
- [Indicators](indicators.md): momentum, trend, volatility, volume, statistical.

## Relationship to the CPZAI operating system

cpz-quant is the open research core. The proprietary [cpz-ai SDK](https://pypi.org/project/cpz-ai/) builds on it with live multi-broker execution, FIX connectivity, and market data access on the CPZAI operating system. Research is open; execution is a product.
