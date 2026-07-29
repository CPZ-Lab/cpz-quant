# Certification

Most backtests are overfit. `cpz_quant.certification` is the referee layer: the same open math that powers the CPZ Certification Standard on the CPZAI operating system, so any grade can be independently reproduced.

## Probability of Backtest Overfitting (PBO)

Implements the combinatorially symmetric cross-validation (CSCV) method. Feed it the per-period returns of **every configuration you tried**, shape `(T, N)`:

```python
import numpy as np
from cpz_quant.certification import probability_of_backtest_overfitting

trials = np.column_stack([cfg1_returns, cfg2_returns, ...])  # all N trials
res = probability_of_backtest_overfitting(trials, n_splits=16)

res.pbo                       # P(best in-sample config is below-median OOS)
res.mean_logit                # < 0 leans overfit
res.performance_degradation   # slope of OOS vs IS performance
```

A PBO above ~0.5 means the selection process is more likely than not overfit.

## Risk analytics

`compute_risk_analytics(equity, benchmark=None, dates=None)` produces the institutional due-diligence block from a daily equity curve: Sortino, Calmar, Ulcer index, tail ratio, CVaR, minimum track record length (MinTRL), up/down capture, and crisis behavior. Rust-accelerated when the optional extension is present, with an identical pure-NumPy fallback.

## The certification grade

`certify` grades a strategy across six dimensions (robustness, out-of-sample, risk, factor independence, capacity/cost, live track record) with hard minimum gates, including the **Deflated Sharpe Ratio**, which discounts the Sharpe ratio for the number of trials that produced it:

```python
from cpz_quant.certification import certify, CertMetrics, CertRigor

result = certify(
    backtest_verified=True,
    metrics=CertMetrics(sharpe_ratio=1.4, max_drawdown=-0.18, total_trades=420),
    rigor=CertRigor(deflated_sharpe=0.9, num_trials=25, oos_sharpe=1.1),
)
print(result.grade, result.composite, [g for g in result.gates if not g.passed])
```

Strategies without a qualifying live record receive a **Provisional** grade; the live dimension activates once enough live trades and days accumulate.

## Regime-conditional performance

`regime_conditional_performance` breaks performance down by market regime, exposing strategies that only work in one environment.

## Factor attribution

`certification.factor_attribution` neutralises common factors and reports how much alpha is genuinely idiosyncratic.
