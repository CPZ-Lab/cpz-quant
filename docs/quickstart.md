# Quickstart

## Install

```bash
pip install cpz-quant              # core: numpy, scipy, polars, pydantic
pip install "cpz-quant[sklearn]"   # + scikit-learn estimator wrappers
pip install "cpz-quant[cvx]"       # + cvxpy convex backend
pip install "cpz-quant[cvx-mip]"   # + SCIP for cardinality constraints
pip install "cpz-quant[all]"       # everything
```

Python 3.9+ on any OS.

## Your first optimization

Every allocator takes the same input: a dict of per-asset return series.

```python
from cpz_quant.portfolio import hierarchical_risk_parity, max_sharpe, risk_parity

returns = {
    "AAPL": [...],   # daily returns, e.g. from your data vendor
    "MSFT": [...],
    "TLT":  [...],
    "GLD":  [...],
}

hrp = hierarchical_risk_parity(returns)
print(hrp.weights)          # {"AAPL": 0.18, "MSFT": 0.17, "TLT": 0.40, "GLD": 0.25}
print(hrp.sharpe_ratio)     # annualised, net of nothing: pure math
```

Results are `OptResult` dataclasses with `weights`, `expected_return`, `volatility`, `sharpe_ratio`, and method-specific `info`.

## Add constraints

```python
from cpz_quant.portfolio import Constraints, mean_variance

res = mean_variance(
    returns,
    constraints=Constraints(long_only=True, max_weight=0.35),
    target_return=0.08,          # annualised
)
```

## Validate out of sample

```python
from cpz_quant.portfolio import WalkForward, cross_validate

cv = cross_validate(
    lambda train: hierarchical_risk_parity(train).weights,
    returns,
    cv=WalkForward(n_splits=4, test_size=63),
)
print(cv.oos_sharpe, cv.stability())
```

## Certify before you deploy

```python
import numpy as np
from cpz_quant.certification import probability_of_backtest_overfitting

# returns of every configuration you tried, shape (T, N)
trials = np.column_stack([config_1, config_2, config_3, ...])
pbo = probability_of_backtest_overfitting(trials)
if pbo.pbo > 0.5:
    print("More likely than not overfit. Do not deploy.")
```

## Next steps

- [Portfolio optimization guide](portfolio.md)
- [Convex backend](convex.md) for cardinality and robust constraints
- [Certification](certification.md) for the full grading pipeline
