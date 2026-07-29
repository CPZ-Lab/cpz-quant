# Model selection

Financial samples overlap and trend; naive K-fold cross-validation leaks future information into training folds. `cpz_quant.portfolio.model_selection` ships splitters built for time series.

## Walk-forward

```python
from cpz_quant.portfolio import WalkForward, cross_validate, hierarchical_risk_parity

cv = cross_validate(
    lambda train: hierarchical_risk_parity(train).weights,
    returns,
    cv=WalkForward(n_splits=5, test_size=63, expanding=True),
)
print(cv.oos_sharpe)          # out-of-sample annualised Sharpe
print(cv.per_split_sharpe)    # fold-by-fold
print(cv.stability())         # fraction of folds with positive OOS Sharpe
```

`expanding=True` grows the training window each fold; `expanding=False` rolls it.

## Combinatorial purged cross-validation

`CombinatorialPurgedCV(n_splits, n_test_splits, purge)` evaluates every combination of test blocks and purges observations adjacent to the test boundary, so serial correlation cannot leak between train and test. This is the same machinery behind the [PBO certification test](certification.md).

## Grid search

```python
from cpz_quant.portfolio import grid_search, WalkForward

result = grid_search(
    lambda **p: (lambda train: my_allocator(train, **p)),
    {"lookback": [60, 120, 252], "risk_aversion": [0.5, 1.0, 2.0]},
    returns,
    WalkForward(n_splits=4),
)
print(result.best_params, result.best_score)
```

!!! warning "Every grid cell is a trial"
    The more configurations you evaluate, the higher the chance the best one is a fluke. Feed the per-trial returns into `probability_of_backtest_overfitting` and check the Deflated Sharpe Ratio before believing any grid search winner.

## scikit-learn interoperability

With `pip install "cpz-quant[sklearn]"`, the estimator wrappers work with sklearn's own model-selection tooling (`Pipeline`, `GridSearchCV`) when you prefer that ecosystem.
