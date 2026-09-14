# Model selection

Financial samples can overlap and depend on one another. Random K-fold splits
can therefore leak information. `cpz_quant.portfolio.model_selection` provides
time-ordered and fixed-gap splitters, but choosing a splitter alone does not
make a backtest leakage-free. Fit preprocessing only on training observations
and choose gaps that cover the actual feature and label horizons.

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

`CombinatorialPurgedCV(n_splits, n_test_splits, purge, embargo)` divides the
observations into contiguous groups and evaluates every combination of held-out
groups. With `N` groups and `k` test groups, there are `C(N, k)` train/test folds
but only `C(N-1, k-1)` complete reconstructed paths. For example, six groups and
two test groups produce **15 fits and 5 paths**, not 15 independent backtests.
This distinction follows [López de Prado's CPCV construction, section 9](https://smallake.kr/wp-content/uploads/2018/07/SSRN-id3104816.pdf).

```python
from cpz_quant.portfolio import CombinatorialPurgedCV, cross_validate

splitter = CombinatorialPurgedCV(6, 2, purge=5, embargo=10)
print(splitter.n_folds(), splitter.n_paths())  # 15, 5
result = cross_validate(my_allocator, returns, splitter)
print(result.per_path_sharpe)                 # five complete-path scores
print(result.oos_sharpe)                      # their arithmetic mean
```

Each path contains every observation exactly once. The first held-out occurrence
of each group is assigned to path 0, the second to path 1, and so on. Every
fold/group output is used once across the paths. Paths still share the original
observations and are not independent histories. To reconstruct your own outputs,
call `splitter.reconstruct_paths(fold_returns, n_samples)`: supply one vector per
fold in `split()` order, aligned to that fold's sorted test indices. It returns
an array shaped `(n_paths, n_samples)`.

`purge` removes a fixed number of observations before and after each test group.
`embargo` removes observations immediately after each test group. These windows
overlap, so the right-hand gap is `max(purge, embargo)`, not their sum. This is
positional purging, not purging from variable event-label start/end times.
The gaps do not guarantee the absence of serial dependence or look-ahead bias.
CPCV may train on groups chronologically after a held-out group; use walk-forward
for strictly past-to-future training. The certification PBO test uses a separate
CSCV procedure; it is not this splitter.

### Result semantics and errors

For CPCV, `oos_sharpe`, `oos_return_ann` and `oos_vol_ann` are arithmetic means
of complete-path metrics. Their corresponding `per_path_*` lists expose the
distribution. `n_paths` counts complete paths, while `n_splits`,
`per_split_sharpe` and `stability()` continue to describe fits/folds.
Walk-forward forms one time-ordered path from held-out observations (omitting
training-only periods). Returns are annualized with 252 observations per year;
annual return is arithmetic mean times 252, not CAGR. The existing convention
of a zero Sharpe score for zero volatility is retained.

This corrects the earlier CPCV behavior that concatenated repeated observations
from overlapping folds as if they formed one history. As a result, CPCV summary
Sharpe/volatility values and `n_paths()` change. Other custom splitters must have
disjoint test indices. Invalid settings, fewer than two train/test observations
per fold, empty cross-validation, non-finite input/output, and incomplete path
outputs raise instead of silently skipping folds or returning empty scores.

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
