# Portfolio optimization

All allocators live in `cpz_quant.portfolio`, take `{asset_id: [returns]}`, and return an `OptResult` (or a method-specific result dataclass). Annualisation assumes 252 trading days.

## Allocators

| Function | Method |
|---|---|
| `mean_variance` | Minimum variance, optionally at an exact target return |
| `min_variance` | Global minimum variance |
| `max_sharpe` | Maximum Sharpe ratio |
| `risk_parity` | Long-only equal risk contribution (risk budgeting), solved by coordinate descent |
| `equal_weight` | 1/N benchmark |
| `hierarchical_risk_parity` | HRP (Lopez de Prado 2016): clustering + inverse-variance recursive bisection, no matrix inversion |
| `hierarchical_equal_risk_contribution` | HERC (Raffinot 2018): splits follow the dendrogram down to K clusters |
| `nested_clustered_optimization` | NCO: intra-cluster then inter-cluster optimization (max-Sharpe raises when no tangency portfolio exists) |
| `schur_complementary_allocation` | Schur-complement-based allocation |
| `black_litterman` | Equilibrium prior + investor views, posterior optimal weights |
| `mean_cvar` | Minimise Conditional VaR on empirical scenarios (Rockafellar-Uryasev LP, HiGHS) |
| `robust_mvo` | Robust mean-variance (scipy native) |
| `max_diversification` | Maximise the diversification ratio |
| `min_tracking_error` | Track a benchmark with minimum tracking error |
| `turnover_penalized` | Mean-variance with turnover penalty against current weights |
| `alpha_risk_cost_optimize` | Grinold-Kahn alpha-risk-cost; costs priced per unit traded, Clarke-de Silva-Thorley transfer coefficient |
| `mean_risk_optimize` | Generic mean-risk optimizer over any of the 17 risk measures (`max_return` needs `max_risk`) |
| `qubo_portfolio_selection` | Binary asset selection as a QUBO problem |
| `quantum_inspired_hrp` | HRP with a seriation order (spectral heuristic, exact, or QUBO) |

The [convex backend](convex.md) adds `mean_variance_cvx`, `mean_cvar_cvx`, `robust_mean_variance_cvx`, and `cardinality_constrained_cvx`.

### Core optimizer constraints

`mean_variance`, `min_variance` and `max_sharpe` require full investment
(`sum(weights) == 1`). They enforce `min_weight`, `max_weight`, `long_only`
and `max_gross_exposure` (2.0 by default; `None` disables the gross cap).
A positive `min_weight` still applies when `long_only=True`.
Because net exposure is fixed at one, `max_net_exposure` below one is infeasible
and raises. Sector, factor, turnover and tracking-error limits are not supported
by these three functions and raise `NotImplementedError` when supplied.
`mean_cvar`, `robust_mvo`, `max_diversification`, `min_tracking_error`,
`turnover_penalized` and `alpha_risk_cost_optimize` enforce the same limits
the same way; the last two also enforce `max_turnover` against the current
weights. When every lower bound is non-negative, full investment already fixes
gross exposure at one, so a gross cap of one or more does not change the
answer. The cvxpy optimizers also raise on sector, factor and tracking-error
limits.

These core optimizers raise on failed solvers or invalid returned weights.
They retain solver precision in `weights` to avoid rounding away tight limits;
constraint validation uses an absolute tolerance of `1e-7`.
`mean_variance` minimizes variance even when no target is supplied; use
`max_sharpe` to maximize the Sharpe ratio. Its optional `target_return` is an
exact annualized decimal return, not a percentage or a lower bound.
The shared mean/covariance preparation rejects non-finite returns and series
with fewer than two observations instead of replacing them with zeros.

## Risk measures

`RiskMeasure` covers 17 measures usable with `compute_risk`, `all_risk_measures`, and `mean_risk_optimize`: variance, standard deviation, semi-variance, mean absolute deviation, first lower partial moment, Gini mean difference, VaR, CVaR, EVaR, worst realization, entropic risk, maximum drawdown, average drawdown, drawdown at risk, CDaR, EDaR, and Ulcer index.

## Covariance estimation

| Function | Estimator |
|---|---|
| `sample_cov` | Sample covariance |
| `ewma_cov` | Exponentially weighted, normalised weights (pandas `ewm(...).cov(bias=True)`) |
| `ledoit_wolf` | Ledoit-Wolf shrinkage with the published intensity per target (constant correlation, single factor, identity) |
| `oracle_approximating` | Oracle Approximating Shrinkage (Chen et al. 2010, as in scikit-learn) |
| `factor_model_cov` | Factor-model covariance |
| `denoise_mp` | Marchenko-Pastur denoising, constant residual eigenvalue method (Lopez de Prado 2020) |
| `detone_cov` | Remove the market mode (detoning) |
| `gerber_cov` | Gerber statistic (Gerber et al. 2021), positive semidefinite |

The estimators, copulas and pre-selection filters raise on a missing or
non-finite return instead of estimating from it; drop or fill blanks first
(for example `frame.dropna()`), or use `select_complete_assets`.

## Views and priors

- `black_litterman(returns, views, ...)` blends an equilibrium prior with investor views.
- `entropy_pooling(prior_probs, view_rows, ...)` implements fully flexible views over scenario probabilities; `posterior_moments` extracts the tilted mean and covariance.

## Scenario generation

`fit_copula` / `synthetic_returns` fit Gaussian, Student-t, Clayton, or Gumbel copulas and generate synthetic joint return scenarios for stress testing tail dependence.

## Costs, capacity, and attribution

- `almgren_chriss`, `linear_impact`, `sqrt_impact`, `spread_cost`, `total_cost`: execution cost models.
- `turnover_analysis`, `alpha_decay_capacity`: strategy capacity under participation limits and alpha decay.
- `brinson_fachler`, `factor_attribution`, `risk_attribution`, `alpha_beta_decomposition`, `rolling_attribution`: performance and risk attribution.

## Pre-selection

`drop_zero_variance`, `select_complete_assets`, `drop_highly_correlated`, `select_k_extremes`, `select_non_dominated` shrink the universe before optimization.

## scikit-learn wrappers

With `pip install "cpz-quant[sklearn]"`, `cpz_quant.portfolio.sklearn_estimators` provides `MeanRiskEstimator`, `HRPEstimator`, `HERCEstimator`, and `NCOEstimator`, compatible with sklearn `Pipeline` and `GridSearchCV`.
