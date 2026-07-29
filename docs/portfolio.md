# Portfolio optimization

All allocators live in `cpz_quant.portfolio`, take `{asset_id: [returns]}`, and return an `OptResult` (or a method-specific result dataclass). Annualisation assumes 252 trading days.

## Allocators

| Function | Method |
|---|---|
| `mean_variance` | Markowitz mean-variance; max-Sharpe or min-variance at a target return |
| `min_variance` | Global minimum variance |
| `max_sharpe` | Maximum Sharpe ratio |
| `risk_parity` | Equal risk contribution (risk budgeting) |
| `equal_weight` | 1/N benchmark |
| `hierarchical_risk_parity` | HRP: correlation clustering + recursive bisection, no matrix inversion |
| `hierarchical_equal_risk_contribution` | HERC: cluster tree with equal risk contribution across clusters |
| `nested_clustered_optimization` | NCO: intra-cluster then inter-cluster optimization |
| `schur_complementary_allocation` | Schur-complement-based allocation |
| `black_litterman` | Equilibrium prior + investor views, posterior optimal weights |
| `mean_cvar` | Minimise Conditional VaR on empirical scenarios (LP) |
| `robust_mvo` | Robust mean-variance (scipy native) |
| `max_diversification` | Maximise the diversification ratio |
| `min_tracking_error` | Track a benchmark with minimum tracking error |
| `turnover_penalized` | Mean-variance with turnover penalty against current weights |
| `alpha_risk_cost_optimize` | Grinold-Kahn alpha-risk-cost with transfer coefficient |
| `mean_risk_optimize` | Generic mean-risk optimizer over any of the 17 risk measures |
| `qubo_portfolio_selection` | Binary asset selection as a QUBO problem |
| `quantum_inspired_hrp` | HRP with QUBO-based cluster ordering |

The [convex backend](convex.md) adds `mean_variance_cvx`, `mean_cvar_cvx`, `robust_mean_variance_cvx`, and `cardinality_constrained_cvx`.

## Risk measures

`RiskMeasure` covers 17 measures usable with `compute_risk`, `all_risk_measures`, and `mean_risk_optimize`: variance, standard deviation, semi-variance, mean absolute deviation, first lower partial moment, Gini mean difference, VaR, CVaR, EVaR, worst realization, entropic risk, maximum drawdown, average drawdown, drawdown at risk, CDaR, EDaR, and Ulcer index.

## Covariance estimation

| Function | Estimator |
|---|---|
| `sample_cov` | Sample covariance |
| `ewma_cov` | Exponentially weighted |
| `ledoit_wolf` | Ledoit-Wolf shrinkage to constant correlation |
| `oracle_approximating` | Oracle Approximating Shrinkage |
| `factor_model_cov` | Factor-model covariance |
| `denoise_mp` | Marchenko-Pastur random-matrix denoising |
| `detone_cov` | Remove the market mode (detoning) |
| `gerber_cov` | Gerber statistic (threshold co-movement) |

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
