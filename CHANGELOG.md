# Changelog

All notable changes to cpz-quant are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

- `hierarchical_risk_parity` splits each cluster by inverse variance, as in
  Lopez de Prado (2016) and PyPortfolioOpt (it used inverse volatility), and
  returns 100% for a single asset instead of crashing. `schur_complementary_allocation`
  and `quantum_inspired_hrp` use the same split.
- `hierarchical_equal_risk_contribution` follows the dendrogram down to K
  clusters (Raffinot 2018; matches riskfolio-lib) instead of halving the
  ordered cluster list; an out-of-range `n_clusters` raises.
- `quantum_inspired_hrp` reads solved positions as positions (the exact and
  QUBO modes used them backwards), and the default heuristic is spectral
  seriation with swap refinement, which reorders dense correlation matrices
  (reverse Cuthill-McKee left them unsorted). Brute force is vectorised.
- `risk_parity` solves equal risk contribution by cyclical coordinate descent;
  negatively correlated pairs (stocks and Treasuries) no longer return NaN.
  A partial or non-positive `budget` raises. `_metrics` rejects NaN weights.
- `nested_clustered_optimization(objective="max_sharpe")` raises when a
  cluster or the cluster-level problem has no tangency portfolio
  (`1'Sigma^-1 mu <= 0`) instead of returning the lowest-Sharpe mix.
- `mean_cvar` is an exact Rockafellar-Uryasev LP (HiGHS). `mean_cvar`,
  `robust_mvo`, `max_diversification`, `min_tracking_error`,
  `turnover_penalized` and `alpha_risk_cost_optimize` enforce the gross cap,
  reject unsupported limits and raise on failed or infeasible solves. Gross
  caps enter through smooth epigraph variables, and a cap made redundant by a
  long-only book is not passed to SLSQP, so `min_variance` and `max_sharpe`
  reach the optimum at `max_gross_exposure=1.0` in any column order.
- `alpha_risk_cost_optimize` prices each trade with the cost model per unit
  traded (new `portfolio_value`, `adv`, `daily_volatility` arguments for
  size-dependent impact, which now raises when they are missing),
  `cost_model=None` is zero cost, and `transfer_coefficient` is the Clarke,
  de Silva & Thorley correlation in [-1, 1].
- `mean_variance_cvx`, `mean_cvar_cvx`, `robust_mean_variance_cvx` and
  `cardinality_constrained_cvx` raise on sector, factor and tracking-error
  limits instead of ignoring them. `robust_mean_variance_cvx` sizes its
  uncertainty set with the standard error of the annualised mean
  (`252 * cov / T`, it was `cov / T`).
- `mean_risk_optimize` accepts Polars, pandas and dict inputs with their
  names, requires `max_risk` for `max_return` (enforced for every objective),
  and raises instead of returning equal weights when no restart converges.
- `entropy_pooling` preconditions the views, reports `converged=True` only
  when every view is met, and adds `max_view_error`.
- Covariance: `oracle_approximating` is the Chen et al. (2010) OAS
  (equals scikit-learn); `ledoit_wolf` uses each paper's published intensity
  (`covCor.m`, `covMarket.m`, sklearn for identity), is unit-invariant, and its
  `single_factor` target is the single-index model; `ewma_cov` normalises its
  weights (pandas `ewm(...).cov(bias=True)`), uses an exact half-life and
  enforces `min_periods`; `denoise_mp` is Lopez de Prado's constant residual
  eigenvalue method and keeps each variance; `gerber_cov` is the PSD Gerber
  et al. (2021) statistic (matches riskfolio-lib).
- Covariance estimators, copulas and pre-selection filters raise on missing or
  non-finite returns; `select_complete_assets` counts blanks as missing.

## [1.2.0] - 2026-09-17

- Expand `cpz_quant.indicators` from 32 to 91 indicator functions and add all
  61 TA-Lib candlestick patterns. New trend: TRIMA, T3, ZLEMA, HMA, ALMA,
  VIDYA, McGinley Dynamic, Parabolic SAR, Ichimoku, Aroon, Aroon Oscillator,
  Vortex, TRIX, Mass Index, Midpoint, Midprice, Schaff Trend Cycle. New
  momentum: slow Stochastic, Stochastic RSI, CMO, MOM, ROCP, ROCR, Ultimate
  Oscillator, Awesome Oscillator, APO, PPO, KST, TSI, Connors RSI, Fisher
  Transform, Coppock, DPO, Relative Vigor Index, Elder Ray, WaveTrend, Balance
  of Power, DMI (+DI, -DI, DX, ADX, ADXR). New volatility: Parkinson,
  Rogers-Satchell, Yang-Zhang, Chaikin Volatility, Ulcer Index, Choppiness
  Index. New volume: VWMA, A/D Line, Chaikin A/D Oscillator, Force Index, Ease
  of Movement, Klinger, NVI, PVI, Price Volume Trend, Percentage Volume
  Oscillator. New statistics: regression end value and angle, Kaufman
  efficiency ratio, rolling Sharpe and Sortino.
- `vwap()` accepts `session_col` to anchor the VWAP per trading date or any
  session label. Without it the result is unchanged.
- Indicator correctness suite: live comparisons against TA-Lib 0.8.0 (new
  test-only dev dependency), frozen pandas-ta references, independent formula
  transcriptions and known-answer properties. Existing EMA, DEMA, TEMA, MACD,
  ATR and ADX seed their smoothing from the first bar; their behaviour is
  unchanged and the suite asserts they converge to TA-Lib.

## [1.1.0] - 2026-09-14

- Core mean/min-variance and max-Sharpe optimizers enforce positive long-only
  weight floors and gross limits, reject unsupported constraints, and validate
  solver results without rounding positions or substituting allocations.
  Shared covariance preparation rejects non-finite or insufficient data.
- Distinguish CPCV fold counts from complete backtest paths; add deterministic
  path reconstruction and per-path metrics. Summary metrics now average full
  path scores instead of pooling overlapping test observations. Invalid folds
  and non-finite outputs raise; purge/embargo tests now assert real boundaries.
- Minimum-variance optimization now raises when the solver fails or returns
  non-finite, unnormalized, or out-of-bounds weights. Infeasible constraints
  no longer silently produce an equal-weight portfolio.
- Replace misleading Braket-to-classical fallbacks with genuine, seeded local
  gate-model QAOA simulation. Install the new quantum-braket extra.
- Paid AWS simulators and physical QPUs now raise NotImplementedError until
  durable budget enforcement and reconciliation are available. Explicitly use
  BraketSolver(device="local"); old device aliases no longer return CPU results
  labeled as hardware execution.
- Correct simulated-annealing off-diagonal coefficients to match x'Qx and add
  matrix/finite-number validation. No quantum advantage is claimed.
- Remove the ineffective quantum usage POST. This standalone library does not
  bill CPZ accounts. Private routing and hedge formulations are not distributed.

## [1.0.0] - 2026-07-29

First public release. cpz-quant is the open-source research core of the CPZAI systematic trading operating system, extracted from the cpz-ai SDK and licensed under Apache-2.0.

### Added

- `cpz_quant.portfolio`: 20+ portfolio allocators (mean-variance, min-variance, max-Sharpe, risk parity, HRP, HERC, NCO, Schur complementary, Black-Litterman, entropy pooling, mean-CVaR, robust MVO, max diversification, min tracking error, turnover-penalized, alpha-risk-cost, QUBO / quantum-inspired), covariance estimation (Ledoit-Wolf, OAS, EWMA, factor model, Marchenko-Pastur denoising, detoning, Gerber), factor models, 17-measure mean-risk optimizer, entropy pooling, copulas and synthetic scenarios, transaction cost and capacity models (Almgren-Chriss), Brinson-Fachler and factor attribution, pre-selection transformers, `WalkForward` and `CombinatorialPurgedCV` model selection, optional scikit-learn estimator wrappers.
- **New in this release**: convex optimization backend (`cpz_quant.portfolio.convex`, `pip install cpz-quant[cvx]`) with exact cvxpy programs: `mean_variance_cvx`, `mean_cvar_cvx` (Rockafellar-Uryasev LP), `robust_mean_variance_cvx` (box and ellipsoidal uncertainty sets), and `cardinality_constrained_cvx` (exact mixed-integer cardinality with semi-continuous bounds, `[cvx-mip]`).
- `cpz_quant.certification`: Probability of Backtest Overfitting (CSCV), Deflated Sharpe Ratio gates, institutional risk analytics (Sortino, Calmar, CVaR, tail ratio, MinTRL, capture ratios), regime-conditional performance, factor attribution, and the CPZ Certification grade engine.
- `cpz_quant.indicators`: vectorised momentum, trend, volatility, volume, and statistical indicators with optional Rust acceleration and pure-Python fallback.
- Strict mypy typing (`py.typed`), 12-module test suite, CI on Python 3.9-3.12.
