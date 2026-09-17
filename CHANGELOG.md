# Changelog

All notable changes to cpz-quant are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
