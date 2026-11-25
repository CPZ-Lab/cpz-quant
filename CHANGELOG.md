# Changelog

All notable changes to cpz-quant are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed (may break callers)

- Scalar indicator calls on an empty input raise `ValueError` (they returned
  0.0). A NaN latest value is returned as NaN with a `RuntimeWarning`
  instead of the last value before it.
- After one missing bar, EMA, DEMA, TEMA, MACD, RSI, ATR, ADX, Supertrend,
  OBV and the other recursive indicators stay NaN for the rest of the
  series (as TA-Lib); window indicators recover once the gap leaves the
  window. Callers that read `ind.atr(...)` as a size input get NaN on data
  with one null high: cpz-ai (cpz-py) `atr_size` needs an `isfinite` guard.
- A NaN inside a window gives NaN in CCI, z-scores, ROC, momentum, STC, MFI,
  Williams %R, stochastic, Fisher, ADX, CMO, CMF, Aroon, rolling correlation
  and beta, and the regression R-squared (they used to read 0, 50, -50, 0.5
  or another finite value); NVI and PVI stay NaN after a missing bar. Flat prices give NaN in RSI, MFI, the
  Ultimate Oscillator, fast stochastic and Williams %R.
- `certify` raises when `max_drawdown` looks like a fraction: a magnitude of
  1 or less without `analytics=` confirming it, or a value equal to
  `analytics.max_drawdown` (including -1.0 for a wipe-out). With
  `analytics=` it also raises when the two drawdowns contradict each other.
- `turnover_analysis.estimated_cost_bps` is 30 to 100 times smaller for
  partial rebalances (it is now bps of the book). Without a cost model the
  cost fields are `None`. With one, a traded symbol without ADV, or without
  volatility for the sqrt and Almgren-Chriss models, raises (they were
  filled with 1% of the book and 2% a day). `sqrt_impact`, `linear_impact`
  and `almgren_chriss` raise on a missing ADV or volatility instead of using
  a volatility of 1.0 or returning zero impact.
- `hurst()` reads about 0.13 lower for the same series (bias removed); the
  default `min_lag` is 10.
- `fundamental_factors` needs market caps as series aligned with the returns,
  or `caps_at_start=True` for single start-of-sample numbers; with momentum
  it regresses only days from `momentum_window` on (`start_index`).
- `alpha_decay_capacity`, `brinson_fachler` and `fundamental_factors` raise on
  inputs they used to drop or default.
- `factor_attribution` returns `None` for t-stats and the idiosyncratic
  Sharpe of an exact fit.

### Fixed

- Indicators: the scalar form returns the value at the latest bar. A missing
  bar no longer hands back a value from before the gap; SMA and other window
  indicators blank only the windows holding the gap (as pandas), recursive
  smoothers (EMA family, MACD, RSI, ATR) turn NaN as in TA-Lib, and a NaN
  latest value warns. An empty input raises `ValueError` instead of
  returning 0.
- Indicators of indicators (SMA, EMA, DEMA, TEMA, Bollinger, MACD, KAMA of
  RSI or ATR; RSI of an SMA) skip the input's warm-up instead of returning
  NaN everywhere or counting it as flat bars; they match TA-Lib.
  Stochastic %D no longer fills the %K warm-up with 50.
- RSI on an unmoving price is NaN (0/0), not 0. The Rust kernel matches.
- `momentum()` keeps working after `rsi()` and other momentum indicators run
  (the submodule of the same name no longer replaces it).
- pandas DataFrames and Series work in every indicator; a 2-D array, a
  derived `source` on a bare series and `obv()` without volume raise. The
  Rust Garman-Klass kernel is only used with `annualize=True` and 252 days.
  `TRADING_DAYS_PER_YEAR` is documented as a default, not a global setting.
- Ichimoku refuses `displacement < 1` (it shifted spans onto later bars);
  Supertrend flips on the bar that closes through the band (pandas-ta);
  EMA and MACD docstrings state their first-value seeding.
- Hurst exponent uses the bias-corrected (Anis-Lloyd-Peters) rescaled
  range, so a random walk reads about 0.5 instead of 0.63; the default
  `min_lag` is 10 and too-short input gives NaN instead of 0.5.
- `fundamental_factors` builds exposures point in time (momentum from days
  `t-252` to `t-21`, size from the previous close's cap) and no longer finds
  a momentum premium in noise; missing caps or book values raise. Adds
  `start_index` and `momentum_skip`.
- `turnover_analysis` reports the rebalance cost in bps of the book (and
  `estimated_cost_usd`) instead of summing per-trade bps.
- `CostModel.estimate` is documented as one leg; `breakdown()` splits impact
  by `permanent_fraction`; unknown model names raise; Almgren-Chriss refuses
  settings it does not use; sells cost the same as buys in every model.
- `alpha_decay_capacity` counts short positions and refuses positions with
  no ADV.
- `brinson_fachler` raises on holdings missing from the sector map or
  without a return, and counts net-short sectors.
- `alpha_beta_decomposition` and `rolling_attribution`: information ratio is
  mean active return over tracking error. Per-factor risk in
  `factor_risk_decomposition` and `risk_attribution` uses the Euler split,
  which adds up to the factor share. Drawdown risk measures count a loss on
  the first return.
- `cross_validate` and `grid_search` accept Polars and pandas DataFrames.
- Certification analytics: time under water counts only days below the
  peak; up/down capture use Morningstar's geometric per-period ratio;
  worst/best month run month-end to month-end; an equity value at or below
  zero ends the curve as a total loss instead of being dropped, also when
  the curve is too short for the statistics. Rust kernel updated to match.
- Regime breakdown labels each day from data known before it (trailing
  volatility, previous benchmark close) and counts a wipe-out.
- `certify`: a missing beta or capacity scores the lowest band, never above a
  measured value; NumPy numbers and booleans are accepted; a
  `max_drawdown` that looks like a fraction raises (it is in percent).
- `factor_attribution`: an exact factor fit gives `None` alpha t-stat and
  idiosyncratic Sharpe instead of rounding noise.
- `momentum` (indicator and submodule) can be pickled and documents the
  indicator; the NumPy SMA is O(n) again.

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
