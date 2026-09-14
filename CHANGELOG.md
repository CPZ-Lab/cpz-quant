# Changelog

All notable changes to cpz-quant are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-09-14

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
