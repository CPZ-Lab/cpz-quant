"""cpz-quant — open-source quantitative portfolio construction, risk, and
strategy certification from CPZ Lab.

Three subpackages, all pure functions (data in, results out — no I/O):

- :mod:`cpz_quant.portfolio` — portfolio optimization (mean-variance, HRP,
  HERC, NCO, Black-Litterman, entropy pooling, CVaR, robust, convex/cvxpy,
  QUBO), covariance estimation, factor models, transaction costs,
  attribution, pre-selection, and walk-forward / combinatorial purged
  cross-validation.
- :mod:`cpz_quant.certification` — anti-overfitting strategy certification:
  Probability of Backtest Overfitting (CSCV), Deflated Sharpe Ratio,
  regime-conditional performance, and the CPZ Certification grade.
- :mod:`cpz_quant.indicators` — vectorised technical indicators (momentum,
  trend, volatility, volume, statistical) with optional Rust acceleration.

The proprietary `cpz-ai` SDK builds on this package for live execution,
broker routing, and platform data access; cpz-quant itself is fully
standalone and Apache-2.0 licensed.
"""

from cpz_quant import certification, indicators, portfolio
from cpz_quant.frames import as_returns

__version__ = "1.0.0"

__all__ = ["portfolio", "certification", "indicators", "as_returns", "__version__"]
