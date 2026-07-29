"""CPZAI Certification Standard — the SDK-side grading + institutional analytics.

Parity with the edge engine (``supabase/functions/_shared``) so the SDK, the
backtest engine, and the hosted certification all produce the same grade and the
same due-diligence analytics from an equity curve.

- :func:`compute_risk_analytics` — institutional analytics (Sortino, Calmar,
  CVaR, tail ratio, MinTRL, up/down capture, crisis behavior). Rust-accelerated.
- :func:`certify` — the six-dimension letter grade with minimum gates.
- :func:`probability_of_backtest_overfitting` — PBO via CSCV.
- :func:`factor_attribution` — idiosyncratic alpha after neutralizing factors.
"""

from __future__ import annotations

from .analytics import RiskAnalytics, compute_risk_analytics, has_rust
from .factors import FactorAttribution, factor_attribution
from .grade import (
    CERTIFICATION_ENGINE_VERSION,
    CertificationResult,
    CertLive,
    CertMetrics,
    CertRigor,
    DimensionScore,
    GateResult,
    certify,
)
from .overfitting import PBOResult, probability_of_backtest_overfitting
from .regime import RegimeBreakdown, RegimeStats, regime_conditional_performance

__all__ = [
    "RiskAnalytics",
    "compute_risk_analytics",
    "has_rust",
    "certify",
    "CertificationResult",
    "CertMetrics",
    "CertRigor",
    "CertLive",
    "DimensionScore",
    "GateResult",
    "CERTIFICATION_ENGINE_VERSION",
    "probability_of_backtest_overfitting",
    "PBOResult",
    "factor_attribution",
    "FactorAttribution",
    "regime_conditional_performance",
    "RegimeBreakdown",
    "RegimeStats",
]
