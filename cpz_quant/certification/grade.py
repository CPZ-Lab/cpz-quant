"""CPZAI Certification Standard — deterministic grading engine (Python parity).

A faithful port of the edge engine (``supabase/functions/_shared/certification.ts``,
v1.1.0): six weighted dimensions, minimum gates, provisional-vs-full, grade
bands, and the tail-risk-aware Risk score. Same thresholds → same grade, so the
SDK, the backtest engine, and the edge all agree.

Fail closed: absent inputs push scores down, never up (a missing beta or
capacity scores the same as the worst measured value, never better); a
strategy with no rigorous, verified backtest cannot be certified. Numeric
inputs may be Python or NumPy numbers and flags Python or NumPy booleans.

Units: ``CertMetrics.max_drawdown`` is in percent (``-62.0`` for a 62%
drawdown), while :class:`RiskAnalytics` reports fractions (``-0.62``).
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .analytics import RiskAnalytics

CERTIFICATION_ENGINE_VERSION = "1.1.0"

MIN_LIVE_TRADES = 20
MIN_LIVE_DAYS = 30

DIMENSION_WEIGHTS = {
    "robustness": 0.25,
    "outOfSample": 0.20,
    "risk": 0.15,
    "factorIndependence": 0.15,
    "capacityCost": 0.10,
    "liveTrackRecord": 0.15,
}

Grade = str  # 'A+' | 'A' | 'B' | 'C' | 'D' | 'Not Certified'


@dataclass
class CertMetrics:
    """Backtest metrics. ``max_drawdown`` is in percent (sign ignored):
    pass ``-62.0`` for a 62% drawdown, i.e. ``RiskAnalytics.max_drawdown * 100``."""
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    volatility: Optional[float] = None
    profit_factor: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    total_trades: Optional[float] = None


@dataclass
class CertRigor:
    deflated_sharpe: Optional[float] = None
    probabilistic_sharpe: Optional[float] = None
    num_trials: Optional[float] = None
    oos_sharpe: Optional[float] = None
    oos_return: Optional[float] = None
    turnover: Optional[float] = None
    capacity: Optional[float] = None
    parameter_stability: Optional[bool] = None


@dataclass
class CertLive:
    sharpe_ratio: Optional[float] = None
    total_return_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    total_trades: Optional[float] = None
    alpha: Optional[float] = None
    elapsed_days: Optional[float] = None


@dataclass
class DimensionScore:
    key: str
    label: str
    weight: float
    score: Optional[float]
    detail: str


@dataclass
class GateResult:
    key: str
    passed: bool
    detail: str


@dataclass
class CertificationResult:
    engine_version: str
    grade: Grade
    composite: Optional[float]
    provisional: bool
    certified: bool
    dimensions: List[DimensionScore]
    gates: List[GateResult]
    gates_passed: bool
    summary: str


# ── helpers ──────────────────────────────────────────────────────────────────
def _finite(v) -> Optional[float]:
    """A finite real number (Python or NumPy, any width) as float, else None."""
    if isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_)):
        f = float(v)
        return f if math.isfinite(f) else None
    return None


def _flag(v) -> Optional[bool]:
    """A Python or NumPy boolean as bool; anything else is not a flag (None)."""
    return bool(v) if isinstance(v, (bool, np.bool_)) else None


def _drawdown_pct(metrics: Optional[CertMetrics], analytics: Optional[RiskAnalytics]) -> Optional[float]:
    """``|max_drawdown|`` in percent, refusing a value that is evidently a fraction.

    ``CertMetrics.max_drawdown`` is in percent; :func:`compute_risk_analytics`
    reports a fraction (``-0.62`` for 62%, ``-1.0`` for a wiped-out account).

    - With *analytics*, a value that matches ``analytics.max_drawdown`` as a
      fraction rather than as a percent raises, whatever its size (``-1.0``
      for a wipe-out would otherwise read as 1%).
    - With *analytics*, a value that contradicts ``analytics.max_drawdown *
      100`` (beyond 10% relative and half a point) raises: the two describe
      the same backtest.
    - A magnitude of 1 or less is accepted only when the analytics confirm
      it as a percent; without them it raises. (Without analytics a
      magnitude above 1 cannot be told apart from a percent and is read as
      one.)
    """
    dd = _finite(metrics.max_drawdown) if metrics else None
    if dd is None:
        return None
    abs_dd = abs(dd)
    a_dd = _finite(analytics.max_drawdown) if analytics is not None else None
    as_pct = a_dd is not None and math.isclose(abs(a_dd) * 100, abs_dd, rel_tol=0.05, abs_tol=0.01)
    as_fraction = (a_dd is not None and a_dd != 0
                   and math.isclose(abs(a_dd), abs_dd, rel_tol=0.05, abs_tol=1e-4))
    if a_dd is not None and as_fraction and not as_pct:
        raise ValueError(
            f"metrics.max_drawdown={dd} equals analytics.max_drawdown ({a_dd}), which is a "
            f"fraction, so it would be read as a {abs_dd:.2f}% drawdown. It is in percent: pass "
            f"{a_dd * 100:.4g} (analytics.max_drawdown * 100)."
        )
    if a_dd is not None and not math.isclose(abs(a_dd) * 100, abs_dd, rel_tol=0.10, abs_tol=0.5):
        raise ValueError(
            f"metrics.max_drawdown={dd} (percent) contradicts analytics.max_drawdown={a_dd} "
            f"({abs(a_dd) * 100:.2f}%); they should describe the same backtest."
        )
    if 0 < abs_dd <= 1 and not as_pct:
        raise ValueError(
            f"metrics.max_drawdown={dd} would be read as a {abs_dd:.2f}% drawdown. It is in "
            "percent (-62.0 for a 62% drawdown, -100.0 for a wipe-out); compute_risk_analytics() "
            "reports a fraction, so multiply it by 100. To certify a drawdown of 1% or less, "
            "pass analytics= so the value can be confirmed."
        )
    return abs_dd


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _band(value: float, bands: List[tuple], below: float) -> float:
    for threshold, points in bands:
        if value >= threshold:
            return points
    return below


# ── dimensions ───────────────────────────────────────────────────────────────
def score_robustness(rigor: Optional[CertRigor]) -> tuple[float, str]:
    dsr = _finite(rigor.deflated_sharpe) if rigor else None
    if dsr is None:
        return 0.0, "No deflated Sharpe from the rigorous engine. Cannot attest robustness."
    if dsr <= 0:
        return 0.0, f"Deflated Sharpe {dsr:.2f} <= 0: edge does not survive multiple-testing deflation."
    score = _band(dsr, [(0.95, 100), (0.90, 90), (0.75, 75), (0.50, 55), (0.25, 35)], 20)
    psr = _finite(rigor.probabilistic_sharpe) if rigor else None
    if psr is not None:
        score -= 0 if psr >= 0.95 else 5 if psr >= 0.90 else 12 if psr >= 0.75 else 22 if psr >= 0.5 else 35
    trials = _finite(rigor.num_trials) if rigor else None
    if trials is not None and trials <= 1:
        score = min(score, 60)
    return _clamp(score, 0, 100), f"Deflated Sharpe {dsr:.2f}."


def score_out_of_sample(rigor: Optional[CertRigor], in_sample_sharpe: Optional[float]) -> tuple[float, str]:
    oos = _finite(rigor.oos_sharpe) if rigor else None
    if oos is None:
        return 0.0, "No out-of-sample Sharpe from walk-forward. Fails closed."
    if oos <= 0:
        return 5.0, f"Out-of-sample Sharpe {oos:.2f} is not positive: edge did not persist OOS."
    is_ = _finite(in_sample_sharpe)
    if is_ is None or is_ <= 0:
        score = _band(oos, [(1.0, 80), (0.5, 60), (0.25, 45)], 30)
        return score, f"Out-of-sample Sharpe {oos:.2f}."
    decay = _clamp(oos / is_, 0, 1.2)
    score = _band(decay, [(1.0, 100), (0.9, 90), (0.7, 75), (0.5, 55), (0.35, 40), (0.2, 25)], 10)
    return score, f"OOS Sharpe {oos:.2f} vs in-sample {is_:.2f}: {round(decay * 100)}% held out of sample."


def score_risk(metrics: Optional[CertMetrics], analytics: Optional[RiskAnalytics] = None) -> tuple[float, str]:
    abs_dd = _drawdown_pct(metrics, analytics)
    if abs_dd is None:
        return 0.0, "No max drawdown recorded. Cannot assess loss control."
    score = _band(100 - abs_dd, [(90, 100), (80, 80), (70, 60), (60, 40), (45, 20)], 5)
    pf = _finite(metrics.profit_factor) if metrics else None
    if pf is not None:
        if pf < 1:
            score = min(score, 45)
        elif pf >= 2:
            score = min(100, score + 5)
    extras: List[str] = []
    if analytics is not None:
        if analytics.sortino is not None:
            score += 5 if analytics.sortino >= 2 else (-15 if analytics.sortino < 0.5 else 0)
            extras.append(f"Sortino {analytics.sortino:.2f}")
        if analytics.calmar is not None:
            score += 5 if analytics.calmar >= 1 else (-10 if analytics.calmar < 0.3 else 0)
            extras.append(f"Calmar {analytics.calmar:.2f}")
        if analytics.tail_ratio is not None and analytics.tail_ratio < 0.8:
            score -= 10
            extras.append(f"tail ratio {analytics.tail_ratio:.2f}")
        if analytics.excess_kurtosis is not None and analytics.excess_kurtosis > 5:
            score -= 5
            extras.append(f"excess kurtosis {analytics.excess_kurtosis:.1f}")
    detail = f"Max drawdown {abs_dd:.1f}%"
    if pf is not None:
        detail += f", profit factor {pf:.2f}"
    if extras:
        detail += ", " + ", ".join(extras)
    return _clamp(score, 0, 100), detail + "."


def score_factor_independence(metrics: Optional[CertMetrics]) -> tuple[float, str]:
    beta = _finite(metrics.beta) if metrics else None
    alpha = _finite(metrics.alpha) if metrics else None
    if beta is None:
        # Fail closed: the floor of the beta bands (as for |beta| >= 0.8), so
        # leaving beta out can never score above a measured beta.
        return 20.0, ("Beta to benchmark unavailable; independence cannot be attested "
                      "(scored as the lowest band).")
    abs_beta = abs(beta)
    score = _band(1 - abs_beta, [(0.8, 100), (0.6, 80), (0.4, 60), (0.2, 40)], 20)
    if alpha is not None:
        if alpha > 0:
            score = min(100, score + 5)
        elif alpha < 0:
            score = min(score, 50)
    return _clamp(score, 0, 100), f"Beta {beta:.2f} to benchmark."


def score_capacity_cost(rigor: Optional[CertRigor]) -> tuple[float, str]:
    cap = _finite(rigor.capacity) if rigor else None
    if cap is None:
        # Fail closed: the floor of the capacity bands (as for under $50,000).
        score = 10.0
        detail = "Capacity not estimated by the cost model (scored as the lowest band)."
    else:
        score = _band(cap, [(50_000_000, 100), (10_000_000, 85), (1_000_000, 65), (250_000, 45), (50_000, 25)], 10)
        detail = f"Estimated capacity ${round(cap):,}."
    turnover = _finite(rigor.turnover) if rigor else None
    if turnover is not None:
        if turnover > 10:
            score -= 15
        elif turnover > 5:
            score -= 8
        detail += f" Annualized turnover {turnover:.2f}x."
    return _clamp(score, 0, 100), detail


def score_live_track_record(live: Optional[CertLive], backtest_sharpe: Optional[float]) -> tuple[float, str]:
    if live is None:
        return 0.0, "No qualifying live/paper track record."
    live_sharpe = _finite(live.sharpe_ratio)
    bt = _finite(backtest_sharpe)
    if live_sharpe is None:
        return 20.0, "Live record present but Sharpe unavailable."
    ratio = _clamp(live_sharpe / bt, 0, 1.5) if (bt is not None and bt > 0) else (1.0 if live_sharpe > 0 else 0.0)
    score = _band(ratio, [(0.8, 100), (0.6, 85), (0.4, 65), (0.2, 45), (0.001, 30)], 10)
    return score, f"Live Sharpe {live_sharpe:.2f} over {live.total_trades or '?'} trades."


# ── gates ────────────────────────────────────────────────────────────────────
def evaluate_gates(
    *, backtest_verified: bool, has_high_severity_lookahead: bool,
    metrics: Optional[CertMetrics], rigor: Optional[CertRigor],
) -> List[GateResult]:
    dsr = _finite(rigor.deflated_sharpe) if rigor else None
    sharpe = _finite(metrics.sharpe_ratio) if metrics else None
    dd = _finite(metrics.max_drawdown) if metrics else None
    verified = _flag(backtest_verified) is True
    no_lookahead = _flag(has_high_severity_lookahead) is False
    return [
        GateResult("backtest_verified", verified,
                   "Backtest passed every Trust Layer verification check." if verified else "Backtest is not fully verified."),
        GateResult("no_high_severity_lookahead", no_lookahead,
                   "No high-severity lookahead patterns." if no_lookahead
                   else "Lookahead check result missing." if _flag(has_high_severity_lookahead) is None
                   else "High-severity lookahead pattern detected."),
        GateResult("deflated_sharpe_positive", dsr is not None and dsr > 0,
                   "No rigorous engine block (deflated Sharpe absent)." if dsr is None else f"Deflated Sharpe {dsr:.2f} {'> 0.' if dsr > 0 else '<= 0.'}"),
        GateResult("core_metrics_present", sharpe is not None and dd is not None,
                   "Core risk/return metrics are present and finite." if (sharpe is not None and dd is not None) else "Core metrics missing or non-finite."),
    ]


def _grade_from_composite(composite: float, provisional: bool) -> Grade:
    if composite >= 90:
        return "A" if provisional else "A+"
    if composite >= 80:
        return "A"
    if composite >= 70:
        return "B"
    if composite >= 60:
        return "C"
    if composite >= 50:
        return "D"
    return "Not Certified"


def certify(
    *,
    backtest_verified: bool,
    has_high_severity_lookahead: bool = False,
    metrics: Optional[CertMetrics] = None,
    rigor: Optional[CertRigor] = None,
    live: Optional[CertLive] = None,
    analytics: Optional[RiskAnalytics] = None,
) -> CertificationResult:
    """Grade a strategy against the CPZAI Certification Standard.

    Raises:
        ValueError: ``metrics.max_drawdown`` looks like a fraction (see
            :class:`CertMetrics`: it is in percent).
    """
    gates = evaluate_gates(
        backtest_verified=backtest_verified, has_high_severity_lookahead=has_high_severity_lookahead,
        metrics=metrics, rigor=rigor,
    )
    gates_passed = all(g.passed for g in gates)
    in_sample_sharpe = _finite(metrics.sharpe_ratio) if metrics else None

    r_score, r_detail = score_robustness(rigor)
    o_score, o_detail = score_out_of_sample(rigor, in_sample_sharpe)
    rk_score, rk_detail = score_risk(metrics, analytics)
    f_score, f_detail = score_factor_independence(metrics)
    c_score, c_detail = score_capacity_cost(rigor)

    live_trades = _finite(live.total_trades) if live else None
    live_days = _finite(live.elapsed_days) if live else None
    live_qualifies = bool(live) and live_trades is not None and live_trades >= MIN_LIVE_TRADES \
        and live_days is not None and live_days >= MIN_LIVE_DAYS
    provisional = not live_qualifies
    if live_qualifies:
        l_score, l_detail = score_live_track_record(live, in_sample_sharpe)
    else:
        l_score = 0.0
        l_detail = (f"Live record below Full thresholds (needs >= {MIN_LIVE_TRADES} trades and >= {MIN_LIVE_DAYS} days)."
                    if live else "No live/paper track record yet.")

    w = dict(DIMENSION_WEIGHTS)
    if provisional:
        kept = w["robustness"] + w["outOfSample"] + w["risk"] + w["factorIndependence"] + w["capacityCost"]
        weights = {k: (w[k] / kept if k != "liveTrackRecord" else 0.0) for k in w}
    else:
        weights = w

    dimensions = [
        DimensionScore("robustness", "Statistical Robustness", weights["robustness"], r_score, r_detail),
        DimensionScore("outOfSample", "Out-of-Sample Integrity", weights["outOfSample"], o_score, o_detail),
        DimensionScore("risk", "Risk Profile", weights["risk"], rk_score, rk_detail),
        DimensionScore("factorIndependence", "Factor Independence", weights["factorIndependence"], f_score, f_detail),
        DimensionScore("capacityCost", "Capacity & Cost", weights["capacityCost"], c_score, c_detail),
        DimensionScore("liveTrackRecord", "Live Track Record", weights["liveTrackRecord"],
                       None if provisional else l_score, l_detail),
    ]

    composite = 0.0
    for d in dimensions:
        if d.weight > 0 and d.score is not None:
            composite += d.score * d.weight
    composite = round(_clamp(composite, 0, 100), 1)

    grade = "Not Certified" if not gates_passed else _grade_from_composite(composite, provisional)
    certified = grade != "Not Certified"

    if not gates_passed:
        summary = "Not Certified: " + ", ".join(g.key for g in gates if not g.passed) + " failed."
    else:
        summary = (f"{'Provisional ' if provisional else ''}{grade} (composite {composite}/100)"
                   + (" — backtest only; add a live track record to lift the cap and reach A+." if provisional else "."))

    return CertificationResult(
        engine_version=CERTIFICATION_ENGINE_VERSION,
        grade=grade,
        composite=composite if gates_passed else None,
        provisional=provisional,
        certified=certified,
        dimensions=dimensions,
        gates=gates,
        gates_passed=gates_passed,
        summary=summary,
    )
