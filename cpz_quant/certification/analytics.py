"""Institutional risk analytics for the CPZAI Certification Standard.

Bit-parity with the edge engine (`supabase/functions/_shared/risk-analytics.ts`)
and the Rust fast path (`rust/src/analytics.rs`): downside-aware ratios, tail
risk, drawdown geometry, statistical significance (Lopez de Prado MinTRL), and
benchmark-conditional behavior — computed deterministically from an equity curve.

Uses the compiled Rust extension `cpz_risk_rs.certification_analytics` when
available and falls back to a pure-NumPy implementation otherwise; both round
identically so the result is backend-independent.

Fail closed: too-short or degenerate series return ``None`` rather than
fabricated numbers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

try:  # optional Rust acceleration (built via `cd rust && maturin develop --release`)
    import cpz_risk_rs  # type: ignore

    # An older compiled extension may predate this symbol — verify it, don't
    # assume it (mirrors cpz_quant.indicators._rust_accel).
    _RUST = hasattr(cpz_risk_rs, "certification_analytics")
except ImportError:  # pragma: no cover - depends on build
    _RUST = False

TRADING_DAYS = 252.0

# Field -> decimal places, matching the TypeScript engine exactly.
_ROUND = {
    "annualized_return": 4, "annualized_vol": 4, "max_drawdown": 4, "downside_deviation": 4,
    "var95": 4, "cvar95": 4, "cvar99": 4, "crisis_return": 4, "crisis_benchmark_return": 4,
    "worst_month": 4, "best_month": 4,
    "sharpe": 3, "sortino": 3, "calmar": 3, "ulcer_index": 3, "tail_ratio": 3,
    "skew": 3, "excess_kurtosis": 3, "beta": 3, "correlation": 3,
    "up_capture": 3, "down_capture": 3,
    "min_trl_years": 2, "track_years": 2,
}


@dataclass
class RiskAnalytics:
    """Certification analytics block. Percentage-like fields are fractions."""

    observations: int
    annualized_return: Optional[float]
    annualized_vol: Optional[float]
    sharpe: Optional[float]
    sortino: Optional[float]
    calmar: Optional[float]
    max_drawdown: Optional[float]
    ulcer_index: Optional[float]
    max_time_under_water_days: Optional[int]
    downside_deviation: Optional[float]
    var95: Optional[float]
    cvar95: Optional[float]
    cvar99: Optional[float]
    tail_ratio: Optional[float]
    skew: Optional[float]
    excess_kurtosis: Optional[float]
    min_trl_years: Optional[float]
    track_years: Optional[float]
    significant: Optional[bool]
    beta: Optional[float]
    correlation: Optional[float]
    up_capture: Optional[float]
    down_capture: Optional[float]
    crisis_return: Optional[float]
    crisis_benchmark_return: Optional[float]
    worst_month: Optional[float] = None
    best_month: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def has_rust() -> bool:
    """True when the Rust acceleration extension is importable."""
    return _RUST


# ── pure-NumPy core (parity with rust/src/analytics.rs) ──────────────────────
def _to_returns(eq: np.ndarray) -> np.ndarray:
    prev = eq[:-1]
    cur = eq[1:]
    mask = (prev > 0) & np.isfinite(cur) & np.isfinite(prev)
    return (cur[mask] / prev[mask]) - 1.0


def _percentile(xs: np.ndarray, p: float) -> float:
    # numpy 'linear' == the (n-1)*p interpolation used by the TS/Rust engines.
    return float(np.percentile(xs, p * 100.0, method="linear"))


def _compute_numpy(equity: Sequence[float], benchmark: Optional[Sequence[float]]) -> Optional[dict]:
    eq_in = np.asarray(equity, dtype=float)
    has_bench = benchmark is not None and len(benchmark) == len(eq_in)
    bench_in = np.asarray(benchmark, dtype=float) if has_bench else None

    keep = np.isfinite(eq_in) & (eq_in > 0)
    eq = eq_in[keep]
    bench = bench_in[keep] if has_bench else None  # type: ignore[index]
    if eq.size < 30:
        return None
    rets = _to_returns(eq)
    n = rets.size
    if n < 20:
        return None

    mu = float(rets.mean())
    sd = float(rets.std(ddof=1))
    ann_return = float((eq[-1] / eq[0]) ** (TRADING_DAYS / n) - 1.0)
    ann_vol = sd * np.sqrt(TRADING_DAYS)
    sharpe = (mu / sd) * np.sqrt(TRADING_DAYS) if sd > 0 else None

    downs = np.where(rets < 0, rets, 0.0)
    down_dev = float(np.sqrt(np.mean(downs * downs)))
    sortino = (mu / down_dev) * np.sqrt(TRADING_DAYS) if down_dev > 0 else None

    peak = eq[0]
    max_dd = 0.0
    sum_sq_dd = 0.0
    cur_uw = 0
    max_uw = 0
    for e in eq:
        if e > peak:
            peak = e
            cur_uw = 0
        else:
            cur_uw += 1
            max_uw = max(max_uw, cur_uw)
        dd = (e / peak - 1.0) if peak > 0 else 0.0
        max_dd = min(max_dd, dd)
        sum_sq_dd += (dd * 100.0) ** 2
    ulcer = float(np.sqrt(sum_sq_dd / eq.size))
    calmar = (ann_return / abs(max_dd)) if max_dd < 0 else None

    var95 = _percentile(rets, 0.05)
    below95 = rets[rets <= var95]
    cvar95 = float(below95.mean()) if below95.size else var95
    var99 = _percentile(rets, 0.01)
    below99 = rets[rets <= var99]
    cvar99 = float(below99.mean()) if below99.size else var99
    p95 = _percentile(rets, 0.95)
    tail_ratio = (abs(p95) / abs(var95)) if var95 != 0 else None

    m3 = float(np.mean((rets - mu) ** 3))
    m4 = float(np.mean((rets - mu) ** 4))
    skew = (m3 / sd**3) if sd > 0 else None
    kurt = (m4 / sd**4) if sd > 0 else None
    excess_kurtosis = (kurt - 3.0) if kurt is not None else None

    min_trl_years = None
    srp = (mu / sd) if sd > 0 else None
    if srp is not None and srp > 0 and skew is not None and kurt is not None:
        z = 1.645
        min_obs = 1.0 + (1.0 - skew * srp + ((kurt - 1.0) / 4.0) * srp * srp) * (z / srp) ** 2
        if np.isfinite(min_obs) and min_obs > 0:
            min_trl_years = min_obs / TRADING_DAYS
    track_years = n / TRADING_DAYS
    significant = (track_years >= min_trl_years) if min_trl_years is not None else None

    beta = correlation = up_capture = down_capture = None
    crisis_return = crisis_benchmark_return = None
    if bench is not None and np.all(np.isfinite(bench)) and np.all(bench > 0):
        b_rets = _to_returns(bench)
        if b_rets.size == n:
            b_mu = float(b_rets.mean())
            b_sd = float(b_rets.std(ddof=1))
            cov = float(np.sum((rets - mu) * (b_rets - b_mu)) / (n - 1))
            beta = (cov / (b_sd * b_sd)) if b_sd > 0 else None
            correlation = (cov / (sd * b_sd)) if sd > 0 and b_sd > 0 else None

            up = b_rets > 0
            dn = b_rets < 0
            s_up = float(np.prod(1 + rets[up])) if up.any() else 1.0
            b_up = float(np.prod(1 + b_rets[up])) if up.any() else 1.0
            s_dn = float(np.prod(1 + rets[dn])) if dn.any() else 1.0
            b_dn = float(np.prod(1 + b_rets[dn])) if dn.any() else 1.0
            if up.any() and b_up != 1.0:
                up_capture = (s_up - 1.0) / (b_up - 1.0)
            if dn.any() and b_dn != 1.0:
                down_capture = (s_dn - 1.0) / (b_dn - 1.0)

            b_peak = bench[0]
            b_peak_idx = 0
            worst_dd = 0.0
            t_idx = 0
            p_for_t = 0
            for i, b in enumerate(bench):
                if b > b_peak:
                    b_peak = b
                    b_peak_idx = i
                dd = b / b_peak - 1.0
                if dd < worst_dd:
                    worst_dd = dd
                    t_idx = i
                    p_for_t = b_peak_idx
            if worst_dd < 0 and t_idx > p_for_t:
                crisis_benchmark_return = float(bench[t_idx] / bench[p_for_t] - 1.0)
                crisis_return = float(eq[t_idx] / eq[p_for_t] - 1.0)

    return {
        "observations": int(n),
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "ulcer_index": ulcer,
        "max_time_under_water_days": int(max_uw),
        "downside_deviation": down_dev * np.sqrt(TRADING_DAYS),
        "var95": var95,
        "cvar95": cvar95,
        "cvar99": cvar99,
        "tail_ratio": tail_ratio,
        "skew": skew,
        "excess_kurtosis": excess_kurtosis,
        "min_trl_years": min_trl_years,
        "track_years": track_years,
        "significant": significant,
        "beta": beta,
        "correlation": correlation,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "crisis_return": crisis_return,
        "crisis_benchmark_return": crisis_benchmark_return,
    }


def _round(raw: dict) -> dict:
    out = dict(raw)
    for k, dp in _ROUND.items():
        v = out.get(k)
        if v is not None and np.isfinite(v):
            out[k] = round(float(v), dp)
        elif v is not None and not np.isfinite(v):
            out[k] = None
    return out


def _worst_best_month(equity: Sequence[float], dates: Sequence[str]) -> tuple[Optional[float], Optional[float]]:
    if not dates or len(dates) != len(equity):
        return None, None
    buckets: dict[str, list[float]] = {}
    for d, p in zip(dates, equity):
        if not isinstance(d, str):
            return None, None
        buckets.setdefault(d[:7], [None, None])  # type: ignore
        b = buckets[d[:7]]
        if b[0] is None:
            b[0] = p
        b[1] = p
    monthlies = [b[1] / b[0] - 1.0 for b in buckets.values() if b[0]]
    if not monthlies:
        return None, None
    return round(min(monthlies), 4), round(max(monthlies), 4)


def compute_risk_analytics(
    equity: Sequence[float],
    benchmark: Optional[Sequence[float]] = None,
    dates: Optional[Sequence[str]] = None,
) -> Optional[RiskAnalytics]:
    """Compute the certification analytics block from a daily equity curve.

    Args:
        equity: daily portfolio equity values (>= 30 points).
        benchmark: optional aligned benchmark equity series (same length).
        dates: optional aligned ``YYYY-MM-DD`` strings (enables worst/best month).

    Returns ``None`` when the series is too short to be meaningful.
    """
    raw: Optional[dict]
    if _RUST:
        raw = cpz_risk_rs.certification_analytics(
            [float(x) for x in equity],
            [float(x) for x in benchmark] if benchmark is not None else None,
        )
    else:
        raw = _compute_numpy(equity, benchmark)
    if raw is None:
        return None
    rounded = _round(raw)
    wm, bm = _worst_best_month(equity, dates) if dates is not None else (None, None)
    rounded["worst_month"] = wm
    rounded["best_month"] = bm
    return RiskAnalytics(**rounded)
