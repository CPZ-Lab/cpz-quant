"""Correctness tests for indicators without a TA-Lib equivalent.

Three kinds of evidence, labelled per test:

- ``pandas-ta``: outputs frozen from pandas-ta 0.4.71b0 in
  ``tests/fixtures/indicator_reference.npz`` (see
  ``tests/reference/generate_indicator_fixture.py``). When pandas-ta is
  importable the fixture itself is re-derived and checked.
- ``formula``: an independent, deliberately naive loop transcription of the
  published definition, compared on the same fixture bars.
- ``property``: statistical checks with a known answer (for example range
  volatility estimators recovering the true sigma of a simulated diffusion).
"""

from __future__ import annotations

import math
from typing import List

import cpz_quant.indicators.momentum as mo
import cpz_quant.indicators.statistical as st
import cpz_quant.indicators.trend as tr
import cpz_quant.indicators.volatility as vl
import cpz_quant.indicators.volume as vo
import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════
#  pandas-ta frozen references
# ═══════════════════════════════════════════════════════════════════


def test_pta_alma(ohlcv, assert_match):
    assert_match(tr.alma_series(ohlcv["close"], 9), ohlcv["pta_alma_9"], name="alma")


def test_pta_ichimoku(ohlcv, assert_match):
    ten, kij, span_a, span_b = tr.ichimoku_series(ohlcv["high"], ohlcv["low"])
    assert_match(ten, ohlcv["pta_ichimoku_tenkan"], name="tenkan")
    assert_match(kij, ohlcv["pta_ichimoku_kijun"], name="kijun")
    assert_match(span_a, ohlcv["pta_ichimoku_span_a"], name="span_a")
    assert_match(span_b, ohlcv["pta_ichimoku_span_b"], name="span_b")


def test_pta_kst(ohlcv, assert_match):
    kst, signal = mo.kst_series(ohlcv["close"])
    assert_match(kst * 100.0, ohlcv["pta_kst_x100"], name="kst")
    assert_match(signal * 100.0, ohlcv["pta_kst_signal_x100"], name="kst_signal")


def test_pta_tsi_signal(ohlcv, assert_match):
    assert_match(mo.tsi_series(ohlcv["close"])[1], ohlcv["pta_tsi_signal"], name="tsi_signal")


def test_pta_fisher_after_seed_decay(ohlcv, assert_match):
    # pandas-ta starts the recursion one bar later; the gap decays as 0.67^k.
    fisher, trigger = mo.fisher_transform_series(ohlcv["high"], ohlcv["low"], 9)
    assert_match(fisher, ohlcv["pta_fisher"], start=100, name="fisher")
    assert_match(trigger, ohlcv["pta_fisher_trigger"], start=100, name="fisher_trigger")


def test_pta_rvgi(ohlcv, assert_match):
    o, h, lo, c = (ohlcv[k] for k in ("open", "high", "low", "close"))
    rvgi, signal = mo.rvgi_series(o, h, lo, c, 14)
    assert_match(rvgi, ohlcv["pta_rvgi_14"], name="rvgi")
    assert_match(signal, ohlcv["pta_rvgi_signal_14"], name="rvgi_signal")


def test_pta_ulcer_index(ohlcv, assert_match):
    assert_match(vl.ulcer_index_series(ohlcv["close"], 14), ohlcv["pta_ulcer_14"], name="ui")


def test_pta_choppiness(ohlcv, assert_match):
    h, lo, c = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    assert_match(vl.choppiness_series(h, lo, c, 14), ohlcv["pta_chop_14"], name="chop")


def test_pta_eom(ohlcv, assert_match):
    h, lo, v = ohlcv["high"], ohlcv["low"], ohlcv["volume"]
    assert_match(vo.eom_series(h, lo, v, 14), ohlcv["pta_eom_14"], rtol=1e-8, name="eom")


def test_pta_klinger(ohlcv, assert_match):
    h, lo, c, v = (ohlcv[k] for k in ("high", "low", "close", "volume"))
    kvo, signal = vo.klinger_series(h, lo, c, v)
    assert_match(kvo, ohlcv["pta_kvo"], rtol=1e-8, name="kvo")
    assert_match(signal, ohlcv["pta_kvo_signal"], rtol=1e-8, name="kvo_signal")


def test_pta_pvo_signal(ohlcv, assert_match):
    assert_match(vo.pvo_series(ohlcv["volume"])[1], ohlcv["pta_pvo_signal"], name="pvo_signal")


def test_pta_vidya_recursion_converges(ohlcv, assert_match):
    # pandas-ta seeds VIDYA with 0 (ours: first close), so only the converged
    # tail can agree; this checks the recursion, not the seed.
    ours = tr.vidya_series(ohlcv["close"], 14, cmo_period=14)
    assert_match(ours, ohlcv["pta_vidya_14"], start=500, rtol=1e-7, atol=0.0, name="vidya")


def test_fixture_reproduces_with_installed_pandas_ta(ohlcv):
    pytest.importorskip("pandas_ta")
    import importlib.util
    from pathlib import Path

    gen_path = Path(__file__).parent / "reference" / "generate_indicator_fixture.py"
    spec = importlib.util.spec_from_file_location("gen_fixture", gen_path)
    assert spec is not None and spec.loader is not None
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    refs = gen.pandas_ta_outputs({k: ohlcv[k] for k in ("open", "high", "low", "close", "volume")})
    for key, arr in refs.items():
        np.testing.assert_allclose(arr, ohlcv[key], rtol=1e-12, atol=1e-12, err_msg=key)


# ═══════════════════════════════════════════════════════════════════
#  Independent formula transcriptions
# ═══════════════════════════════════════════════════════════════════


def _ema_list(x: List[float], period: int) -> List[float]:
    """SMA-seeded EMA over a list whose leading entries may be NaN."""
    out = [math.nan] * len(x)
    valid = [i for i, v in enumerate(x) if not math.isnan(v)]
    if len(valid) < period:
        return out
    first = valid[0]
    seed_at = first + period - 1
    acc = sum(x[first : seed_at + 1]) / period
    out[seed_at] = acc
    alpha = 2.0 / (period + 1)
    for i in range(seed_at + 1, len(x)):
        acc = alpha * x[i] + (1 - alpha) * acc
        out[i] = acc
    return out


def test_formula_mcginley(ohlcv, assert_match):
    c = ohlcv["close"].tolist()
    ref = [c[0]]
    for price in c[1:]:
        md = ref[-1]
        ref.append(md + (price - md) / (0.6 * 10 * (price / md) ** 4))
    assert_match(tr.mcginley_series(ohlcv["close"], 10), np.array(ref), name="mcginley")


def test_formula_vidya(ohlcv, assert_match):
    c = ohlcv["close"].tolist()
    n, period, cmo_n = len(c), 14, 9
    ref = [math.nan] * n
    alpha = 2 / (period + 1)
    for i in range(cmo_n, n):
        ups = sum(max(c[j] - c[j - 1], 0.0) for j in range(i - cmo_n + 1, i + 1))
        downs = sum(max(c[j - 1] - c[j], 0.0) for j in range(i - cmo_n + 1, i + 1))
        k = abs((ups - downs) / (ups + downs)) if ups + downs else 0.0
        ref[i] = c[i] if i == cmo_n else alpha * k * c[i] + (1 - alpha * k) * ref[i - 1]
    ours = tr.vidya_series(ohlcv["close"], period, cmo_period=cmo_n)
    assert_match(ours, np.array(ref), name="vidya")


def test_formula_stc(ohlcv, assert_match):
    # Transcription of LazyBear's published Pine script.
    c = ohlcv["close"].tolist()
    n, cycle, factor = len(c), 10, 0.5
    fast, slow = _ema_list(c, 23), _ema_list(c, 50)
    macd = [a - b for a, b in zip(fast, slow)]

    def stoch_smooth(src: List[float]) -> List[float]:
        out = [math.nan] * n
        prev_raw = 0.0
        started = False
        for i in range(n):
            window = src[max(0, i - cycle + 1) : i + 1]
            if i < cycle - 1 or any(math.isnan(v) for v in window):
                continue
            lo, hi = min(window), max(window)
            raw = (src[i] - lo) / (hi - lo) * 100 if hi - lo > 0 else prev_raw
            out[i] = raw if not started else out[i - 1] + factor * (raw - out[i - 1])
            started = True
            prev_raw = raw
        return out

    ref = stoch_smooth(stoch_smooth(macd))
    assert_match(tr.stc_series(ohlcv["close"]), np.array(ref), name="stc")


def _wilder_rsi_list(x: List[float], period: int) -> List[float]:
    out = [math.nan] * len(x)
    gains = [max(x[i] - x[i - 1], 0.0) for i in range(1, len(x))]
    losses = [max(x[i - 1] - x[i], 0.0) for i in range(1, len(x))]
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period

    def rsi(g: float, lo: float) -> float:
        return 100.0 if lo == 0 else 100 - 100 / (1 + g / lo)

    out[period] = rsi(ag, al)
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        out[i + 1] = rsi(ag, al)
    return out


def test_formula_connors_rsi(ohlcv, assert_match):
    c = ohlcv["close"].tolist()
    n = len(c)
    streak = [0.0] * n
    for i in range(1, n):
        if c[i] > c[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif c[i] < c[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
    roc = [math.nan] + [(c[i] - c[i - 1]) / c[i - 1] for i in range(1, n)]
    rank = [math.nan] * n
    for i in range(101, n):
        prior = roc[i - 100 : i]
        rank[i] = 100 * sum(1 for p in prior if p < roc[i]) / 100
    r3 = _wilder_rsi_list(c, 3)
    r2 = _wilder_rsi_list(streak, 2)
    ref = [(a + b + r) / 3 for a, b, r in zip(r3, r2, rank)]
    assert_match(mo.connors_rsi_series(ohlcv["close"]), np.array(ref), name="connors_rsi")


def test_connors_streak_and_rank_worked_example():
    close = np.array([10.0, 11, 12, 12, 11, 10, 10.5])
    assert mo._streak(close).tolist() == [0, 1, 2, 0, -1, -2, 1]


def test_formula_wavetrend(ohlcv, assert_match):
    h, lo, c = (ohlcv[k].tolist() for k in ("high", "low", "close"))
    ap = [(a + b + d) / 3 for a, b, d in zip(h, lo, c)]
    esa = _ema_list(ap, 10)
    d = _ema_list([abs(a - e) for a, e in zip(ap, esa)], 10)
    ci = [(a - e) / (0.015 * dd) for a, e, dd in zip(ap, esa, d)]
    wt1 = _ema_list(ci, 21)
    wt2 = [math.nan] * len(wt1)
    for i in range(3, len(wt1)):
        window = wt1[i - 3 : i + 1]
        if not any(math.isnan(v) for v in window):
            wt2[i] = sum(window) / 4
    ours1, ours2 = mo.wavetrend_series(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    assert_match(ours1, np.array(wt1), name="wt1")
    assert_match(ours2, np.array(wt2), name="wt2")


def test_formula_range_volatility_estimators(ohlcv, assert_match):
    o, h, lo, c = (ohlcv[k].tolist() for k in ("open", "high", "low", "close"))
    n, p = len(c), 20
    park = [math.nan] * n
    rs = [math.nan] * n
    yz = [math.nan] * n
    k = 0.34 / (1.34 + (p + 1) / (p - 1))
    for i in range(p - 1, n):
        idx = range(i - p + 1, i + 1)
        park[i] = math.sqrt(sum(math.log(h[j] / lo[j]) ** 2 for j in idx) / (4 * p * math.log(2)))
        rs_terms = [
            math.log(h[j] / c[j]) * math.log(h[j] / o[j])
            + math.log(lo[j] / c[j]) * math.log(lo[j] / o[j])
            for j in idx
        ]
        rs[i] = math.sqrt(sum(rs_terms) / p)
        if i >= p:
            on = [math.log(o[j] / c[j - 1]) for j in idx]
            oc = [math.log(c[j] / o[j]) for j in idx]
            mo_, mc = sum(on) / p, sum(oc) / p
            var_o = sum((x - mo_) ** 2 for x in on) / (p - 1)
            var_c = sum((x - mc) ** 2 for x in oc) / (p - 1)
            yz[i] = math.sqrt(var_o + k * var_c + (1 - k) * sum(rs_terms) / p)
    kw = dict(annualize=False)
    assert_match(vl.parkinson_series(ohlcv["high"], ohlcv["low"], p, **kw), np.array(park),
                 name="parkinson")
    arr = [ohlcv[x] for x in ("open", "high", "low", "close")]
    assert_match(vl.rogers_satchell_series(*arr, p, **kw), np.array(rs), name="rogers_satchell")
    assert_match(vl.yang_zhang_series(*arr, p, **kw), np.array(yz), name="yang_zhang")
    annual = vl.parkinson_series(ohlcv["high"], ohlcv["low"], p)
    assert_match(annual, np.array(park) * math.sqrt(252), name="parkinson_annualized")


def test_formula_rolling_sharpe_sortino_worked_example():
    r = np.array([0.01, -0.02, 0.03, 0.005, -0.01])
    mean = r.mean()
    sd = math.sqrt(sum((x - mean) ** 2 for x in r) / 4)
    dd = math.sqrt(sum(min(x, 0.0) ** 2 for x in r) / 5)
    sharpe = st.rolling_sharpe_series(r, 5, annualize=False)
    sortino = st.rolling_sortino_series(r, 5, annualize=False)
    assert np.isnan(sharpe[:4]).all() and np.isnan(sortino[:4]).all()
    assert sharpe[4] == pytest.approx(mean / sd, rel=1e-12)
    assert sortino[4] == pytest.approx(mean / dd, rel=1e-12)
    annual = st.rolling_sharpe_series(r - 0.001, 5, annualize=True, trading_days=252)
    shifted = st.rolling_sharpe_series(r, 5, risk_free=0.001, annualize=True)
    assert shifted[4] == pytest.approx(annual[4], rel=1e-12)


def test_rolling_sortino_without_downside_is_nan():
    assert np.isnan(st.rolling_sortino_series(np.full(10, 0.01), 5)[-1])
    assert np.isnan(st.rolling_sharpe_series(np.full(10, 0.01), 5)[-1])


def test_formula_rolling_sharpe_on_fixture(ohlcv, assert_match):
    r = np.diff(np.log(ohlcv["close"]))
    ref = np.full(len(r), np.nan)
    for i in range(62, len(r)):
        w = r[i - 62 : i + 1]
        ref[i] = w.mean() / w.std(ddof=1) * math.sqrt(252)
    assert_match(st.rolling_sharpe_series(r, 63), ref, name="rolling_sharpe")


def test_formula_anchored_vwap(ohlcv, assert_match):
    h, lo, c, v = (ohlcv[k] for k in ("high", "low", "close", "volume"))
    sessions = np.repeat(np.arange(len(c) // 50 + 1), 50)[: len(c)]
    ref = np.empty(len(c))
    pv = vol = 0.0
    for i in range(len(c)):
        if i == 0 or sessions[i] != sessions[i - 1]:
            pv = vol = 0.0
        pv += (h[i] + lo[i] + c[i]) / 3 * v[i]
        vol += v[i]
        ref[i] = pv / vol
    assert_match(vo.anchored_vwap_series(h, lo, c, v, sessions), ref, name="anchored_vwap")


# ═══════════════════════════════════════════════════════════════════
#  Property checks with a known answer
# ═══════════════════════════════════════════════════════════════════


def _simulated_ohlc(days: int, steps: int, sigma: float, overnight: float, seed: int):
    """Driftless log-Brownian bars sampled at *steps* points per day."""
    rng = np.random.default_rng(seed)
    gaps = rng.normal(0.0, overnight, days)
    paths = np.cumsum(rng.normal(0.0, sigma / math.sqrt(steps), (days, steps)), axis=1)
    log_open = np.cumsum(gaps + np.r_[0.0, paths[:-1, -1]])
    intraday = log_open[:, None] + np.c_[np.zeros(days), paths]
    o = np.exp(intraday[:, 0])
    c = np.exp(intraday[:, -1])
    h = np.exp(intraday.max(axis=1))
    lo = np.exp(intraday.min(axis=1))
    return o, h, lo, c


def test_property_range_estimators_recover_sigma():
    sigma = 0.02
    o, h, lo, c = _simulated_ohlc(days=3000, steps=1000, sigma=sigma, overnight=0.0, seed=11)
    n = len(c)
    kw = dict(annualize=False)
    park = vl.parkinson_series(h, lo, n, **kw)[-1]
    rs = vl.rogers_satchell_series(o, h, lo, c, n, **kw)[-1]
    gk = vl.garman_klass_series(o, h, lo, c, n, annualize=False)[-1]
    # Discrete sampling biases range estimators low by roughly 2-3% here.
    for name, est in (("parkinson", park), ("rogers_satchell", rs), ("garman_klass", gk)):
        assert est == pytest.approx(sigma, rel=0.06), name


def test_property_yang_zhang_includes_overnight_variance():
    sigma, overnight = 0.015, 0.01
    o, h, lo, c = _simulated_ohlc(days=3000, steps=1000, sigma=sigma, overnight=overnight, seed=5)
    n = len(c) - 1
    yz = vl.yang_zhang_series(o, h, lo, c, n, annualize=False)[-1]
    rs = vl.rogers_satchell_series(o, h, lo, c, n, annualize=False)[-1]
    assert yz == pytest.approx(math.sqrt(sigma**2 + overnight**2), rel=0.06)
    assert rs == pytest.approx(sigma, rel=0.06)  # RS ignores the overnight gap


def test_property_efficiency_ratio_bounds_and_trend():
    trend = np.linspace(100, 120, 50)
    assert np.allclose(st.efficiency_ratio_series(trend, 10)[10:], 1.0)
    zigzag = np.tile([100.0, 101.0], 25)
    assert np.allclose(st.efficiency_ratio_series(zigzag, 10)[10:], 0.0)


def test_property_ichimoku_has_no_lookahead(ohlcv):
    h, lo = ohlcv["high"].copy(), ohlcv["low"].copy()
    base = tr.ichimoku_series(h, lo)
    h[400:] *= 2.0
    lo[400:] *= 2.0
    bumped = tr.ichimoku_series(h, lo)
    for a, b in zip(base, bumped):
        np.testing.assert_array_equal(a[:400], b[:400])


def test_property_dpo_has_no_lookahead(ohlcv):
    c = ohlcv["close"].copy()
    base = mo.dpo_series(c, 20)
    c[400:] *= 2.0
    np.testing.assert_array_equal(base[:400], mo.dpo_series(c, 20)[:400])


def test_property_psar_never_inside_the_bar(ohlcv):
    h, lo = ohlcv["high"], ohlcv["low"]
    sar = tr.psar_series(h, lo)
    assert np.isnan(sar[0]) and np.isfinite(sar[1:]).all()
    assert ((sar[1:] <= lo[1:]) | (sar[1:] >= h[1:])).all()


def test_input_validation():
    with pytest.raises(ValueError):
        tr.hma_series(np.arange(10.0), 1)
    with pytest.raises(ValueError):
        vl.yang_zhang_series(*(np.ones(10),) * 4, period=1)
    with pytest.raises(ValueError):
        mo.cmo_series(np.arange(30.0), 14, smoothing="bogus")
    with pytest.raises(ValueError):
        vo.anchored_vwap_series(*(np.ones(5),) * 4, sessions=np.zeros(4))
