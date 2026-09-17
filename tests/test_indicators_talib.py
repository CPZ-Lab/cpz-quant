"""Numerical parity of cpz_quant.indicators against TA-Lib.

TA-Lib (``TA-Lib==0.8.0``, a dev dependency) is the reference. Every case
compares the full output on the frozen synthetic OHLCV fixture: identical
NaN warm-up and values within ``rtol = atol = 1e-9``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import cpz_quant.indicators.momentum as mo
import cpz_quant.indicators.statistical as st
import cpz_quant.indicators.trend as tr
import cpz_quant.indicators.volatility as vl
import cpz_quant.indicators.volume as vo
import numpy as np
import pytest

talib = pytest.importorskip("talib")

D = Dict[str, np.ndarray]
Case = Tuple[str, Callable[[D], np.ndarray], Callable[[D], np.ndarray]]


def _ohlcv(d: D):
    return d["open"], d["high"], d["low"], d["close"], d["volume"]


def _returns(x: np.ndarray) -> np.ndarray:
    return np.diff(x) / x[:-1]


CASES: List[Case] = [
    # ── Trend ────────────────────────────────────────────────────────
    ("TRIMA_30", lambda d: tr.trima_series(d["close"], 30), lambda d: talib.TRIMA(d["close"], 30)),
    ("TRIMA_10", lambda d: tr.trima_series(d["close"], 10), lambda d: talib.TRIMA(d["close"], 10)),
    ("TRIMA_11", lambda d: tr.trima_series(d["close"], 11), lambda d: talib.TRIMA(d["close"], 11)),
    ("T3_5", lambda d: tr.t3_series(d["close"], 5), lambda d: talib.T3(d["close"], 5, 0.7)),
    ("T3_8_v05", lambda d: tr.t3_series(d["close"], 8, vfactor=0.5),
     lambda d: talib.T3(d["close"], 8, 0.5)),
    ("ZLEMA_30", lambda d: tr.zlema_series(d["close"], 30), lambda d: talib.ZLEMA(d["close"], 30)),
    ("ZLEMA_9", lambda d: tr.zlema_series(d["close"], 9), lambda d: talib.ZLEMA(d["close"], 9)),
    ("HMA_20", lambda d: tr.hma_series(d["close"], 20), lambda d: talib.HMA(d["close"], 20)),
    ("HMA_9", lambda d: tr.hma_series(d["close"], 9), lambda d: talib.HMA(d["close"], 9)),
    ("SAR", lambda d: tr.psar_series(d["high"], d["low"]), lambda d: talib.SAR(d["high"], d["low"])),
    ("SAR_03_03", lambda d: tr.psar_series(d["high"], d["low"], acceleration=0.03, maximum=0.3),
     lambda d: talib.SAR(d["high"], d["low"], 0.03, 0.3)),
    ("AROON_down", lambda d: tr.aroon_series(d["high"], d["low"], 14)[0],
     lambda d: talib.AROON(d["high"], d["low"], 14)[0]),
    ("AROON_up", lambda d: tr.aroon_series(d["high"], d["low"], 14)[1],
     lambda d: talib.AROON(d["high"], d["low"], 14)[1]),
    ("AROONOSC", lambda d: tr.aroon_osc_series(d["high"], d["low"], 25),
     lambda d: talib.AROONOSC(d["high"], d["low"], 25)),
    ("VORTEX_plus", lambda d: tr.vortex_series(d["high"], d["low"], d["close"], 14)[0],
     lambda d: talib.VORTEX(d["high"], d["low"], d["close"], 14)[0]),
    ("VORTEX_minus", lambda d: tr.vortex_series(d["high"], d["low"], d["close"], 14)[1],
     lambda d: talib.VORTEX(d["high"], d["low"], d["close"], 14)[1]),
    ("TRIX_30", lambda d: tr.trix_series(d["close"], 30), lambda d: talib.TRIX(d["close"], 30)),
    ("TRIX_5", lambda d: tr.trix_series(d["close"], 5), lambda d: talib.TRIX(d["close"], 5)),
    ("MASSI", lambda d: tr.mass_index_series(d["high"], d["low"]),
     lambda d: talib.MASSI(d["high"], d["low"], 9, 25)),
    ("MIDPOINT", lambda d: tr.midpoint_series(d["close"], 14),
     lambda d: talib.MIDPOINT(d["close"], 14)),
    ("MIDPRICE", lambda d: tr.midprice_series(d["high"], d["low"], 14),
     lambda d: talib.MIDPRICE(d["high"], d["low"], 14)),
    # ── Momentum ─────────────────────────────────────────────────────
    ("STOCH_slowk", lambda d: mo.stochastic_slow_series(d["high"], d["low"], d["close"])[0],
     lambda d: talib.STOCH(d["high"], d["low"], d["close"])[0]),
    ("STOCH_slowd", lambda d: mo.stochastic_slow_series(d["high"], d["low"], d["close"])[1],
     lambda d: talib.STOCH(d["high"], d["low"], d["close"])[1]),
    ("STOCH_14_3_3_slowd",
     lambda d: mo.stochastic_slow_series(d["high"], d["low"], d["close"], 14, 3, 3)[1],
     lambda d: talib.STOCH(d["high"], d["low"], d["close"], 14, 3, 0, 3, 0)[1]),
    ("STOCHRSI_k", lambda d: mo.stochrsi_series(d["close"])[0],
     lambda d: talib.STOCHRSI(d["close"])[0]),
    ("STOCHRSI_d", lambda d: mo.stochrsi_series(d["close"])[1],
     lambda d: talib.STOCHRSI(d["close"])[1]),
    ("STOCHRSI_14_14_3_d", lambda d: mo.stochrsi_series(d["close"], 14, 14, 3)[1],
     lambda d: talib.STOCHRSI(d["close"], 14, 14, 3)[1]),
    ("CMO_wilder", lambda d: mo.cmo_series(d["close"], 14, smoothing="wilder"),
     lambda d: talib.CMO(d["close"], 14)),
    ("CMOU_sum", lambda d: mo.cmo_series(d["close"], 14), lambda d: talib.CMOU(d["close"], 14)),
    ("MOM", lambda d: mo.mom_series(d["close"], 10), lambda d: talib.MOM(d["close"], 10)),
    ("ROCP", lambda d: mo.rocp_series(d["close"], 10), lambda d: talib.ROCP(d["close"], 10)),
    ("ROCR", lambda d: mo.rocr_series(d["close"], 10), lambda d: talib.ROCR(d["close"], 10)),
    ("ROCR100", lambda d: mo.rocr_series(d["close"], 10, scale=100.0),
     lambda d: talib.ROCR100(d["close"], 10)),
    ("ULTOSC", lambda d: mo.ultosc_series(d["high"], d["low"], d["close"]),
     lambda d: talib.ULTOSC(d["high"], d["low"], d["close"])),
    ("ULTOSC_5_10_20", lambda d: mo.ultosc_series(d["high"], d["low"], d["close"], 5, 10, 20),
     lambda d: talib.ULTOSC(d["high"], d["low"], d["close"], 5, 10, 20)),
    ("AO", lambda d: mo.awesome_oscillator_series(d["high"], d["low"]),
     lambda d: talib.AO(d["high"], d["low"], 5, 34)),
    ("APO_ema", lambda d: mo.apo_series(d["close"], 12, 26, ma_type="ema"),
     lambda d: talib.APO(d["close"], 12, 26, 1)),
    ("APO_sma", lambda d: mo.apo_series(d["close"], 12, 26, ma_type="sma"),
     lambda d: talib.APO(d["close"], 12, 26, 0)),
    ("PPO_ema", lambda d: mo.ppo_series(d["close"], 12, 26, ma_type="ema"),
     lambda d: talib.PPO(d["close"], 12, 26, 1)),
    ("PPO_sma", lambda d: mo.ppo_series(d["close"], 12, 26, ma_type="sma"),
     lambda d: talib.PPO(d["close"], 12, 26, 0)),
    ("TSI", lambda d: mo.tsi_series(d["close"], 25, 13)[0],
     lambda d: talib.TSI(d["close"], 25, 13)),
    ("COPPOCK", lambda d: mo.coppock_series(d["close"]),
     lambda d: talib.COPPOCK(d["close"], 10, 11, 14)),
    ("DPO_20", lambda d: mo.dpo_series(d["close"], 20), lambda d: talib.DPO(d["close"], 20)),
    ("DPO_15", lambda d: mo.dpo_series(d["close"], 15), lambda d: talib.DPO(d["close"], 15)),
    ("ERI_bull", lambda d: mo.elder_ray_series(d["high"], d["low"], d["close"], 13)[0],
     lambda d: talib.ERI(d["high"], d["low"], d["close"], 13)[0]),
    ("ERI_bear", lambda d: mo.elder_ray_series(d["high"], d["low"], d["close"], 13)[1],
     lambda d: talib.ERI(d["high"], d["low"], d["close"], 13)[1]),
    ("BOP", lambda d: mo.bop_series(*_ohlcv(d)[:4]),
     lambda d: talib.BOP(*_ohlcv(d)[:4])),
    ("PLUS_DM", lambda d: mo.dmi_components_series(d["high"], d["low"], d["close"], 14)[0],
     lambda d: talib.PLUS_DM(d["high"], d["low"], 14)),
    ("MINUS_DM", lambda d: mo.dmi_components_series(d["high"], d["low"], d["close"], 14)[1],
     lambda d: talib.MINUS_DM(d["high"], d["low"], 14)),
    ("PLUS_DI", lambda d: mo.dmi_components_series(d["high"], d["low"], d["close"], 14)[2],
     lambda d: talib.PLUS_DI(d["high"], d["low"], d["close"], 14)),
    ("MINUS_DI", lambda d: mo.dmi_components_series(d["high"], d["low"], d["close"], 14)[3],
     lambda d: talib.MINUS_DI(d["high"], d["low"], d["close"], 14)),
    ("DX", lambda d: mo.dmi_components_series(d["high"], d["low"], d["close"], 14)[4],
     lambda d: talib.DX(d["high"], d["low"], d["close"], 14)),
    ("ADX", lambda d: mo.adx_talib_series(d["high"], d["low"], d["close"], 14)[0],
     lambda d: talib.ADX(d["high"], d["low"], d["close"], 14)),
    ("ADXR", lambda d: mo.adx_talib_series(d["high"], d["low"], d["close"], 14)[1],
     lambda d: talib.ADXR(d["high"], d["low"], d["close"], 14)),
    ("ADXR_7", lambda d: mo.adx_talib_series(d["high"], d["low"], d["close"], 7)[1],
     lambda d: talib.ADXR(d["high"], d["low"], d["close"], 7)),
    # ── Volatility ───────────────────────────────────────────────────
    ("CVI", lambda d: vl.chaikin_volatility_series(d["high"], d["low"], 10, 10),
     lambda d: talib.CVI(d["high"], d["low"], 10, 10)),
    # ── Volume ───────────────────────────────────────────────────────
    ("AD", lambda d: vo.ad_series(*_ohlcv(d)[1:]), lambda d: talib.AD(*_ohlcv(d)[1:])),
    ("ADOSC", lambda d: vo.adosc_series(*_ohlcv(d)[1:]), lambda d: talib.ADOSC(*_ohlcv(d)[1:])),
    ("ADOSC_5_20", lambda d: vo.adosc_series(*_ohlcv(d)[1:], 5, 20),
     lambda d: talib.ADOSC(*_ohlcv(d)[1:], 5, 20)),
    ("VWMA", lambda d: vo.vwma_series(d["close"], d["volume"], 20),
     lambda d: talib.VWMA(d["close"], d["volume"], 20)),
    ("EFI", lambda d: vo.force_index_series(d["close"], d["volume"], 13),
     lambda d: talib.EFI(d["close"], d["volume"], 13)),
    ("NVI", lambda d: vo.nvi_series(d["close"], d["volume"]),
     lambda d: talib.NVI(d["close"], d["volume"])),
    ("PVI", lambda d: vo.pvi_series(d["close"], d["volume"]),
     lambda d: talib.PVI(d["close"], d["volume"])),
    ("PVT", lambda d: vo.pvt_series(d["close"], d["volume"]),
     lambda d: talib.PVT(d["close"], d["volume"])),
    ("PVO", lambda d: vo.pvo_series(d["volume"], 12, 26)[0],
     lambda d: talib.PVO(d["volume"], 12, 26, 1)),
    ("VWAP_cumulative", lambda d: vo.vwap_series(*_ohlcv(d)[1:]),
     lambda d: talib.VWAP(*_ohlcv(d)[1:])),
    ("VWAP_single_session", lambda d: vo.anchored_vwap_series(
        *_ohlcv(d)[1:], np.zeros(len(d["close"]))),
     lambda d: talib.VWAP(*_ohlcv(d)[1:])),
    # ── Statistical ──────────────────────────────────────────────────
    ("LINEARREG", lambda d: st.linear_reg_value_series(d["close"], 14),
     lambda d: talib.LINEARREG(d["close"], 14)),
    ("LINEARREG_ANGLE", lambda d: st.linear_reg_angle_series(d["close"], 14),
     lambda d: talib.LINEARREG_ANGLE(d["close"], 14)),
    ("ER", lambda d: st.efficiency_ratio_series(d["close"], 10),
     lambda d: talib.ER(d["close"], 10)),
]

# Indicators that predate this suite and already match TA-Lib exactly.
EXISTING_EXACT: List[Case] = [
    ("SMA", lambda d: tr.sma_series(d["close"], 20), lambda d: talib.SMA(d["close"], 20)),
    ("WMA", lambda d: tr.wma_series(d["close"], 20), lambda d: talib.WMA(d["close"], 20)),
    ("RSI", lambda d: mo.rsi_series(d["close"], 14), lambda d: talib.RSI(d["close"], 14)),
    ("WILLR", lambda d: mo.williams_r_series(d["high"], d["low"], d["close"], 14),
     lambda d: talib.WILLR(d["high"], d["low"], d["close"], 14)),
    ("CCI", lambda d: mo.cci_series(d["high"], d["low"], d["close"], 20),
     lambda d: talib.CCI(d["high"], d["low"], d["close"], 20)),
    ("ROC", lambda d: mo.roc_series(d["close"], 12), lambda d: talib.ROC(d["close"], 12)),
    ("MFI", lambda d: mo.mfi_series(d["high"], d["low"], d["close"], d["volume"], 14),
     lambda d: talib.MFI(d["high"], d["low"], d["close"], d["volume"], 14)),
    ("BBANDS_upper", lambda d: vl.bollinger_series(d["close"], 20)[0],
     lambda d: talib.BBANDS(d["close"], 20, 2, 2)[0]),
    ("BBANDS_lower", lambda d: vl.bollinger_series(d["close"], 20)[2],
     lambda d: talib.BBANDS(d["close"], 20, 2, 2)[2]),
    ("LINEARREG_SLOPE", lambda d: st.linear_reg_series(d["close"], 20)[0],
     lambda d: talib.LINEARREG_SLOPE(d["close"], 20)),
    ("LINEARREG_INTERCEPT", lambda d: st.linear_reg_series(d["close"], 20)[2],
     lambda d: talib.LINEARREG_INTERCEPT(d["close"], 20)),
    ("TSF", lambda d: st.linear_reg_series(d["close"], 20)[3],
     lambda d: talib.TSF(d["close"], 20)),
    ("CORREL", lambda d: st.rolling_corr_series(d["close"], d["open"], 30),
     lambda d: talib.CORREL(d["close"], d["open"], 30)),
    # TA-Lib BETA(price0, price1) regresses price1's simple returns on price0's.
    ("BETA", lambda d: st.rolling_beta_series(_returns(d["open"]), _returns(d["close"]), 5),
     lambda d: talib.BETA(d["close"], d["open"], 5)[1:]),
]


@pytest.mark.parametrize("name,ours,ref", CASES + EXISTING_EXACT, ids=lambda x: x
                         if isinstance(x, str) else "")
def test_matches_talib(name, ours, ref, ohlcv, assert_match):
    assert_match(ours(ohlcv), ref(ohlcv), name=name)


# Older indicators that seed their smoothing differently from TA-Lib. They
# are not changed here (that would silently alter published behaviour), but
# they must converge to TA-Lib once the seed has decayed.
CONVERGING: List[Tuple[str, Callable[[D], np.ndarray], Callable[[D], np.ndarray], int]] = [
    ("EMA", lambda d: tr.ema_series(d["close"], 20), lambda d: talib.EMA(d["close"], 20), 300),
    ("DEMA", lambda d: tr.dema_series(d["close"], 20), lambda d: talib.DEMA(d["close"], 20), 400),
    ("TEMA", lambda d: tr.tema_series(d["close"], 20), lambda d: talib.TEMA(d["close"], 20), 450),
    ("MACD", lambda d: mo.macd_series(d["close"])[0], lambda d: talib.MACD(d["close"])[0], 450),
    ("ATR", lambda d: vl.atr_series(d["high"], d["low"], d["close"], 14),
     lambda d: talib.ATR(d["high"], d["low"], d["close"], 14), 450),
    ("ADX_legacy", lambda d: mo.adx_series(d["high"], d["low"], d["close"], 14),
     lambda d: talib.ADX(d["high"], d["low"], d["close"], 14), 450),
]


@pytest.mark.parametrize("name,ours,ref,start", CONVERGING, ids=[c[0] for c in CONVERGING])
def test_legacy_seeds_converge_to_talib(name, ours, ref, start, ohlcv, assert_match):
    assert_match(ours(ohlcv), ref(ohlcv), start=start, rtol=1e-8, atol=1e-8, name=name)
