"""Regenerate ``tests/fixtures/indicator_reference.npz``.

The fixture freezes two things so the indicator tests are reproducible
without optional reference libraries:

1. A seeded synthetic OHLCV series (stored, not regenerated at test time,
   so the tests do not depend on NumPy's random stream staying stable).
2. Outputs of pandas-ta for indicators that TA-Lib does not provide.
   pandas-ta pins an old NumPy through numba, so it is deliberately not a
   dev dependency; TA-Lib comparisons run live in the test suite instead.

Run from the repo root with a scratch environment that has pandas-ta:

    pip install "pandas-ta==0.4.71b0" pandas
    python tests/reference/generate_indicator_fixture.py

Only indicators whose pandas-ta definition matches the documented
cpz-quant definition are frozen here; see the notes next to each entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "indicator_reference.npz"
N_BARS = 600
SEED = 20260917


def synthetic_ohlcv(n: int = N_BARS, seed: int = SEED) -> Dict[str, np.ndarray]:
    """Random-walk OHLCV rounded to cents, so ties in price occur."""
    rng = np.random.default_rng(seed)
    close = np.round(100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.015, n)), 2)
    prev_close = np.r_[close[0], close[:-1]]
    open_ = np.round(prev_close * (1.0 + rng.normal(0.0, 0.004, n)), 2)
    high = np.round(np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.006, n))), 2)
    low = np.round(np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.006, n))), 2)
    volume = (rng.integers(10, 1000, n) * 1000).astype(np.float64)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def pandas_ta_outputs(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    import pandas as pd
    import pandas_ta as ta

    o, h, lo, c, v = (pd.Series(data[k]) for k in ("open", "high", "low", "close", "volume"))
    out: Dict[str, np.ndarray] = {}
    out["pta_alma_9"] = ta.alma(c, 9).to_numpy()
    ichi = ta.ichimoku(h, lo, c, include_chikou=False)[0]
    out["pta_ichimoku_span_a"] = ichi["ISA_9"].to_numpy()
    out["pta_ichimoku_span_b"] = ichi["ISB_26"].to_numpy()
    out["pta_ichimoku_tenkan"] = ichi["ITS_9"].to_numpy()
    out["pta_ichimoku_kijun"] = ichi["IKS_26"].to_numpy()
    # pandas-ta scales KST by an extra factor of 100.
    kst = ta.kst(c)
    out["pta_kst_x100"] = kst.iloc[:, 0].to_numpy()
    out["pta_kst_signal_x100"] = kst.iloc[:, 1].to_numpy()
    out["pta_tsi_signal"] = ta.tsi(c).iloc[:, 1].to_numpy()
    # pandas-ta starts the Fisher recursion one bar later; compare the tail.
    fisher = ta.fisher(h, lo, 9)
    out["pta_fisher"] = fisher.iloc[:, 0].to_numpy()
    out["pta_fisher_trigger"] = fisher.iloc[:, 1].to_numpy()
    rvgi = ta.rvgi(o, h, lo, c, 14)
    out["pta_rvgi_14"] = rvgi.iloc[:, 0].to_numpy()
    out["pta_rvgi_signal_14"] = rvgi.iloc[:, 1].to_numpy()
    out["pta_ulcer_14"] = ta.ui(c, 14).to_numpy()
    out["pta_chop_14"] = ta.chop(h, lo, c, 14).to_numpy()
    out["pta_eom_14"] = ta.eom(h, lo, c, v, 14).to_numpy()
    kvo = ta.kvo(h, lo, c, v)
    out["pta_kvo"] = kvo.iloc[:, 0].to_numpy()
    out["pta_kvo_signal"] = kvo.iloc[:, 1].to_numpy()
    out["pta_pvo_signal"] = ta.pvo(v).iloc[:, 2].to_numpy()
    # pandas-ta seeds VIDYA with zero; the recursion converges, compare the tail.
    out["pta_vidya_14"] = ta.vidya(c, 14, talib=False).to_numpy()
    return {k: np.asarray(val, dtype=np.float64) for k, val in out.items()}


def main() -> None:
    import pandas_ta

    data = synthetic_ohlcv()
    refs = pandas_ta_outputs(data)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FIXTURE,
        pandas_ta_version=np.array(pandas_ta.version),
        **data,
        **refs,
    )
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes, {len(refs)} reference arrays)")


if __name__ == "__main__":
    main()
