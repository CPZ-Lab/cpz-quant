"""Regenerate ``tests/fixtures/allocator_reference.npz``.

Freezes seeded ETF-like daily returns plus the outputs of established
libraries on them, so the allocator and covariance tests compare against
published implementations without making those libraries dev dependencies:

- PyPortfolioOpt ``HRPOpt``: Lopez de Prado (2016) hierarchical risk parity.
- riskfolio-lib ``HCPortfolio(model="HERC")``: Raffinot (2018) HERC.
- riskfolio-lib ``Portfolio.rp_optimization``: equal risk contribution.
- PyPortfolioOpt Ledoit-Wolf single-factor shrinkage (``covMarket.m``).
- riskfolio-lib ``gerber_cov_stat1``: Gerber et al. (2021) statistic.

Run from the repo root in a scratch environment:

    pip install "PyPortfolioOpt==1.6.0" "riskfolio-lib==7.4.1" pandas
    python tests/reference/generate_allocator_fixture.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "allocator_reference.npz"
NAMES = ["EQ1", "EQ2", "EQ3", "BD1", "BD2", "GLD", "CMD", "REIT"]
RP_NAMES = ["EQ1", "EQ2", "BD1", "BD2", "GLD"]
GERBER_NAMES = ["EQ1", "EQ2", "EQ3", "BD1", "GLD", "CMD", "REIT"]
N_DAYS = 600
SEED = 20261125


def synthetic_returns(n: int = N_DAYS, seed: int = SEED) -> np.ndarray:
    """Equity, rates, gold and commodity factors; bonds load negatively on equity."""
    rng = np.random.default_rng(seed)
    eq, rates, gold, cmdty = (rng.normal(0, s, n) for s in (0.010, 0.005, 0.008, 0.011))
    cols = [
        1.00 * eq + rng.normal(0, 0.003, n) + 4e-4,
        1.20 * eq + rng.normal(0, 0.005, n) + 5e-4,
        0.80 * eq + rng.normal(0, 0.006, n) + 2e-4,
        -0.25 * eq + 1.0 * rates + rng.normal(0, 0.002, n) + 1e-4,
        -0.35 * eq + 1.6 * rates + rng.normal(0, 0.003, n) + 1.5e-4,
        0.05 * eq + 0.3 * rates + gold + 2e-4,
        0.40 * eq + cmdty + 1e-4,
        0.70 * eq + 0.8 * rates + rng.normal(0, 0.007, n) + 3e-4,
    ]
    return np.column_stack(cols)


def references(R: np.ndarray) -> Dict[str, np.ndarray]:
    import pandas as pd
    import riskfolio as rp
    from pypfopt import HRPOpt
    from pypfopt.risk_models import CovarianceShrinkage
    from riskfolio.src.GerberStatistic import gerber_cov_stat1

    df = pd.DataFrame(R, columns=NAMES)
    out: Dict[str, np.ndarray] = {}
    for method in ("single", "ward"):
        w = HRPOpt(df).optimize(linkage_method=method)
        out[f"hrp_{method}"] = np.array([w[k] for k in NAMES])
    for k in (2, 3, 4, 5):
        port = rp.HCPortfolio(returns=df)
        w = port.optimization(model="HERC", codependence="pearson", rm="MV", rf=0,
                              linkage="ward", k=k, leaf_order=False)
        out[f"herc_ward_k{k}"] = w.loc[NAMES, "weights"].to_numpy()
    port = rp.Portfolio(returns=df[RP_NAMES])
    port.assets_stats(method_mu="hist", method_cov="hist")
    w = port.rp_optimization(model="Classic", rm="MV", rf=0, b=None, hist=True)
    out["risk_parity"] = w.loc[RP_NAMES, "weights"].to_numpy()
    _, delta = CovarianceShrinkage(df, returns_data=True, frequency=1)._ledoit_wolf_single_factor()
    out["lw_single_factor_delta"] = np.array([delta])
    gerber_window = df[GERBER_NAMES].iloc[:90].to_numpy()
    out["gerber_stat1_90d"] = np.asarray(gerber_cov_stat1(gerber_window, 0.5))
    return out


def main() -> None:
    R = synthetic_returns()
    np.savez_compressed(FIXTURE, returns=R, names=np.array(NAMES), **references(R))
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main()
