"""Compare five allocators on the same universe, then validate out of sample.

Self-contained: uses synthetic returns so it runs without any data vendor.
Swap `make_returns()` for your own `{asset: [daily_returns]}` dict.

Run:  python examples/portfolio_optimization.py
"""

from __future__ import annotations

import numpy as np

from cpz_quant.portfolio import (
    Constraints,
    WalkForward,
    cross_validate,
    equal_weight,
    hierarchical_risk_parity,
    max_sharpe,
    mean_cvar,
    risk_parity,
)


def make_returns(n_assets: int = 8, T: int = 756, seed: int = 42) -> dict:
    """Three years of synthetic daily returns with a common market factor."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0003, 0.009, T)
    betas = rng.uniform(0.3, 1.4, n_assets)
    alphas = rng.normal(0.0001, 0.0002, n_assets)
    out = {}
    for i in range(n_assets):
        idio = rng.normal(0, 0.008, T)
        out[f"ASSET_{i}"] = (alphas[i] + betas[i] * market + idio).tolist()
    return out


def main() -> None:
    returns = make_returns()
    constraints = Constraints(long_only=True, max_weight=0.35)

    allocators = {
        # equal_weight takes the universe (ids), not return series
        "equal_weight": lambda r: equal_weight(list(r)),
        "max_sharpe": lambda r: max_sharpe(r, constraints=constraints),
        "risk_parity": lambda r: risk_parity(r),
        "hrp": lambda r: hierarchical_risk_parity(r),
        "mean_cvar_95": lambda r: mean_cvar(r, confidence=0.95, constraints=constraints),
    }

    print(f"{'method':<14} {'ret%':>7} {'vol%':>7} {'sharpe':>7}   in-sample")
    for name, fn in allocators.items():
        res = fn(returns)
        if name == "equal_weight":
            print(f"{name:<14} {'-':>7} {'-':>7} {'-':>7}   (baseline, no estimate)")
        else:
            print(f"{name:<14} {res.expected_return:>7.2f} {res.volatility:>7.2f} "
                  f"{res.sharpe_ratio:>7.2f}")

    print(f"\n{'method':<14} {'oos_sharpe':>10} {'stability':>10}   walk-forward (4 folds)")
    cv = WalkForward(n_splits=4, test_size=63)
    for name, fn in allocators.items():
        result = cross_validate(lambda train, f=fn: f(train).weights, returns, cv)
        print(f"{name:<14} {result.oos_sharpe:>10.2f} {result.stability():>10.0%}")


if __name__ == "__main__":
    main()
