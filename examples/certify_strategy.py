"""Certification walkthrough: is your grid-search winner real or overfit?

Simulates the classic mistake: try 100 configurations of a noise strategy,
pick the best backtest, then let the certification math judge it.

Run:  python examples/certify_strategy.py
"""

from __future__ import annotations

import numpy as np

from cpz_quant.certification import (
    compute_risk_analytics,
    probability_of_backtest_overfitting,
)


def main() -> None:
    rng = np.random.default_rng(11)
    T, n_trials = 252, 100

    # 40 "strategies" that are pure noise: any winner is luck.
    trials = rng.normal(0.0, 0.01, size=(T, n_trials))

    best = int(np.argmax(trials.mean(axis=0) / trials.std(axis=0, ddof=1)))
    ann_sharpe = float(trials[:, best].mean() / trials[:, best].std(ddof=1) * np.sqrt(252))
    print(f"Best of {n_trials} trials: config #{best}, in-sample Sharpe {ann_sharpe:.2f}")
    print("Looks deployable... let's certify.\n")

    pbo = probability_of_backtest_overfitting(trials, n_splits=16)
    print(f"Probability of Backtest Overfitting: {pbo.pbo:.0%} "
          f"({pbo.n_combinations} IS/OOS combinations)")
    print(f"Mean OOS rank logit: {pbo.mean_logit:+.2f}  (< 0 leans overfit)")
    if pbo.performance_degradation is not None:
        print(f"IS->OOS performance slope: {pbo.performance_degradation:+.2f}")

    equity = 100_000 * np.cumprod(1 + trials[:, best])
    analytics = compute_risk_analytics(equity.tolist())
    if analytics is not None:
        print(f"\nDue-diligence block for the 'winner': "
              f"sortino={analytics.sortino}, max_dd and tail stats in full result")

    verdict = "OVERFIT - do not deploy" if pbo.pbo > 0.5 else "passes the PBO gate"
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()
