"""Visualization: four institutional figures, Plotly-native, zero required deps.

Optional — install with ``pip install cpz-quant[viz]``. Each function
returns a ``plotly.graph_objects.Figure`` you can ``.show()``, embed, or
export; cpz-quant never renders anything itself and never imports plotly
unless you call into this module.

Deliberately small: weights, efficient frontier, equity/drawdown, and a
cluster-ordered correlation matrix. Anything fancier belongs in your own
notebook (or the CPZAI operating system, where the full tearsheets live).
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

from cpz_quant.frames import as_returns


def _require_plotly() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "Visualization requires plotly. Install it with: pip install cpz-quant[viz]"
        ) from None
    return go


_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=13),
    margin=dict(l=60, r=30, t=60, b=50),
)


def plot_weights(weights: Mapping[str, float], *, title: str = "Portfolio weights") -> Any:
    """Horizontal bar chart of portfolio weights, largest first."""
    go = _require_plotly()
    items = sorted(weights.items(), key=lambda kv: kv[1])
    fig = go.Figure(
        go.Bar(
            x=[w for _, w in items],
            y=[k for k, _ in items],
            orientation="h",
            marker_color=["#1f77b4" if w >= 0 else "#d62728" for _, w in items],
        )
    )
    fig.update_layout(title=title, xaxis_title="weight", xaxis_tickformat=".1%", **_LAYOUT)
    return fig


def plot_frontier(
    returns: Any,
    *,
    n_points: int = 25,
    risk_free_rate: float = 0.0,
    title: str = "Efficient frontier",
) -> Any:
    """Efficient frontier traced with the native mean-variance optimizer.

    Accepts a returns dict or a Polars/pandas DataFrame. Marks the
    max-Sharpe and min-variance portfolios.
    """
    go = _require_plotly()
    from cpz_quant.portfolio import Constraints, max_sharpe, mean_variance, min_variance

    data = as_returns(returns)
    cons = Constraints(long_only=True)
    lo = min_variance(data, constraints=cons)
    hi = max_sharpe(data, risk_free_rate=risk_free_rate, constraints=cons)
    lo_ret, hi_ret = lo.expected_return / 100, hi.expected_return / 100
    span = max(hi_ret - lo_ret, 1e-6)
    vols: List[float] = []
    rets: List[float] = []
    for t in np.linspace(lo_ret, lo_ret + 1.6 * span, n_points):
        try:
            p = mean_variance(data, target_return=float(t), constraints=cons)
        except Exception:
            continue
        vols.append(p.volatility / 100)
        rets.append(p.expected_return / 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vols, y=rets, mode="lines", name="frontier",
                             line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=[lo.volatility / 100], y=[lo_ret], mode="markers",
                             name="min variance", marker=dict(size=11, color="#2ca02c")))
    fig.add_trace(go.Scatter(x=[hi.volatility / 100], y=[hi.expected_return / 100],
                             mode="markers", name="max Sharpe",
                             marker=dict(size=11, color="#d62728", symbol="star")))
    fig.update_layout(title=title, xaxis_title="annualised volatility",
                      yaxis_title="annualised return", xaxis_tickformat=".1%",
                      yaxis_tickformat=".1%", **_LAYOUT)
    return fig


def plot_drawdown(
    equity: Sequence[float],
    *,
    dates: Optional[Sequence[str]] = None,
    title: str = "Equity and drawdown",
) -> Any:
    """Two-panel figure: equity curve on top, underwater drawdown below."""
    go = _require_plotly()
    from plotly.subplots import make_subplots

    eq = np.asarray(list(equity), dtype=float)
    if eq.size < 2:
        raise ValueError("equity needs at least 2 points")
    x: Sequence[Any] = list(dates) if dates is not None else list(range(eq.size))
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=x, y=eq, name="equity",
                             line=dict(color="#1f77b4", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=dd, name="drawdown", fill="tozeroy",
                             line=dict(color="#d62728", width=1)), row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_layout(title=title, showlegend=False, **_LAYOUT)
    return fig


def plot_corr_clusters(
    returns: Any,
    *,
    title: str = "Correlation (cluster-ordered)",
) -> Any:
    """Correlation heatmap with assets ordered by hierarchical clustering.

    The same quasi-diagonalisation HRP uses, so blocks of co-moving
    assets are visible instead of scattered.
    """
    go = _require_plotly()
    from scipy.cluster.hierarchy import leaves_list, linkage

    data = as_returns(returns)
    ids = list(data)
    min_len = min(len(data[i]) for i in ids)
    R = np.column_stack([np.asarray(data[i][-min_len:], dtype=float) for i in ids])
    corr = np.corrcoef(R.T)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    order = leaves_list(linkage(dist[np.triu_indices_from(dist, k=1)], method="single"))
    labels = [ids[i] for i in order]
    fig = go.Figure(
        go.Heatmap(z=corr[np.ix_(order, order)], x=labels, y=labels,
                   colorscale="RdBu", zmid=0, zmin=-1, zmax=1)
    )
    fig.update_layout(title=title, yaxis_autorange="reversed", **_LAYOUT)
    return fig


__all__ = ["plot_weights", "plot_frontier", "plot_drawdown", "plot_corr_clusters"]
