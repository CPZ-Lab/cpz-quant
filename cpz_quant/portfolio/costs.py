"""Transaction cost models: Almgren-Chriss, linear/sqrt impact, spread, turnover analysis.

All functions are pure: data in, results out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

EPSILON: float = 1e-15


@dataclass
class CostEstimate:
    """Breakdown of transaction cost for a single trade."""
    permanent_cost_bps: float = 0.0
    temporary_cost_bps: float = 0.0
    spread_cost_bps: float = 0.0
    commission_bps: float = 0.0
    total_cost_bps: float = 0.0


@dataclass
class TurnoverResult:
    """Portfolio turnover analysis."""
    one_way_turnover: float = 0.0
    two_way_turnover: float = 0.0
    estimated_cost_bps: float = 0.0
    per_asset_turnover: Dict[str, float] = field(default_factory=dict)
    per_asset_cost_bps: Dict[str, float] = field(default_factory=dict)


@dataclass
class CostModel:
    """Pluggable cost model for portfolio optimisation.

    All parameters are configurable so users can calibrate to their
    broker and market conditions.
    """
    spread_bps: float = 5.0
    impact_coefficient: float = 0.1
    impact_exponent: float = 0.5
    impact_model: str = "sqrt"
    commission_bps: float = 0.0
    permanent_fraction: float = 0.5

    def estimate(
        self,
        notional: float,
        adv: float = 0.0,
        volatility: float = 0.0,
    ) -> float:
        """Estimate round-trip cost in basis points.

        Args:
            notional: Trade notional value.
            adv: Average daily volume (notional).
            volatility: Daily volatility of the asset.
        """
        spread = self.spread_bps / 2.0

        if self.impact_model == "sqrt":
            impact = sqrt_impact(notional, adv, coefficient=self.impact_coefficient,
                                 volatility=volatility, exponent=self.impact_exponent)
        elif self.impact_model == "linear":
            impact = linear_impact(notional, adv, coefficient=self.impact_coefficient)
        elif self.impact_model == "almgren_chriss":
            result = almgren_chriss(notional, adv, volatility)
            impact = result.permanent_cost_bps + result.temporary_cost_bps
        else:
            impact = 0.0

        return spread + impact + self.commission_bps


def almgren_chriss(
    notional: float,
    adv: float,
    volatility: float,
    *,
    gamma: float = 0.314,
    eta: float = 0.142,
    alpha: float = 0.6,
    beta: float = 0.6,
    participation_rate: float = 0.1,
) -> CostEstimate:
    """Almgren-Chriss (2001) market impact model.

    permanent_impact = gamma * sigma * (Q/V)^alpha
    temporary_impact = eta * sigma * (Q/(V*T))^beta

    Args:
        notional: Trade notional.
        adv: Average daily volume (notional).
        volatility: Daily price volatility.
        gamma: Permanent impact coefficient.
        eta: Temporary impact coefficient.
        alpha: Permanent impact exponent.
        beta: Temporary impact exponent.
        participation_rate: Fraction of ADV consumed per time unit.
    """
    if adv < EPSILON or volatility < EPSILON:
        return CostEstimate()

    q_over_v = notional / adv
    perm = gamma * volatility * (q_over_v ** alpha) * 10000.0

    execution_time = q_over_v / max(participation_rate, EPSILON)
    temp = eta * volatility * ((q_over_v / max(execution_time, EPSILON)) ** beta) * 10000.0

    return CostEstimate(
        permanent_cost_bps=round(perm, 4),
        temporary_cost_bps=round(temp, 4),
        total_cost_bps=round(perm + temp, 4),
    )


def linear_impact(
    notional: float,
    adv: float,
    *,
    coefficient: float = 0.1,
) -> float:
    """Linear market impact in basis points.

    ``cost_bps = coefficient * (notional / ADV) * 10000``

    Args:
        notional: Trade notional.
        adv: Average daily volume.
        coefficient: Impact coefficient (calibrate to your market).
    """
    if adv < EPSILON:
        return 0.0
    return coefficient * (notional / adv) * 10000.0


def sqrt_impact(
    notional: float,
    adv: float,
    *,
    coefficient: float = 0.1,
    volatility: float = 0.0,
    exponent: float = 0.5,
) -> float:
    """Square-root market impact (Torre 1997, Barra).

    ``cost_bps = coefficient * sigma * (notional / ADV)^exponent * 10000``

    Args:
        notional: Trade notional.
        adv: Average daily volume.
        coefficient: Impact coefficient.
        volatility: Daily price volatility (if 0, uses coefficient alone).
        exponent: Impact exponent (default 0.5 = square root).
    """
    if adv < EPSILON:
        return 0.0
    sigma = volatility if volatility > EPSILON else 1.0
    return coefficient * sigma * ((notional / adv) ** exponent) * 10000.0


def spread_cost(bid_ask_spread_bps: float) -> float:
    """Half-spread cost: ``cost = spread / 2``. The minimum cost of any trade."""
    return bid_ask_spread_bps / 2.0


def total_cost(
    notional: float,
    adv: float,
    volatility: float,
    *,
    bid_ask_bps: float = 5.0,
    commission_bps: float = 0.0,
    impact_model: str = "sqrt",
    impact_coefficient: float = 0.1,
    impact_exponent: float = 0.5,
) -> CostEstimate:
    """Combined transaction cost: spread + impact + commission.

    Args:
        notional: Trade notional.
        adv: Average daily volume.
        volatility: Daily price volatility.
        bid_ask_bps: Bid-ask spread in bps.
        commission_bps: Broker commission in bps.
        impact_model: ``"sqrt"`` (default), ``"linear"``, or ``"almgren_chriss"``.
        impact_coefficient: Impact coefficient.
        impact_exponent: Impact exponent (for sqrt model).
    """
    spread = spread_cost(bid_ask_bps)

    if impact_model == "sqrt":
        impact = sqrt_impact(notional, adv, coefficient=impact_coefficient,
                             volatility=volatility, exponent=impact_exponent)
        return CostEstimate(
            spread_cost_bps=round(spread, 4),
            temporary_cost_bps=round(impact, 4),
            commission_bps=commission_bps,
            total_cost_bps=round(spread + impact + commission_bps, 4),
        )
    elif impact_model == "linear":
        impact = linear_impact(notional, adv, coefficient=impact_coefficient)
        return CostEstimate(
            spread_cost_bps=round(spread, 4),
            temporary_cost_bps=round(impact, 4),
            commission_bps=commission_bps,
            total_cost_bps=round(spread + impact + commission_bps, 4),
        )
    elif impact_model == "almgren_chriss":
        ac = almgren_chriss(notional, adv, volatility)
        ac.spread_cost_bps = round(spread, 4)
        ac.commission_bps = commission_bps
        ac.total_cost_bps = round(
            ac.permanent_cost_bps + ac.temporary_cost_bps + spread + commission_bps, 4
        )
        return ac
    else:
        raise ValueError(f"Unknown impact_model '{impact_model}'. Use: sqrt, linear, almgren_chriss")


@dataclass
class CapacityEstimate:
    """Strategy capacity: the AUM a strategy can run before it stops paying.

    ``capacity_usd`` is the binding minimum of two limits:
      * alpha-decay — the AUM at which annualized round-trip costs (spread +
        commission + square-root market impact, scaled by turnover) equal the
        gross annual alpha, so the net edge hits zero;
      * participation — the AUM at which the largest position would exceed
        ``max_participation`` of that name's ADV (a liquidity ceiling that binds
        regardless of alpha).
    """
    capacity_usd: Optional[float] = None
    alpha_decay_capacity_usd: Optional[float] = None
    participation_capacity_usd: Optional[float] = None
    binding_constraint: str = "unknown"          # 'alpha_decay' | 'participation' | 'none'
    bottleneck_symbol: Optional[str] = None
    gross_alpha_bps: float = 0.0
    annual_turnover: float = 0.0
    net_alpha_bps_at_capacity: Optional[float] = None
    per_symbol: List[Dict[str, Any]] = field(default_factory=list)
    note: str = ""


def alpha_decay_capacity(
    gross_alpha_bps: float,
    annual_turnover: float,
    holdings: List[Dict[str, Any]],
    *,
    spread_bps: float = 5.0,
    commission_bps: float = 0.0,
    impact_coefficient: float = 0.1,
    impact_exponent: float = 0.5,
    max_participation: float = 0.10,
    max_aum: float = 1e12,
    iterations: int = 60,
) -> CapacityEstimate:
    """Estimate strategy capacity via alpha-decay and a participation ceiling.

    Args:
        gross_alpha_bps: gross annual alpha (before costs), in basis points.
        annual_turnover: two-way annual turnover (e.g. 2.0 == 200%/yr).
        holdings: list of ``{"symbol", "weight", "adv_usd", "volatility"}`` for
            the strategy's typical book (weights are fractions of AUM;
            ``adv_usd`` is dollar ADV; ``volatility`` is daily return vol).
        max_participation: liquidity ceiling — a position may not exceed this
            fraction of the name's ADV.

    Returns a :class:`CapacityEstimate`. ``capacity_usd`` is the binding minimum
    of the alpha-decay and participation limits.
    """
    holdings = [h for h in holdings if float(h.get("weight", 0.0)) > 0 and float(h.get("adv_usd", 0.0)) > 0]
    if gross_alpha_bps <= 0 or annual_turnover <= 0 or not holdings:
        return CapacityEstimate(
            gross_alpha_bps=gross_alpha_bps, annual_turnover=annual_turnover,
            binding_constraint="none",
            note="No positive gross alpha, turnover, or tradable holdings — capacity is undefined.",
        )

    fixed_per_turn_bps = (spread_bps / 2.0) + commission_bps  # AUM-independent, per unit turnover

    def annual_cost_bps(aum: float) -> float:
        impact = 0.0
        for h in holdings:
            w = float(h["weight"])
            notional = w * aum
            imp = sqrt_impact(notional, float(h["adv_usd"]), coefficient=impact_coefficient,
                              volatility=float(h.get("volatility", 0.0)), exponent=impact_exponent)
            impact += w * imp
        weight_sum = sum(float(h["weight"]) for h in holdings)
        return annual_turnover * (fixed_per_turn_bps * weight_sum + impact)

    # Alpha-decay: binary-search the AUM where net alpha crosses zero.
    if gross_alpha_bps - annual_cost_bps(1.0) <= 0:
        alpha_cap = 0.0
    elif gross_alpha_bps - annual_cost_bps(max_aum) > 0:
        alpha_cap = max_aum  # impact never eats the alpha within the search range
    else:
        lo, hi = 1.0, max_aum
        for _ in range(iterations):
            mid = (lo + hi) / 2.0
            if gross_alpha_bps - annual_cost_bps(mid) > 0:
                lo = mid
            else:
                hi = mid
        alpha_cap = lo

    # Participation ceiling: position (weight * AUM) <= max_participation * ADV.
    part_caps = {h["symbol"]: (max_participation * float(h["adv_usd"]) / float(h["weight"]))
                 for h in holdings if h.get("symbol") is not None}
    if not part_caps:
        part_caps = {f"idx{i}": (max_participation * float(h["adv_usd"]) / float(h["weight"]))
                     for i, h in enumerate(holdings)}
    part_bottleneck_sym = min(part_caps, key=part_caps.get)  # type: ignore[arg-type]
    part_cap = part_caps[part_bottleneck_sym]

    capacity = min(alpha_cap, part_cap)
    binding = "alpha_decay" if alpha_cap <= part_cap else "participation"

    per_symbol = [{
        "symbol": h.get("symbol"),
        "weight": round(float(h["weight"]), 6),
        "adv_usd": round(float(h["adv_usd"]), 2),
        "participation_capacity_usd": round(max_participation * float(h["adv_usd"]) / float(h["weight"]), 2),
    } for h in holdings]

    return CapacityEstimate(
        capacity_usd=round(capacity, 2),
        alpha_decay_capacity_usd=round(alpha_cap, 2),
        participation_capacity_usd=round(part_cap, 2),
        binding_constraint=binding,
        bottleneck_symbol=(part_bottleneck_sym if binding == "participation" else None),
        gross_alpha_bps=round(gross_alpha_bps, 3),
        annual_turnover=round(annual_turnover, 4),
        net_alpha_bps_at_capacity=round(gross_alpha_bps - annual_cost_bps(capacity), 3),
        per_symbol=per_symbol,
    )


def turnover_analysis(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    *,
    cost_model: Optional[CostModel] = None,
    adv: Optional[Dict[str, float]] = None,
    volatility: Optional[Dict[str, float]] = None,
    portfolio_value: float = 1_000_000.0,
) -> TurnoverResult:
    """Analyse portfolio turnover and estimated rebalancing cost.

    Args:
        current_weights: ``{symbol: current_weight}``.
        target_weights: ``{symbol: target_weight}``.
        cost_model: Cost model for per-asset cost estimation.
        adv: ``{symbol: average_daily_volume}`` (optional).
        volatility: ``{symbol: daily_volatility}`` (optional).
        portfolio_value: Total portfolio value for notional calculation.
    """
    all_symbols = set(current_weights.keys()) | set(target_weights.keys())
    per_asset_turn: Dict[str, float] = {}
    per_asset_cost: Dict[str, float] = {}

    for sym in all_symbols:
        curr = current_weights.get(sym, 0.0)
        tgt = target_weights.get(sym, 0.0)
        turn = abs(tgt - curr)
        per_asset_turn[sym] = round(turn, 6)

        if cost_model and turn > EPSILON:
            notional = turn * portfolio_value
            sym_adv = (adv or {}).get(sym, portfolio_value * 0.01)
            sym_vol = (volatility or {}).get(sym, 0.02)
            per_asset_cost[sym] = round(cost_model.estimate(notional, sym_adv, sym_vol), 4)
        else:
            per_asset_cost[sym] = 0.0

    one_way = sum(per_asset_turn.values()) / 2.0
    two_way = sum(per_asset_turn.values())
    total_cost = sum(per_asset_cost.values())

    return TurnoverResult(
        one_way_turnover=round(one_way, 6),
        two_way_turnover=round(two_way, 6),
        estimated_cost_bps=round(total_cost, 4),
        per_asset_turnover=per_asset_turn,
        per_asset_cost_bps=per_asset_cost,
    )
