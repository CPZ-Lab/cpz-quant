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
    """Portfolio turnover analysis.

    ``estimated_cost_bps`` is the cost of the whole rebalance in basis points
    of portfolio value (``estimated_cost_usd`` in dollars); both are ``None``
    when no cost model was given (the cost is unknown, not zero).
    ``per_asset_cost_bps`` is each trade's cost in basis points of that
    trade's own notional, so those figures are not additive.
    """
    one_way_turnover: float = 0.0
    two_way_turnover: float = 0.0
    estimated_cost_bps: Optional[float] = None
    per_asset_turnover: Dict[str, float] = field(default_factory=dict)
    per_asset_cost_bps: Dict[str, float] = field(default_factory=dict)
    estimated_cost_usd: Optional[float] = None


IMPACT_MODELS = ("sqrt", "linear", "almgren_chriss")


@dataclass
class CostModel:
    """Pluggable cost model for portfolio optimisation.

    All parameters are configurable so users can calibrate to their
    broker and market conditions.

    - ``impact_model``: ``"sqrt"`` (default), ``"linear"`` or
      ``"almgren_chriss"``; any other name raises ``ValueError``.
    - ``impact_coefficient`` / ``impact_exponent``: parameters of the sqrt
      (coefficient and exponent) and linear (coefficient) models.
    - ``permanent_fraction``: share of the sqrt or linear impact that is
      permanent; :meth:`breakdown` reports the split. A single trade pays
      both parts, so the total in :meth:`estimate` does not depend on it.
    - ``almgren_chriss`` uses the Almgren et al. (2005) calibration built
      into :func:`almgren_chriss` and has its own permanent/temporary split;
      it refuses a non-default ``impact_coefficient``, ``impact_exponent``
      or ``permanent_fraction`` rather than silently ignore them. Call
      :func:`almgren_chriss` directly to set its parameters.
    """
    spread_bps: float = 5.0
    impact_coefficient: float = 0.1
    impact_exponent: float = 0.5
    impact_model: str = "sqrt"
    commission_bps: float = 0.0
    permanent_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.impact_model not in IMPACT_MODELS:
            raise ValueError(
                f"Unknown impact_model '{self.impact_model}'. Use: {', '.join(IMPACT_MODELS)}"
            )
        if not 0.0 <= self.permanent_fraction <= 1.0:
            raise ValueError(f"permanent_fraction must be in [0, 1], got {self.permanent_fraction}")
        if self.impact_model == "almgren_chriss":
            ignored = [
                name for name, value, default in (
                    ("impact_coefficient", self.impact_coefficient, 0.1),
                    ("impact_exponent", self.impact_exponent, 0.5),
                    ("permanent_fraction", self.permanent_fraction, 0.5),
                ) if value != default
            ]
            if ignored:
                raise ValueError(
                    f"impact_model='almgren_chriss' does not use {', '.join(ignored)}: it applies "
                    "the Almgren et al. (2005) calibration. Call almgren_chriss() directly to set "
                    "gamma, eta, alpha and beta, or use the sqrt or linear model."
                )

    def breakdown(
        self,
        notional: float,
        adv: float = 0.0,
        volatility: float = 0.0,
    ) -> CostEstimate:
        """Cost of one trade (one leg, a buy or a sell) split into its parts, in bps.

        A sell costs the same as a buy of the same size: impact depends on
        ``abs(notional)``. When impact is priced (non-zero notional and
        coefficient) every model needs ``adv > 0`` and the sqrt and
        Almgren-Chriss models also ``volatility > 0`` (``ValueError``
        otherwise). With ``impact_coefficient=0`` (spread and commission
        only) no market inputs are needed.
        """
        spread = self.spread_bps / 2.0
        if self.impact_model == "almgren_chriss":
            ac = almgren_chriss(notional, adv, volatility)
            perm, temp = ac.permanent_cost_bps, ac.temporary_cost_bps
        else:
            if self.impact_model == "sqrt":
                impact = sqrt_impact(notional, adv, coefficient=self.impact_coefficient,
                                     volatility=volatility, exponent=self.impact_exponent)
            else:
                impact = linear_impact(notional, adv, coefficient=self.impact_coefficient)
            perm = impact * self.permanent_fraction
            temp = impact - perm
        return CostEstimate(
            permanent_cost_bps=perm,
            temporary_cost_bps=temp,
            spread_cost_bps=spread,
            commission_bps=self.commission_bps,
            total_cost_bps=spread + perm + temp + self.commission_bps,
        )

    def estimate(
        self,
        notional: float,
        adv: float = 0.0,
        volatility: float = 0.0,
    ) -> float:
        """Cost of one trade (one leg) in basis points of its notional.

        Half the spread, plus market impact, plus commission, for a single
        buy or sell. A round trip (into a position and back out) is two
        trades and costs about twice this. :func:`turnover_analysis` adds up
        these one-leg costs over the trades of a rebalance.

        Args:
            notional: Trade notional value (a sell may be negative).
            adv: Average daily volume (notional).
            volatility: Daily volatility of the asset.
        """
        return self.breakdown(notional, adv, volatility).total_cost_bps


def _require_liquidity(
    adv: float, volatility: Optional[float] = None, *, model: str, needs_volatility: bool = False,
) -> None:
    """Impact needs the name's ADV (and, for vol-scaled models, its volatility)."""
    if adv is None or not (adv == adv and adv > 0):  # also rejects NaN
        raise ValueError(f"{model} impact needs the average daily volume (adv > 0), got {adv}")
    if needs_volatility and (volatility is None or not (volatility == volatility and volatility > 0)):
        raise ValueError(
            f"{model} impact needs the asset's daily volatility (> 0), got {volatility}"
        )


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
    if notional == 0 or (gamma == 0 and eta == 0):
        return CostEstimate()  # nothing traded, or impact not priced: no inputs needed
    _require_liquidity(adv, volatility, model="Almgren-Chriss", needs_volatility=True)
    q_over_v = abs(notional) / adv  # a sell moves the price as much as a buy
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

    ``cost_bps = coefficient * (|notional| / ADV) * 10000``

    Args:
        notional: Trade notional (a sell may be negative; it costs the same).
        adv: Average daily volume.
        coefficient: Impact coefficient (calibrate to your market).

    Raises:
        ValueError: ``adv`` is not positive (unknown liquidity is not free).
            A zero notional or a zero coefficient has zero impact and needs
            no ADV.
    """
    if notional == 0 or coefficient == 0:
        return 0.0
    _require_liquidity(adv, model="linear")
    return coefficient * (abs(notional) / adv) * 10000.0


def sqrt_impact(
    notional: float,
    adv: float,
    *,
    coefficient: float = 0.1,
    volatility: Optional[float] = None,
    exponent: float = 0.5,
) -> float:
    """Square-root market impact (Torre 1997, Barra).

    ``cost_bps = coefficient * sigma * (|notional| / ADV)^exponent * 10000``

    Args:
        notional: Trade notional (a sell may be negative; it costs the same).
        adv: Average daily volume.
        coefficient: Impact coefficient.
        volatility: Daily return volatility of the asset (required, > 0).
        exponent: Impact exponent (default 0.5 = square root).

    Raises:
        ValueError: ``adv`` or ``volatility`` is missing or not positive. A
            missing volatility used to become 1.0 (100% a day), which made
            the impact about 50 times too large for a typical stock. A zero
            notional or a zero coefficient has zero impact and needs neither.
    """
    if notional == 0 or coefficient == 0:
        return 0.0
    _require_liquidity(adv, volatility, model="square-root", needs_volatility=True)
    return coefficient * float(volatility) * ((abs(notional) / adv) ** exponent) * 10000.0  # type: ignore[arg-type]


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
        impact_coefficient: Impact coefficient (sqrt and linear models).
        impact_exponent: Impact exponent (sqrt model). The Almgren-Chriss
            model uses its own calibration and refuses non-default values.
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
        if impact_coefficient != 0.1 or impact_exponent != 0.5:
            raise ValueError(
                "impact_model='almgren_chriss' does not use impact_coefficient or "
                "impact_exponent: it applies the Almgren et al. (2005) calibration. Call "
                "almgren_chriss() directly to set gamma, eta, alpha and beta."
            )
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
            the strategy's typical book (weights are fractions of AUM, negative
            for shorts; ``adv_usd`` is dollar ADV; ``volatility`` is daily
            return vol). Short positions trade and hold liquidity like longs,
            so both limits use ``abs(weight)``. Zero weights are skipped; a
            non-zero position without a positive ``adv_usd`` raises
            ``ValueError``, because the thinnest name sets capacity and
            dropping it would overstate it.
        max_participation: liquidity ceiling — a position may not exceed this
            fraction of the name's ADV.

    Returns a :class:`CapacityEstimate`. ``capacity_usd`` is the binding minimum
    of the alpha-decay and participation limits.
    """
    holdings = [h for h in holdings if float(h.get("weight", 0.0)) != 0.0]
    no_adv = [h.get("symbol", f"holding {i}") for i, h in enumerate(holdings)
              if not float(h.get("adv_usd") or 0.0) > 0]
    if no_adv:
        raise ValueError(
            f"holdings without a positive adv_usd: {no_adv}. Capacity cannot be estimated "
            "without each position's liquidity."
        )
    no_vol = [h.get("symbol", f"holding {i}") for i, h in enumerate(holdings)
              if not float(h.get("volatility") or 0.0) > 0]
    if no_vol:
        raise ValueError(
            f"holdings without a positive daily volatility: {no_vol}. The square-root impact "
            "model scales with it."
        )
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
            w = abs(float(h["weight"]))
            notional = w * aum
            imp = sqrt_impact(notional, float(h["adv_usd"]), coefficient=impact_coefficient,
                              volatility=float(h.get("volatility", 0.0)), exponent=impact_exponent)
            impact += w * imp
        weight_sum = sum(abs(float(h["weight"])) for h in holdings)
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
    part_caps = {h["symbol"]: (max_participation * float(h["adv_usd"]) / abs(float(h["weight"])))
                 for h in holdings if h.get("symbol") is not None}
    if not part_caps:
        part_caps = {f"idx{i}": (max_participation * float(h["adv_usd"]) / abs(float(h["weight"])))
                     for i, h in enumerate(holdings)}
    part_bottleneck_sym = min(part_caps, key=part_caps.get)  # type: ignore[arg-type]
    part_cap = part_caps[part_bottleneck_sym]

    capacity = min(alpha_cap, part_cap)
    binding = "alpha_decay" if alpha_cap <= part_cap else "participation"

    per_symbol = [{
        "symbol": h.get("symbol"),
        "weight": round(float(h["weight"]), 6),
        "adv_usd": round(float(h["adv_usd"]), 2),
        "participation_capacity_usd": round(
            max_participation * float(h["adv_usd"]) / abs(float(h["weight"])), 2),
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

    Each trade's cost comes from ``cost_model.estimate`` in bps of that
    trade's notional (``per_asset_cost_bps``). The rebalance cost is the sum
    of the trades' dollar costs, ``sum(|dw_i| * cost_bps_i)``, reported in
    bps of portfolio value (``estimated_cost_bps``) and in dollars
    (``estimated_cost_usd``). Adding the per-trade bps directly would count
    a 0.01% top-up as heavily as a full position switch.

    Without a cost model the cost is unknown: ``estimated_cost_bps`` and
    ``estimated_cost_usd`` are ``None`` and ``per_asset_cost_bps`` is empty.

    Args:
        current_weights: ``{symbol: current_weight}``.
        target_weights: ``{symbol: target_weight}``.
        cost_model: Cost model for per-asset cost estimation.
        adv: ``{symbol: average_daily_volume}``; required for every traded
            symbol when *cost_model* prices market impact (any model except
            ``impact_coefficient=0``).
        volatility: ``{symbol: daily_volatility}``; required for every traded
            symbol when the model is sqrt (with a non-zero coefficient) or
            Almgren-Chriss.
        portfolio_value: Total portfolio value for notional calculation.

    Raises:
        ValueError: a traded symbol has no ADV, or no volatility where the
            model needs one, or a value that is not positive. These used to
            be filled with 1% of the book and 2% a day.
    """
    all_symbols = set(current_weights.keys()) | set(target_weights.keys())
    per_asset_turn: Dict[str, float] = {}

    for sym in all_symbols:
        curr = current_weights.get(sym, 0.0)
        tgt = target_weights.get(sym, 0.0)
        per_asset_turn[sym] = round(abs(tgt - curr), 6)

    one_way = sum(per_asset_turn.values()) / 2.0
    two_way = sum(per_asset_turn.values())
    if cost_model is None:
        return TurnoverResult(
            one_way_turnover=round(one_way, 6),
            two_way_turnover=round(two_way, 6),
            per_asset_turnover=per_asset_turn,
        )

    traded = sorted(s for s in all_symbols
                    if abs(target_weights.get(s, 0.0) - current_weights.get(s, 0.0)) > EPSILON)
    adv_map = adv or {}
    vol_map = volatility or {}

    def _bad(value: Optional[float]) -> bool:
        return value is None or not (value == value and value > 0)

    # A model that prices no impact (spread and commission only) needs no
    # market inputs; one that does needs them for every traded symbol.
    prices_impact = cost_model.impact_model == "almgren_chriss" or cost_model.impact_coefficient != 0
    needs_vol = prices_impact and cost_model.impact_model in ("sqrt", "almgren_chriss")
    no_adv = [s for s in traded if prices_impact and _bad(adv_map.get(s))]
    no_vol = [s for s in traded if needs_vol and _bad(vol_map.get(s))]
    if no_adv or no_vol:
        parts = []
        if no_adv:
            parts.append(f"no positive ADV for {no_adv}")
        if no_vol:
            parts.append(f"no positive daily volatility for {no_vol}")
        raise ValueError("turnover_analysis: " + "; ".join(parts) + " (needed to cost the trades)")

    per_asset_cost: Dict[str, float] = {}
    book_cost_bps = 0.0
    for sym in traded:
        dw = abs(target_weights.get(sym, 0.0) - current_weights.get(sym, 0.0))
        cost_bps = cost_model.estimate(dw * portfolio_value, adv_map.get(sym, 0.0), vol_map.get(sym, 0.0))
        per_asset_cost[sym] = round(cost_bps, 4)
        # bps of the book: each trade's bps weighted by the share of the book it trades.
        book_cost_bps += dw * cost_bps

    return TurnoverResult(
        one_way_turnover=round(one_way, 6),
        two_way_turnover=round(two_way, 6),
        estimated_cost_bps=round(book_cost_bps, 4),
        per_asset_turnover=per_asset_turn,
        per_asset_cost_bps=per_asset_cost,
        estimated_cost_usd=round(book_cost_bps / 1e4 * portfolio_value, 2),
    )
