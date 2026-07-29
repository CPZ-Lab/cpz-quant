"""Tests for alpha-decay + participation strategy capacity."""

from __future__ import annotations

from cpz_quant.portfolio import alpha_decay_capacity


def _book(adv=400e6, weight_split=(0.5, 0.5), vol=0.02):
    return [
        {"symbol": "AAA", "weight": weight_split[0], "adv_usd": adv, "volatility": vol},
        {"symbol": "BBB", "weight": weight_split[1], "adv_usd": adv * 0.75, "volatility": vol},
    ]


class TestCapacity:
    def test_positive_capacity_for_a_liquid_book(self):
        cap = alpha_decay_capacity(gross_alpha_bps=300, annual_turnover=4.0, holdings=_book())
        assert cap.capacity_usd is not None and cap.capacity_usd > 0
        assert cap.binding_constraint in ("alpha_decay", "participation")

    def test_more_gross_alpha_raises_capacity(self):
        low = alpha_decay_capacity(gross_alpha_bps=150, annual_turnover=4.0, holdings=_book())
        high = alpha_decay_capacity(gross_alpha_bps=600, annual_turnover=4.0, holdings=_book())
        assert high.alpha_decay_capacity_usd > low.alpha_decay_capacity_usd

    def test_more_turnover_lowers_capacity(self):
        slow = alpha_decay_capacity(gross_alpha_bps=300, annual_turnover=2.0, holdings=_book())
        fast = alpha_decay_capacity(gross_alpha_bps=300, annual_turnover=12.0, holdings=_book())
        assert fast.alpha_decay_capacity_usd < slow.alpha_decay_capacity_usd

    def test_less_liquid_lowers_capacity(self):
        liquid = alpha_decay_capacity(gross_alpha_bps=300, annual_turnover=4.0, holdings=_book(adv=800e6))
        illiquid = alpha_decay_capacity(gross_alpha_bps=300, annual_turnover=4.0, holdings=_book(adv=20e6))
        assert illiquid.capacity_usd < liquid.capacity_usd

    def test_illiquid_name_makes_participation_bind(self):
        book = [
            {"symbol": "BIG", "weight": 0.5, "adv_usd": 1e9, "volatility": 0.02},
            {"symbol": "TINY", "weight": 0.5, "adv_usd": 2e5, "volatility": 0.05},  # micro-cap
        ]
        cap = alpha_decay_capacity(gross_alpha_bps=800, annual_turnover=2.0, holdings=book)
        assert cap.binding_constraint == "participation"
        assert cap.bottleneck_symbol == "TINY"

    def test_net_alpha_near_zero_when_alpha_decay_binds(self):
        cap = alpha_decay_capacity(gross_alpha_bps=300, annual_turnover=8.0, holdings=_book(adv=50e6))
        if cap.binding_constraint == "alpha_decay":
            assert abs(cap.net_alpha_bps_at_capacity) < 1.0  # net edge ~ 0 at capacity

    def test_no_alpha_is_undefined(self):
        cap = alpha_decay_capacity(gross_alpha_bps=0, annual_turnover=4.0, holdings=_book())
        assert cap.capacity_usd is None
        assert cap.binding_constraint == "none"
