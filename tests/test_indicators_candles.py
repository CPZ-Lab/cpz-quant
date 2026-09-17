"""Tests for cpz_quant.indicators.candles (TA-Lib compatible candlestick patterns).

Two layers:

* Reference parity: every kernel must equal ``talib.CDL*`` element for element on
  a seeded synthetic OHLC fixture built to trigger every pattern many times,
  including threshold ties (2-decimal price grid, equal opens/closes/highs/lows).
  These tests skip when the ``talib`` package is not installed.
* Textbook examples: hand-built bars for a few key patterns, checked without the
  reference library so behaviour stays covered everywhere.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Dict, List, Tuple

import cpz_quant.indicators.candles as candles
import numpy as np
import pytest
from cpz_quant.indicators.candles import CANDLE_PATTERNS, TALIB_NAMES

OHLC = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
Bar = Tuple[float, float, float, float]

PENETRATION_PATTERNS = {
    "abandoned_baby",
    "dark_cloud_cover",
    "evening_doji_star",
    "evening_star",
    "mat_hold",
    "morning_doji_star",
    "morning_star",
}

# Minimum number of non-zero TA-Lib signals each pattern must produce on the
# fixture, so no parity check passes vacuously on an all-zero series.
MIN_HITS = 5

# Non-zero signal counts on make_ohlc() (n=5000, default seed), recorded from
# TA-Lib C 0.8.1 via the talib 0.8.0 wrapper. The parity tests assert TA-Lib
# still produces these counts; the talib-free regression test asserts the
# kernels do.
# SHA-256 of make_ohlc() at the default arguments when EXPECTED_HITS was
# recorded. NumPy does not promise a stable random stream across releases, so
# the recorded counts are only asserted when the fixture reproduces exactly;
# element-wise TA-Lib parity does not depend on it.
FIXTURE_SHA256 = "c25a4c0502476cb0e4b2e3a5842c43235e05e1f020f435652278a2cdf5bdcd4e"

EXPECTED_HITS: Dict[str, int] = {
    "abandoned_baby": 7,
    "advance_block": 24,
    "belt_hold": 1282,
    "breakaway": 9,
    "closing_marubozu": 1226,
    "concealing_baby_swallow": 12,
    "counterattack": 19,
    "dark_cloud_cover": 9,
    "doji": 1471,
    "doji_star": 277,
    "dragonfly_doji": 310,
    "engulfing": 393,
    "evening_doji_star": 12,
    "evening_star": 36,
    "gap_side_side_white": 31,
    "gravestone_doji": 349,
    "hammer": 348,
    "hanging_man": 300,
    "harami": 323,
    "harami_cross": 180,
    "high_wave": 361,
    "hikkake": 669,
    "hikkake_modified": 9,
    "homing_pigeon": 49,
    "identical_three_crows": 21,
    "in_neck": 16,
    "inverted_hammer": 266,
    "kicking": 13,
    "kicking_by_length": 13,
    "ladder_bottom": 10,
    "long_legged_doji": 1087,
    "long_line": 1113,
    "marubozu": 905,
    "mat_hold": 10,
    "matching_low": 86,
    "morning_doji_star": 9,
    "morning_star": 19,
    "on_neck": 24,
    "piercing": 8,
    "rickshaw_man": 233,
    "rise_fall_three_methods": 8,
    "separating_lines": 42,
    "shooting_star": 238,
    "short_line": 1756,
    "spinning_top": 499,
    "stalled_pattern": 18,
    "stick_sandwich": 14,
    "takuri": 415,
    "tasuki_gap": 7,
    "three_black_crows": 14,
    "three_inside": 46,
    "three_line_strike": 9,
    "three_outside": 126,
    "three_stars_in_south": 12,
    "three_white_soldiers": 18,
    "thrusting": 29,
    "tristar": 54,
    "two_crows": 16,
    "unique_three_river": 10,
    "upside_gap_two_crows": 14,
    "xside_gap_three_methods": 5,
}


# ── Synthetic fixture ────────────────────────────────────────────────

# Rare multi-bar shapes, as (open, high, low, close) offsets from a base price.
# Offsets are jittered and rounded when injected, so some instances land just
# inside and some just outside TA-Lib's thresholds. Shapes listed in
# _TWO_SIDED are also injected mirrored (bullish <-> bearish).
_TEMPLATES: Dict[str, List[Bar]] = {
    "mat_hold": [
        (0.00, 2.05, -0.02, 2.00),
        (2.40, 2.45, 2.20, 2.25),
        (2.10, 2.20, 1.85, 1.90),
        (2.00, 2.05, 1.75, 1.80),
        (1.90, 2.85, 1.88, 2.80),
    ],
    "rise_fall_three_methods": [
        (0.00, 2.05, -0.02, 2.00),
        (1.90, 1.95, 1.70, 1.75),
        (1.70, 1.74, 1.50, 1.55),
        (1.50, 1.55, 1.30, 1.35),
        (1.45, 2.65, 1.43, 2.60),
    ],
    "three_stars_in_south": [
        (1.00, 1.02, -1.20, 0.00),
        (0.50, 0.55, -0.60, 0.10),
        (0.20, 0.20, 0.05, 0.05),
    ],
    "two_crows": [
        (0.00, 2.02, -0.02, 2.00),
        (2.60, 2.65, 2.28, 2.30),
        (2.50, 2.52, 0.95, 1.00),
    ],
    "upside_gap_two_crows": [
        (0.00, 2.02, -0.02, 2.00),
        (2.50, 2.52, 2.33, 2.35),
        (2.60, 2.62, 2.08, 2.10),
    ],
    "breakaway": [
        (2.00, 2.02, -0.02, 0.00),
        (-0.30, -0.28, -0.62, -0.60),
        (-0.50, -0.45, -0.85, -0.80),
        (-0.70, -0.65, -1.05, -1.00),
        (-0.90, -0.12, -0.92, -0.15),
    ],
    "concealing_baby_swallow": [
        (2.00, 2.00, 1.00, 1.00),
        (0.90, 0.90, 0.00, 0.00),
        (-0.20, 0.40, -0.60, -0.50),
        (0.60, 0.70, -0.90, -0.80),
    ],
    "unique_three_river": [
        (2.00, 2.05, -0.02, 0.00),
        (1.00, 1.05, -0.50, 0.20),
        (-0.30, -0.10, -0.35, -0.20),
    ],
    "counterattack": [
        (2.00, 2.02, -0.02, 0.00),
        (-1.90, 0.02, -1.95, 0.00),
    ],
    "kicking": [
        (0.00, 1.60, 0.00, 1.60),
        (-0.40, -0.40, -2.00, -2.00),
    ],
    "abandoned_baby": [
        (2.00, 2.02, -0.02, 0.00),
        (-0.40, -0.30, -0.50, -0.40),
        (0.20, 1.40, 0.15, 1.30),
    ],
    "three_black_crows": [
        (0.00, 2.10, -0.02, 2.00),
        (1.90, 1.95, 1.18, 1.20),
        (1.30, 1.35, 0.49, 0.50),
        (0.60, 0.65, -0.19, -0.20),
    ],
    "identical_three_crows": [
        (2.00, 2.05, 1.19, 1.20),
        (1.20, 1.25, 0.39, 0.40),
        (0.40, 0.45, -0.40, -0.39),
    ],
    "ladder_bottom": [
        (2.00, 2.05, 1.50, 1.55),
        (1.60, 1.65, 1.10, 1.15),
        (1.20, 1.25, 0.70, 0.75),
        (0.70, 1.30, 0.40, 0.45),
        (0.80, 1.60, 0.78, 1.50),
    ],
    "stick_sandwich": [
        (1.00, 1.02, -0.02, 0.00),
        (0.20, 1.10, 0.10, 1.00),
        (1.20, 1.22, -0.03, 0.00),
    ],
    "tasuki_gap": [
        (0.00, 1.02, -0.02, 1.00),
        (1.40, 2.42, 1.38, 2.40),
        (2.00, 2.02, 1.18, 1.20),
    ],
}


_TWO_SIDED = {
    "abandoned_baby",
    "breakaway",
    "counterattack",
    "kicking",
    "rise_fall_three_methods",
    "tasuki_gap",
}


def _mirror(bars: List[Bar]) -> List[Bar]:
    """Reflect a template through its base price (bullish <-> bearish)."""
    return [(-o, -lo, -h, -c) for o, h, lo, c in bars]


def make_ohlc(n: int = 5000, seed: int = 20260917) -> OHLC:
    """Deterministic OHLC series that exercises every TA-Lib candlestick rule.

    Mixes trending and flat regimes, doji and marubozu candles, zero and long
    shadows, body gaps and exact equalities on a 2-decimal price grid, and
    injects jittered templates of the rarest multi-bar shapes.
    """
    rng = np.random.default_rng(seed)
    names = sorted(_TEMPLATES)
    rows: List[Bar] = []
    price = 30.0
    drift = 0.0
    regime_left = 0

    def shadow() -> float:
        v = rng.random()
        if v < 0.3:
            return 0.0
        if v < 0.55:
            return float(rng.choice([0.01, 0.02]))
        if v < 0.85:
            return abs(float(rng.normal(0.0, 0.2)))
        return float(rng.uniform(0.4, 1.5))

    while len(rows) < n:
        if rng.random() < 0.035:
            name = names[int(rng.integers(len(names)))]
            bars = _TEMPLATES[name]
            if name in _TWO_SIDED and rng.random() < 0.5:
                bars = _mirror(bars)
            scale = float(rng.uniform(0.85, 1.2))
            for o, h, lo, c in bars:
                rows.append((price + o * scale, price + h * scale, price + lo * scale,
                             price + c * scale))
            price = rows[-1][3]
        else:
            if regime_left <= 0:
                drift = float(rng.choice([-0.35, -0.15, 0.0, 0.15, 0.35]))
                regime_left = int(rng.integers(3, 25))
            regime_left -= 1
            u = rng.random()
            if u < 0.25:
                gap = 0.0
            elif u < 0.85:
                gap = float(rng.normal(0.0, 0.15))
            else:
                gap = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.5))
            op = price + gap
            kind = rng.random()
            if kind < 0.15:
                body = float(rng.choice([0.0, 0.01, 0.02]))
            elif kind < 0.5:
                body = abs(float(rng.normal(0.0, 0.15)))
            elif kind < 0.9:
                body = abs(float(rng.normal(0.5, 0.3)))
            else:
                body = float(rng.uniform(1.0, 2.5))
            sign = 1.0 if drift + float(rng.normal(0.0, 0.4)) >= 0 else -1.0
            cl = op + sign * body
            rows.append((op, max(op, cl) + shadow(), min(op, cl) - shadow(), cl))
            price = cl + drift * float(rng.random())
        if price < 12.0:
            price += 3.0
        elif price > 48.0:
            price -= 3.0
    arr = np.round(np.asarray(rows[:n], dtype=np.float64), 2)
    o, h, lo, c = arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy(), arr[:, 3].copy()
    h = np.maximum(h, np.maximum(o, c))
    lo = np.minimum(lo, np.minimum(o, c))
    return o, h, lo, c


@pytest.fixture(scope="module")
def ohlc() -> OHLC:
    return make_ohlc()


def _fixture_is_recorded(ohlc: OHLC) -> bool:
    digest = hashlib.sha256(
        b"".join(np.ascontiguousarray(a, dtype="<f8").tobytes() for a in ohlc)
    ).hexdigest()
    return digest == FIXTURE_SHA256


def test_fixture_is_valid_ohlc(ohlc: OHLC) -> None:
    o, h, lo, c = ohlc
    assert len(o) == 5000
    assert np.all(h >= np.maximum(o, c))
    assert np.all(lo <= np.minimum(o, c))
    assert np.all(np.isfinite(np.stack(ohlc)))
    again = make_ohlc()
    assert all(np.array_equal(a, b) for a, b in zip(ohlc, again))


# ── Registry ─────────────────────────────────────────────────────────


def test_registry_keys_match() -> None:
    assert set(CANDLE_PATTERNS) == set(TALIB_NAMES) == set(EXPECTED_HITS)
    assert len(set(TALIB_NAMES.values())) == len(TALIB_NAMES)
    assert all(name.startswith("CDL") for name in TALIB_NAMES.values())


@pytest.mark.parametrize("name", sorted(CANDLE_PATTERNS))
def test_output_length_and_dtype(name: str, ohlc: OHLC) -> None:
    out = CANDLE_PATTERNS[name](*ohlc)
    assert out.dtype == np.int32
    assert out.shape == ohlc[0].shape
    assert set(np.unique(out).tolist()) <= {-200, -100, -80, 0, 80, 100, 200}


@pytest.mark.parametrize("name", sorted(CANDLE_PATTERNS))
@pytest.mark.parametrize("n", [0, 1, 5, 14, 15])
def test_short_inputs_return_zeros_or_valid(name: str, n: int, ohlc: OHLC) -> None:
    o, h, lo, c = (x[:n] for x in ohlc)
    out = CANDLE_PATTERNS[name](o, h, lo, c)
    assert out.shape == (n,)
    assert out.dtype == np.int32


def test_mismatched_lengths_raise() -> None:
    a = np.ones(20)
    with pytest.raises(ValueError):
        candles.cdl_doji(a, a, a, a[:-1])


@pytest.mark.parametrize("name", sorted(PENETRATION_PATTERNS))
def test_negative_penetration_raises(name: str, ohlc: OHLC) -> None:
    with pytest.raises(ValueError):
        CANDLE_PATTERNS[name](*ohlc, penetration=-0.1)


# ── TA-Lib parity ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def talib_mod() -> Any:
    return pytest.importorskip("talib")


def _talib_fn(talib_mod, name: str) -> Callable[..., np.ndarray]:
    return getattr(talib_mod, TALIB_NAMES[name])


@pytest.mark.parametrize("name", sorted(CANDLE_PATTERNS))
def test_matches_talib(name: str, ohlc: OHLC, talib_mod: Any) -> None:
    expected = _talib_fn(talib_mod, name)(*ohlc)
    ours = CANDLE_PATTERNS[name](*ohlc)
    assert ours.dtype == expected.dtype
    diff = np.flatnonzero(ours != expected)
    assert diff.size == 0, (
        f"{name}: {diff.size} mismatches, first rows {diff[:5].tolist()}, "
        f"ours {ours[diff[:5]].tolist()}, talib {expected[diff[:5]].tolist()}"
    )
    hits = int(np.count_nonzero(expected))
    assert hits >= MIN_HITS, f"{name} fired only {hits} times; fixture does not exercise it"
    if _fixture_is_recorded(ohlc):
        assert hits == EXPECTED_HITS[name]


@pytest.mark.parametrize("name", sorted(CANDLE_PATTERNS))
def test_recorded_hit_counts(name: str, ohlc: OHLC) -> None:
    # Runs without TA-Lib: the kernels must reproduce the recorded counts.
    if not _fixture_is_recorded(ohlc):
        pytest.skip("this NumPy produces a different random stream than the recorded fixture")
    assert int(np.count_nonzero(CANDLE_PATTERNS[name](*ohlc))) == EXPECTED_HITS[name]


@pytest.mark.parametrize("name", sorted(PENETRATION_PATTERNS))
@pytest.mark.parametrize("penetration", [0.0, 0.1, 0.25, 0.7, 1.3])
def test_matches_talib_penetration(
    name: str, penetration: float, ohlc: OHLC, talib_mod: Any
) -> None:
    expected = _talib_fn(talib_mod, name)(*ohlc, penetration=penetration)
    ours = CANDLE_PATTERNS[name](*ohlc, penetration=penetration)
    assert np.array_equal(ours, expected)


def test_penetration_changes_results(ohlc: OHLC) -> None:
    loose = candles.cdl_morning_star(*ohlc, penetration=0.0)
    tight = candles.cdl_morning_star(*ohlc, penetration=0.9)
    assert np.count_nonzero(loose) > np.count_nonzero(tight)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("scale", [1.0, 1.1, 0.37])
def test_matches_talib_other_series(seed: int, scale: float, talib_mod: Any) -> None:
    # Non-grid scalings exercise the running-total and fused multiply-add
    # replication on values that are not exactly representable.
    o, h, lo, c = (x * scale for x in make_ohlc(2000, seed))
    for name, fn in CANDLE_PATTERNS.items():
        assert np.array_equal(fn(o, h, lo, c), _talib_fn(talib_mod, name)(o, h, lo, c)), name


def test_matches_talib_with_nans(ohlc: OHLC, talib_mod: Any) -> None:
    o, h, lo, c = (x[:800].copy() for x in ohlc)
    o[:7] = np.nan
    h[:3] = np.nan
    c[300] = np.nan
    lo[450] = np.nan
    for name, fn in CANDLE_PATTERNS.items():
        assert np.array_equal(fn(o, h, lo, c), _talib_fn(talib_mod, name)(o, h, lo, c)), name


@pytest.mark.parametrize("name", sorted(CANDLE_PATTERNS))
@pytest.mark.parametrize("n", [0, 2, 3, 10, 11, 12, 13, 14, 15, 16])
def test_matches_talib_short(name: str, n: int, ohlc: OHLC, talib_mod: Any) -> None:
    o, h, lo, c = (np.ascontiguousarray(x[:n]) for x in ohlc)
    expected = _talib_fn(talib_mod, name)(o, h, lo, c)
    assert np.array_equal(CANDLE_PATTERNS[name](o, h, lo, c), expected)


def test_fused_multiply_add_fallback_is_exact() -> None:
    from fractions import Fraction

    rng = np.random.default_rng(5)
    for _ in range(2000):
        x, y, z = float(rng.uniform(0, 5)), float(rng.choice([0.3, 0.1, 0.7])), float(
            rng.uniform(10, 50)
        )
        exact = float(Fraction(x) * Fraction(y) + Fraction(z))
        assert candles._fma_scalar(x, y, z) == exact
        if hasattr(math, "fma"):
            assert exact == math.fma(x, y, z)


# ── Textbook examples (no reference library needed) ──────────────────


def _warmup(k: int = 12, start: float = 30.0, step: float = 0.0) -> List[Bar]:
    """Quiet bars: body 0.5, range 1.0, alternating colour."""
    bars: List[Bar] = []
    p = start
    for i in range(k):
        if i % 2 == 0:
            bars.append((p, p + 0.75, p - 0.25, p + 0.5))
        else:
            bars.append((p + 0.5, p + 0.75, p - 0.25, p))
        p += step
    return bars


def _arrays(bars: List[Bar]) -> OHLC:
    arr = np.asarray(bars, dtype=np.float64)
    return arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy(), arr[:, 3].copy()


def test_textbook_bullish_engulfing() -> None:
    bars = _warmup(4) + [(31.0, 31.1, 29.9, 30.0), (29.8, 31.3, 29.7, 31.2)]
    out = candles.cdl_engulfing(*_arrays(bars))
    assert out[-1] == 100
    assert out[-2] == 0


def test_textbook_bearish_engulfing() -> None:
    bars = _warmup(4) + [(30.0, 31.1, 29.9, 31.0), (31.2, 31.3, 29.7, 29.8)]
    assert candles.cdl_engulfing(*_arrays(bars))[-1] == -100


def test_textbook_doji() -> None:
    bars = _warmup(12) + [(30.0, 30.8, 29.2, 30.0)]
    out = candles.cdl_doji(*_arrays(bars))
    assert out[-1] == 100
    assert np.all(out[:-1] == 0)


def test_textbook_hammer_after_decline() -> None:
    bars = _warmup(12, start=36.0, step=-0.5)
    last_low = bars[-1][2]
    # Small body at the prior low, long lower shadow, no upper shadow.
    o = last_low + 0.1
    c = o + 0.2
    bars.append((o, c, o - 1.2, c))
    out = candles.cdl_hammer(*_arrays(bars))
    assert out[-1] == 100


def test_textbook_shooting_star() -> None:
    bars = _warmup(12)
    top = max(bars[-1][0], bars[-1][3])
    o = top + 0.3
    c = o + 0.1
    bars.append((o, c + 1.2, o, c))
    assert candles.cdl_shooting_star(*_arrays(bars))[-1] == -100


def test_textbook_morning_star() -> None:
    bars = _warmup(12)
    bars += [
        (31.0, 31.05, 28.95, 29.0),  # long black
        (28.6, 28.8, 28.4, 28.7),  # small star gapping down
        (28.9, 30.6, 28.85, 30.5),  # white closing well into the first body
    ]
    out = candles.cdl_morning_star(*_arrays(bars))
    assert out[-1] == 100
    # Requiring 90% penetration rejects the same bars.
    assert candles.cdl_morning_star(*_arrays(bars), penetration=0.9)[-1] == 0


def test_textbook_three_white_soldiers() -> None:
    bars = _warmup(12)
    bars += [
        (30.0, 30.82, 29.95, 30.8),
        (30.6, 31.62, 30.55, 31.6),
        (31.4, 32.42, 31.35, 32.4),
    ]
    assert candles.cdl_three_white_soldiers(*_arrays(bars))[-1] == 100


def test_textbook_harami_values() -> None:
    bars = _warmup(12) + [(32.0, 32.1, 29.9, 30.0), (30.8, 31.0, 30.6, 31.0)]
    out = candles.cdl_harami(*_arrays(bars))
    assert out[-1] == 100


def test_leading_rows_are_zero_before_lookback(ohlc: OHLC) -> None:
    out = candles.cdl_doji(*ohlc)
    assert np.all(out[:10] == 0)
    out = candles.cdl_mat_hold(*ohlc)
    assert np.all(out[:14] == 0)
