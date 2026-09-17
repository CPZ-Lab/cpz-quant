# Indicators

`cpz_quant.indicators` provides vectorised technical indicators over NumPy arrays and Polars data, organised by family. An optional Rust extension accelerates a few hot paths; without it, pure-Python/NumPy implementations run automatically.

## Families

| Module | Indicators |
|---|---|
| `trend` | SMA, EMA, WMA, DEMA, TEMA, KAMA, Supertrend, TRIMA, T3, ZLEMA, HMA, ALMA, VIDYA, McGinley Dynamic, Parabolic SAR, Ichimoku, Aroon, Aroon Oscillator, Vortex, TRIX, Mass Index, Midpoint, Midprice, Schaff Trend Cycle |
| `momentum` | RSI, MACD, Stochastic (fast and slow), Stochastic RSI, Williams %R, CCI, ROC, ROCP, ROCR / ROCR100, MOM, momentum %, ADX, DMI (+DM, -DM, +DI, -DI, DX, ADX, ADXR), MFI, CMO, Ultimate Oscillator, Awesome Oscillator, APO, PPO, KST, TSI, Connors RSI, Fisher Transform, Coppock Curve, DPO, Relative Vigor Index, Elder Ray, WaveTrend, Balance of Power |
| `volatility` | True Range, ATR, NATR, Bollinger Bands, Keltner Channels, Donchian Channels, close-to-close realized volatility, Garman-Klass, Parkinson, Rogers-Satchell, Yang-Zhang, Chaikin Volatility, Ulcer Index, Choppiness Index |
| `volume` | VWAP (cumulative or session-anchored), VWMA, OBV, Chaikin Money Flow, Accumulation/Distribution Line, Chaikin A/D Oscillator, Force Index, Ease of Movement, Klinger Volume Oscillator, NVI, PVI, Price Volume Trend, Percentage Volume Oscillator, volume z-score |
| `statistical` | rolling z-score, rolling correlation, rolling beta, Hurst exponent, linear regression (slope, intercept, R^2, forecast, end value, angle), Kaufman efficiency ratio, rolling Sharpe, rolling Sortino |
| `candles` | candlestick pattern recognition; see [Candlestick patterns](#candlestick-patterns) |

## Usage

```python
import polars as pl
import cpz_quant.indicators as ind

bars = pl.DataFrame({"open": ..., "high": ..., "low": ..., "close": ..., "volume": ...})

ind.hma(bars, 20)                     # latest value (float)
ind.hma(bars, 20, series=True)        # full pl.Series with leading NaN warm-up
ind.dmi(bars, 14).adx                 # named result for multi-output indicators
ind.vwap(bars, session_col="ts")      # VWAP that restarts every trading date
```

The numpy kernels live in the family modules and return arrays aligned to the input:

```python
import cpz_quant.indicators.trend as trend
import cpz_quant.indicators.momentum as momentum

sar = trend.psar_series(high, low)
k, d = momentum.stochrsi_series(close)
```

Note that the `momentum` *function* at the package root shadows the `momentum` module, so import submodules directly as above when you want the numpy kernels.

## Conventions

- Warm-up bars are NaN. Nothing is back-filled or fabricated; an undefined ratio (for example a Sortino ratio with no downside observations) is NaN.
- Exponential smoothers in the indicators added after 1.1.0 are seeded with the simple mean of their first `period` valid inputs, the TA-Lib and TradingView convention. The older `ema`, `dema`, `tema`, `macd`, `atr` and `adx` seed from the first bar instead and emit no or shorter warm-up; they converge to TA-Lib once the seed decays, and the test suite checks that. `dmi()` provides a TA-Lib-exact ADX.
- Where published definitions disagree, the docstring names the choice. Examples: Ichimoku spans are displaced forward (no look-ahead) and the Chikou span is not returned; DPO is not centred (centring uses future prices); PPO defaults to EMAs although TA-Lib's own default is SMA; Klinger uses the simplified signed-volume form; Connors RSI's percent rank counts strictly smaller values.

## Verification

Correctness is tested numerically on a frozen, seeded synthetic OHLCV series (`tests/fixtures/indicator_reference.npz`):

- **TA-Lib 0.8.0** (a dev dependency, compared live with identical NaN warm-up and `rtol = atol = 1e-9`): TRIMA, T3, ZLEMA, HMA, SAR, AROON, AROONOSC, VORTEX, TRIX, MASSI, MIDPOINT, MIDPRICE, STOCH, STOCHRSI, CMO, CMOU, MOM, ROCP, ROCR, ROCR100, ULTOSC, AO, APO, PPO, TSI, COPPOCK, DPO, ERI, BOP, PLUS_DM, MINUS_DM, PLUS_DI, MINUS_DI, DX, ADX, ADXR, CVI, AD, ADOSC, VWMA, EFI, NVI, PVI, PVT, PVO, VWAP, LINEARREG, LINEARREG_ANGLE, ER, and the existing SMA, WMA, RSI, WILLR, CCI, ROC, MFI, BBANDS, LINEARREG_SLOPE, LINEARREG_INTERCEPT, TSF, CORREL, BETA. All candlestick patterns are compared to TA-Lib's `CDL*` functions.
- **pandas-ta 0.4.71b0** (outputs frozen in the fixture, re-derived when pandas-ta is installed): ALMA, Ichimoku, KST, TSI signal, Fisher Transform, Relative Vigor Index, Ulcer Index, Choppiness Index, Ease of Movement, Klinger, PVO signal.
- **Independent formula transcriptions** of the published definitions: McGinley Dynamic, VIDYA, Schaff Trend Cycle, Connors RSI, WaveTrend, Parkinson, Rogers-Satchell, Yang-Zhang, rolling Sharpe and Sortino, anchored VWAP.
- **Known-answer properties**: range volatility estimators recover the sigma of a simulated diffusion; Yang-Zhang includes overnight variance; Ichimoku and DPO use no future data.

## Candlestick patterns

`cpz_quant.indicators.candles` implements all 61 TA-Lib candlestick functions (`CDL2CROWS` through `CDLXSIDEGAP3METHODS`), and each one is tested for element-by-element equality with TA-Lib, including penetration overrides, NaN-prefixed input and very short series. The kernels replicate TA-Lib's candle settings (body and shadow averages over the preceding candles), its running-total arithmetic and its fused multiply-add comparisons, so results agree even when a value sits exactly on a threshold.

```python
ind.candle_pattern(bars, "engulfing")                    # latest bar: 100, -100, 80, -80 or 0
ind.candle_pattern(bars, "morning_star", penetration=0.5, series=True)
ind.candle_patterns(bars)                                # DataFrame, one Int32 column per pattern
```

Outputs follow TA-Lib: `+100` bullish, `-100` bearish, `0` none, `+/-80` for engulfing, harami and harami cross when a body edge is equal, `+/-200` for a confirmed hikkake. Tweezer tops and bottoms are not included because TA-Lib has no reference definition for them.

Pattern names (`cpz_quant.indicators.CANDLE_PATTERN_NAMES`; the TA-Lib name for each is in `cpz_quant.indicators.candles.TALIB_NAMES`):

`abandoned_baby`, `advance_block`, `belt_hold`, `breakaway`, `closing_marubozu`, `concealing_baby_swallow`, `counterattack`, `dark_cloud_cover`, `doji`, `doji_star`, `dragonfly_doji`, `engulfing`, `evening_doji_star`, `evening_star`, `gap_side_side_white`, `gravestone_doji`, `hammer`, `hanging_man`, `harami`, `harami_cross`, `high_wave`, `hikkake`, `hikkake_modified`, `homing_pigeon`, `identical_three_crows`, `in_neck`, `inverted_hammer`, `kicking`, `kicking_by_length`, `ladder_bottom`, `long_legged_doji`, `long_line`, `marubozu`, `mat_hold`, `matching_low`, `morning_doji_star`, `morning_star`, `on_neck`, `piercing`, `rickshaw_man`, `rise_fall_three_methods`, `separating_lines`, `shooting_star`, `short_line`, `spinning_top`, `stalled_pattern`, `stick_sandwich`, `takuri`, `tasuki_gap`, `three_black_crows`, `three_inside`, `three_line_strike`, `three_outside`, `three_stars_in_south`, `three_white_soldiers`, `thrusting`, `tristar`, `two_crows`, `unique_three_river`, `upside_gap_two_crows`, `xside_gap_three_methods`.
