# Indicators

`cpz_quant.indicators` provides vectorised technical indicators over NumPy arrays and Polars series, organised by family. An optional Rust extension accelerates hot paths; without it, identical pure-Python/NumPy implementations run automatically.

## Families

| Module | Indicators |
|---|---|
| `trend` | SMA, EMA, WMA, DEMA, TEMA, KAMA, Supertrend |
| `momentum` | RSI, MACD, Stochastic, Williams %R, CCI, ROC, ADX, MFI |
| `volatility` | True Range, ATR, NATR, Bollinger Bands, Keltner Channels, Donchian Channels, realized volatility, Garman-Klass |
| `volume` | VWAP, OBV, Chaikin Money Flow, volume z-score |
| `statistical` | rolling z-score, rolling correlation, rolling beta, Hurst exponent, linear regression |

## Usage

```python
import numpy as np
import cpz_quant.indicators.trend as trend
import cpz_quant.indicators.momentum as momentum

close = np.asarray(prices)
sma20 = trend.sma_series(close, 20)
rsi14 = momentum.rsi_series(close, 14)
```

Series functions return arrays aligned to the input (leading NaN warm-up), so they drop straight into Polars or NumPy pipelines.

The package root also exposes convenience wrappers; note that the `momentum` *function* shadows the `momentum` module at the package root, so import submodules directly as above when you want the full family.
