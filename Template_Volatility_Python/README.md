# Volatility Models (ARCH/GARCH)

Volatility forecasting using ARCH, GARCH, and EGARCH models for financial time series.

## Features

- ✅ **ARCH**: Autoregressive Conditional Heteroskedasticity
- ✅ **GARCH**: Generalized ARCH
- ✅ **EGARCH**: Exponential GARCH
- ✅ Multiple distribution support (Normal, Student-t, Skew-t)
- ✅ Conditional volatility estimation
- ✅ Volatility forecasting with confidence intervals

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your data file in the shared `data/` directory
2. Update `config.yaml` with your data file name and column names
3. Run:

```bash
python main.py
```

## Configuration

Edit `config.yaml` to customize:

- **data**:
  - `compute_returns`: Whether to compute returns from prices
  - `test_size`: Test set size for evaluation
- **model**:
  - `type`: `"ARCH"`, `"GARCH"`, or `"EGARCH"`
  - `p`: ARCH order
  - `q`: GARCH order (ignored for ARCH)
  - `distribution`: `"normal"`, `"t"`, or `"skewt"`
  - `forecast_horizon`: Steps ahead to forecast

## Models

### ARCH
- Models volatility as function of past squared errors
- Good for short-term volatility clustering
- Order `p` specifies lag length

### GARCH
- Extends ARCH with moving average component
- More parsimonious than ARCH
- Order `(p, q)` specifies ARCH and GARCH lags

### EGARCH
- Allows asymmetric volatility responses
- Log-volatility specification
- Better for leverage effects

## Outputs

- `outputs/volatility_forecast.png`: Four-panel plot showing:
  - Time series returns
  - Conditional volatility (fitted)
  - Forecasted variance
  - Forecasted volatility

## Notes

- Best for financial returns data
- Requires stationary returns (use `compute_returns: true`)
- GARCH(1,1) is most common specification
- EGARCH captures asymmetric volatility responses
- Student-t distribution handles fat tails

