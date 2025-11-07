# Bayesian Structural Time Series (BSTS) with pybsts

Bayesian Structural Time Series forecasting using the pybsts library. Alternative implementation to Orbit.

## Features

- ✅ Bayesian state space models
- ✅ Local level and trend components
- ✅ Seasonal components
- ✅ Autoregressive components
- ✅ MCMC sampling for uncertainty quantification
- ✅ Forecast confidence intervals
- ✅ Multiple distribution support (Gaussian, Poisson, Logit)

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

- **model**:
  - `distribution`: `"gaussian"`, `"poisson"`, or `"logit"`
  - `ar_order`: Autoregressive order (0-3)
  - `local_level`: Include local level component
  - `local_slope`: Include local slope component
  - `seasonal_period`: Seasonal period (null = no seasonality)
  - `niter`: Number of MCMC iterations
  - `burn`: Burn-in iterations
  - `forecast_horizon`: Steps ahead to forecast
  - `ping`: Print progress every N iterations

## Model Components

### Local Level
- Captures slowly varying mean
- Good for trend without slope

### Local Slope
- Captures trend with changing slope
- Good for linear trends

### Seasonal
- Captures periodic patterns
- Specify `seasonal_period` (e.g., 24 for hourly daily pattern)

### Autoregressive
- Captures short-term dependencies
- `ar_order` specifies lag order

## Outputs

- `outputs/pybsts_forecast.png`: Forecast plot with:
  - Historical data (train/test)
  - Forecast mean
  - 95% confidence intervals

## Comparison with Orbit

- **pybsts**: C++ backend, faster for large datasets
- **Orbit**: More features, better documentation, Python-native
- Both implement Bayesian Structural Time Series
- Choose based on your needs and preferences

## Notes

- More iterations (`niter`) = better estimates but slower
- Burn-in (`burn`) should be ~10% of iterations
- Seasonal period should match your data frequency
- Gaussian distribution is most common
- Poisson for count data, Logit for binary data

