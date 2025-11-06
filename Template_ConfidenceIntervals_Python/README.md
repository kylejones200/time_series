# Confidence Intervals for Time Series Forecasts

Bootstrap and parametric confidence intervals for time series predictions.

## Features

- ✅ Bootstrap confidence intervals
- ✅ Parametric confidence intervals (ARIMA)
- ✅ Configurable confidence levels
- ✅ Visualization with uncertainty bands

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

- **method**: `bootstrap` or `parametric`
- **arima_order**: ARIMA parameters `[p, d, q]`
- **n_bootstrap**: Number of bootstrap samples (for bootstrap method)
- **alpha**: Significance level (default: 0.05 for 95% CI)

## Methods

### Bootstrap
- Resamples data to generate distribution of forecasts
- Non-parametric approach
- More computationally intensive

### Parametric
- Uses ARIMA model's built-in confidence intervals
- Faster computation
- Assumes model assumptions are met

## Outputs

- `outputs/confidence_intervals.png`: Forecast with confidence intervals

