# Exponential Smoothing for Time Series Forecasting

Classical exponential smoothing methods for time series forecasting using statsmodels.

## Features

- ✅ Simple Exponential Smoothing (SES)
- ✅ Double Exponential Smoothing (Holt's method)
- ✅ Triple Exponential Smoothing (Holt-Winters)
- ✅ Additive and multiplicative trend/seasonality
- ✅ Automatic parameter optimization option

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

- **Model type**: `simple`, `double`, or `triple`
- **Smoothing parameters**: Level, trend, and seasonal smoothing factors
- **Trend/Seasonal**: Additive (`add`) or multiplicative (`mul`)
- **Seasonal periods**: Number of periods in a season (default: 12)

## Model Types

- **Simple (SES)**: For data with no trend or seasonality
- **Double (Holt)**: For data with trend but no seasonality
- **Triple (Holt-Winters)**: For data with both trend and seasonality

## When to Use

- ✅ Short-term forecasting
- ✅ Data with clear trend and/or seasonality
- ✅ When interpretability is important
- ✅ Baseline model for comparison

## Outputs

- `outputs/exponential_smoothing_forecast.png`: Historical data, fitted values, and forecast

