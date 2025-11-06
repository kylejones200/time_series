# Vector Autoregression (VAR) for Multivariate Time Series

VAR modeling for multiple interdependent time series using statsmodels.

## Features

- ✅ Automatic stationarity testing and differencing
- ✅ Optimal lag selection (AIC/BIC)
- ✅ Granger causality tests
- ✅ Durbin-Watson test for residual diagnostics
- ✅ Forecast multiple series simultaneously

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your data file in the shared `data/` directory
2. Update `config.yaml` with your data file name, date column, and **list of value columns**
3. Run:

```bash
python main.py
```

## Configuration

Edit `config.yaml` to customize:

- **value_cols**: List of column names for multivariate time series (e.g., `["var1", "var2", "var3"]`)
- **max_lags**: Maximum number of lags to consider (default: 15)
- **use_aic**: Use AIC for lag selection (true) or BIC (false)
- **granger_test**: Perform Granger causality tests between variables
- **force_differencing**: Force differencing even if series appear stationary

## When to Use VAR

- ✅ Multiple interdependent time series
- ✅ When variables may influence each other
- ✅ Economic/financial data with multiple indicators
- ✅ When you need to forecast multiple series together

## VAR vs Univariate Models

- **VAR**: Models relationships between multiple series
- **ARIMA**: Models a single series independently
- **VAR**: Captures cross-variable dependencies
- **ARIMA**: Simpler, faster, for single series

## Outputs

- `outputs/var_forecast.png`: Historical data, actual test values, and forecasts for all variables

## Notes

- VAR requires all series to be stationary (differencing applied automatically if needed)
- More variables = more parameters = need more data
- Granger causality tests help identify which variables predict others

