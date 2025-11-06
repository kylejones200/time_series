# Box-Jenkins Methodology for ARIMA Modeling

Systematic approach to ARIMA model identification, estimation, and diagnostics following the Box-Jenkins methodology.

## Features

- ✅ Automatic differencing order detection (ADF test)
- ✅ ACF/PACF plots for model identification
- ✅ Automatic ARIMA parameter selection (auto_arima)
- ✅ Manual ARIMA parameter specification
- ✅ Ljung-Box test for residual diagnostics
- ✅ Forecast with confidence intervals

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

- **use_auto_arima**: Use automatic parameter selection (recommended)
- **Manual parameters**: Set `p`, `d`, `q` if `use_auto_arima` is false
- **Search ranges**: `start_p`, `start_q`, `max_p`, `max_q` for auto_arima

## Box-Jenkins Methodology

1. **Identification**: Determine differencing order (d) and AR/MA orders (p, q)
2. **Estimation**: Fit the ARIMA model
3. **Diagnostics**: Check residuals for autocorrelation (Ljung-Box test)
4. **Forecasting**: Generate forecasts with confidence intervals

## When to Use

- ✅ Systematic approach to ARIMA modeling
- ✅ When you need to understand the modeling process
- ✅ Educational purposes
- ✅ Baseline for comparison with other methods

## Outputs

- `outputs/box_jenkins_diagnostics.png`: Original series, differenced series, ACF, PACF
- `outputs/box_jenkins_forecast.png`: Historical data, actual test values, forecast with CI

