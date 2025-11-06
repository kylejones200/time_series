# Darts: Time Series Forecasting Library

Unified interface for multiple forecasting models using Darts.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- **Data**: Input file path and column names
- **Model**: Model type (ExponentialSmoothing, ARIMA, Prophet, NBEATS, etc.) and parameters
- **Plotting**: Visualization styling
- **Output**: Plot settings

## Run

```bash
python main.py
```

## Model Types

- **ExponentialSmoothing**: Exponential smoothing
- **ARIMA**: ARIMA model
- **Prophet**: Facebook Prophet
- **NBEATS**: Neural basis expansion
- **RandomForest**: Random Forest regressor
- **XGBModel**: XGBoost model
- **LightGBMModel**: LightGBM model

## Outputs

Forecast plots saved to `outputs/` directory.

