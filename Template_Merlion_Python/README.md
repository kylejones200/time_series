# Merlion: Time Series Forecasting and Anomaly Detection

Unified framework for time series forecasting and anomaly detection with enhanced capabilities from anomaly_detection folder.

## Features

- ✅ Time series forecasting (Prophet, ARIMA)
- ✅ Anomaly detection (Isolation Forest, AutoEncoder)
- ✅ Comprehensive evaluation metrics
- ✅ Config-driven model selection

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

- **task**: `forecast` or `anomaly`
- **forecaster_type**: `Prophet` or `ARIMA`
- **detector_type**: `IsolationForest` or `AutoEncoder`
- **arima_order**: `[p, d, q]` parameters for ARIMA

## Tasks

### Forecasting
- **Prophet**: Facebook Prophet forecaster
- **ARIMA**: ARIMA model with configurable order

### Anomaly Detection
- **IsolationForest**: Isolation Forest anomaly detector
- **AutoEncoder**: Deep learning-based anomaly detection

## Outputs

- `outputs/merlion_forecast.png`: Forecast visualization with evaluation metrics
- `outputs/merlion_anomalies.png`: Anomaly detection visualization with anomaly statistics

