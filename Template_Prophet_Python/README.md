# Prophet: Facebook's Time Series Forecasting

Automatic forecasting procedure for business time series using Prophet.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- **Data**: Input file path and column names
- **Model**: Forecast horizon, seasonality settings, growth mode
- **Plotting**: Visualization styling
- **Output**: Plot settings

## Run

```bash
python main.py
```

## Features

- Automatic seasonality detection
- Handles holidays and events
- Robust to missing data
- Prediction intervals

## Outputs

Forecast plots with uncertainty intervals saved to `outputs/` directory.

