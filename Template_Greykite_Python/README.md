# Greykite: Forecasting Library

LinkedIn's Greykite for flexible, powerful time series forecasting.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- **Data**: Input file path, column names, and optional regressors
- **Model**: Forecast horizon, coverage, growth term, seasonality settings
- **Plotting**: Visualization styling
- **Output**: Plot settings

## Run

```bash
python main.py
```

## Features

- Automatic seasonality detection
- Support for regressors
- Prediction intervals
- Multiple growth terms

## Outputs

Forecast plots with prediction intervals saved to `outputs/` directory.

