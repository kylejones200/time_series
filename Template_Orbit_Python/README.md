# Orbit: Bayesian Time Series Forecasting

Bayesian structural time series models for forecasting using Orbit.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- **Data**: Input file path and column names
- **Model**: Model type (DLT, KTR, LGT) and parameters
- **Plotting**: Visualization styling
- **Output**: Plot settings

## Run

```bash
python main.py
```

## Model Types

- **DLT**: Dynamic Linear Trend
- **KTR**: Kernel Trend Regression
- **LGT**: Local Global Trend

## Outputs

Forecast plots saved to `outputs/` directory.

