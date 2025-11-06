# Bayesian Time Series: PyMC

Bayesian time series modeling using PyMC for probabilistic forecasting.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- **Data**: Input file path and column names
- **Model**: Model type (AR1, RandomWalk, LinearTrend) and MCMC parameters
- **Plotting**: Visualization styling
- **Output**: Plot settings

## Run

```bash
python main.py
```

## Model Types

- **AR1**: Autoregressive model of order 1
- **RandomWalk**: Gaussian random walk model
- **LinearTrend**: Linear trend model

## Outputs

Forecast plots with credible intervals saved to `outputs/` directory.

