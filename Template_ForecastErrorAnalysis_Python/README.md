# Forecast Error Analysis

Compute classic forecast error diagnostics using an ETS (seasonal exponential
smoothing) model. Mimics the exploratory work from the `Forecast Error Analysis`
notebook.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The default configuration pulls the Airline Passengers dataset from GitHub and
uses monthly seasonality. To run:

```bash
python main.py
```

Outputs in `outputs/`:

- `forecast_error_analysis.png` — actual vs fitted, errors, exponentially smoothed errors
- `forecast_error_metrics.yaml` — summary statistics (MAE, RMSE, MAPE, variance, etc.)

## Configuration

`config.yaml` controls:

- **data**: provide either a URL or local CSV (`data/your_file.csv`)
- **model**: seasonal period for ETS
- **analysis**: moving average window and exponential smoothing alpha used in diagnostics
