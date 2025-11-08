# Chronos Transformer Forecasting

Template for running Amazon’s Chronos time-series models via the
`ChronosPipeline`. It follows the same config-driven structure used across the
repo.

## Features

- ✅ Loads time series from CSV (`config.yaml` controls columns and resampling)
- ✅ Uses `amazon/chronos-*` models from Hugging Face
- ✅ Configurable context length, prediction length, samples, dtype
- ✅ Produces forecast CSV + PNG with history, forecast, and 80% interval
- ✅ Saves evaluation metrics (MAE, RMSE, MAPE)

## Installation

```bash
pip install -r requirements.txt
```

> If the chosen Chronos model requires authentication, export
> `HF_TOKEN=<your-token>` before running.

## Usage

1. Place your time series in `data/` (default uses
   `amtrak_ridership_time_series_data.csv`).
2. Adjust `config.yaml` for column names, forecast horizon, model name, etc.
3. Run the template:

```bash
python main.py
```

The defaults aggregate ridership by year; modify the data-loading logic if you
need a different granularity.

Outputs are written to `outputs/`:

- `chronos_forecast.csv`
- `chronos_forecast.png`
- `chronos_metrics.yaml`

## Notes

- `torch_dtype: float32` works universally; `bfloat16`/`float16` need hardware
  support.
- `context_length` defaults to the last 256 observations; increase/decrease as
  needed but it must exceed `prediction_length`.
- The template uses simple train/test split (last `prediction_length` points as
  hold-out). Feel free to extend it with rolling backtests.
