# TimesFM Forecasting Template

Run Google’s TimesFM foundation model on a univariate time series. The template
converts a CSV into the wide format expected by `forecast_on_df`, produces a
hold-out forecast, and saves metrics/plots.

## Installation

```bash
pip install -r requirements.txt
```

> TimesFM depends on JAX and accelerator libraries. CPU inference works but can
> be slower; adjust requirements if running on a different platform.

## Usage

1. Place your dataset in `data/` (default uses `amtrak_ridership_time_series_data.csv`).
2. Update `config.yaml` with:
   - `date_col`, `value_col`, `frequency`
   - Prediction horizon and TimesFM checkpoint (`google/timesfm-*`).
3. Run the template:

```bash
python main.py
```

Outputs are written to `outputs/`:

- `timesfm_forecast.csv` — actual vs forecast values
- `timesfm_metrics.yaml` — MAE/RMSE/MAPE
- `timesfm_forecast.png` — overlay plot of history vs TimesFM forecast

## Notes

- `forecast_column` defaults to `timesfm`; adjust if the library changes its
  output column naming.
- The template uses the last `prediction_length` points as a test window.
- Aggregation step groups rows by `date_col` and sums `value_col`; customize if
  your dataset already has unique timestamps.
