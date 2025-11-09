# ARAR: Autoregressive Autoregressive

Config-driven ARAR forecasting with ERCOT load data, rolling evaluation, and ARIMA comparison.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

`config.yaml` controls every step:

- **data**  
  - `input_file`: CSV in `../data/` (defaults to `ercot_load_data.csv`)  
  - `date_col`, `value_col`: timestamp and target columns  
  - `resample`: optional frequency aggregation (`enabled`, `freq`, `method`)
- **model**  
  - `train_size`: fallback split ratio if horizon is unavailable  
  - `differenced`: toggle ARAR differencing  
  - `lag_method`: `powers_of_2`, `custom`, or `auto` (ACF-based)  
  - `custom_lags`, `max_lag`, `forecast_horizon`
- **evaluation**  
  - `horizon`: hold-out size (e.g., 96 × 15-minute steps)  
  - `compare_arima`: enable ARIMA baseline and specify `order`
- **plotting**  
  - Unified styling for all figures
- **output**  
  - File format, destination directory, and plot filenames

## Run

```bash
python main.py
```

The script will:
1. Load and optionally resample the series.
2. Fit ARAR with a reduced lag set and forecast the hold-out window.
3. (Optionally) fit an ARIMA baseline for side-by-side comparison.
4. Write metrics to `outputs/metrics.yaml`.
5. Save three plots under `outputs/`:
   - `arar_series_visualization.png` – original vs. differenced
   - `arar_forecast_vs_actual.png` – hold-out evaluation
   - `arar_vs_arima_forecast.png` – full-series comparison

## Assets

- `notebooks/ARAR with ERCOT data.ipynb`: original exploratory notebook
- `outputs/reference/`: sample figures generated prior to this refactor

Adapt the config to point at new datasets or tweak lags, then re-run `python main.py` for reproducible ARAR experiments.