# Sundial Transformer Forecasting

Forecast time series using the Sundial transformer model from THUML. The template fetches FRED data, prepares input windows, and generates probabilistic forecasts with uncertainty bounds.

## Features

- ✅ Fetches data directly from FRED (`pandas_datareader`)
- ✅ Normalizes series and builds rolling windows
- ✅ Runs the `thuml/sundial-base-128m` transformer
- ✅ Generates multiple stochastic samples for uncertainty
- ✅ Saves forecast mean and 80% intervals to CSV
- ✅ Minimalist matplotlib visualization with history + forecast
- ✅ Optional comparison with held-out actual values

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** Access to the Sundial model on Hugging Face may require an access token. Export `HF_TOKEN` in your environment if needed.

## Usage

1. Place or configure your data fetch in `config.yaml` (default uses FRED ID `DCOILWTICO`).
2. (Optional) Set `HF_TOKEN` environment variable if the model requires authentication.
3. Run the template:

```bash
python main.py
```

## Configuration

Edit `config.yaml` to customize:

- **data**
  - `series_id`: FRED series identifier
  - `start_date`: Earliest date to fetch
  - `resample_rule`: Pandas resample rule (e.g., `"D"`, `"M"`, or `null`)
- **model**
  - `lookback_length`: Input context length
  - `forecast_length`: Forecast horizon
  - `num_samples`: Number of stochastic samples
  - `generation`: Sampling parameters (`temperature`, `top_p`, etc.)
- **plotting**
  - `history_window`: Length of historical window to display
- **evaluation**
  - `compare_to_actual`: Compare forecast mean to held-out ground truth

## Outputs

- `outputs/sundial_forecast.csv`: Forecast mean and 80% confidence interval
- `outputs/sundial_forecast.png`: Plot with history, forecast, and interval

## Notes

- Sundial expects normalized input; the script standardizes and reverts scaling automatically.
- The model is autoregressive; we rely on `max_new_tokens` to extend the series.
- Increase `num_samples` for smoother intervals (runtime increases linearly).
- GPU support is not required but will speed up generation if available.
