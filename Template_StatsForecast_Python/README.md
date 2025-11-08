# StatsForecast AutoARIMA

Use Nixtla’s `statsforecast` AutoARIMA model for quick forecasting.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your data in `data/` (default: `amtrak_ridership_time_series_data.csv`).
2. Adjust `config.yaml` for date/value columns, frequency, and horizon.
3. Run:

```bash
python main.py
```

Outputs (`outputs/`):

- `statsforecast_forecast.csv`
- `statsforecast_metrics.yaml`
- `statsforecast_forecast.png`

## Notes

- `freq` uses pandas offset aliases (`'H'`, `'D'`, `'M'`, `'A'`, etc.)
- Ensure the season length matches your data (e.g., 24 for hourly daily seasonality).
