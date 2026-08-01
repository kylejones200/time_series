# Reference Forecasting Implementation

This is the **reference implementation** for all forecasting examples in this repository. It demonstrates the standard workflow that all other examples should follow.

## Structure

```
time_series/
├── src/                    # Shared helpers
│   ├── loader.py          # Load CSV with date and value columns
│   ├── model.py           # Model wrappers (ARIMA, etc.)
│   ├── evaluator.py       # Evaluation utilities
│   └── __init__.py
├── reference_forecast.py  # Reference script
├── data/
│   └── reference/
│       └── example_series.csv  # Example data with date and value columns
└── outputs/
    └── reference/         # Output folder
        ├── forecast_plot.png
        ├── forecast.csv
        └── metrics.csv
```

## Workflow

The reference script (`reference_forecast.py`) follows this standard workflow:

1. **Load** time series data from CSV (date and value columns)
2. **Split** into train/test sets (hold out last 20%)
3. **Fit** model (ARIMA in this case)
4. **Generate** forecast for test period
5. **Evaluate** forecast (calculate RMSE)
6. **Save** results:
   - Plot: `outputs/reference/forecast_plot.png`
   - Forecast CSV: `outputs/reference/forecast.csv`
   - Metrics CSV: `outputs/reference/metrics.csv`

## Usage

```bash
python reference_forecast.py
```

## Data Format

Input CSV must have two columns:
- `date`: Date column (YYYY-MM-DD format)
- `value`: Numeric value column

Example:
```csv
date,value
2020-01-01,100.0
2020-01-02,99.5
2020-01-03,100.2
...
```

## Components

### `src/loader.py`
Loads time series from CSV file with date and value columns.

```python
from src.loader import load_time_series

series = load_time_series("data/reference/example_series.csv")
```

### `src/model.py`
Model wrappers that provide a consistent interface:
- `fit(series)`: Fit model to time series
- `forecast(n_periods)`: Generate forecast
- `get_order()`: Get model parameters

```python
from src.model import ARIMAModel

model = ARIMAModel()
model.fit(train)
forecast, conf_int = model.forecast(n_periods=20, return_conf_int=True)
```

### `src/evaluator.py`
Evaluation utilities:
- `split(series)`: Split into train/test
- `evaluate(forecast, actual)`: Calculate metrics (RMSE)

```python
from src.evaluator import Evaluator

evaluator = Evaluator(test_size=0.2)
train, test = evaluator.split(series)
metrics = evaluator.evaluate(forecast, test)
print(f"RMSE: {metrics['RMSE']:.4f}")
```

## Output

The script saves:

1. **Plot** (`outputs/reference/forecast_plot.png`): Visualization with:
   - Historical training data
   - Actual test data
   - Forecast
   - 95% confidence intervals

2. **Forecast CSV** (`outputs/reference/forecast.csv`): Forecast values with confidence intervals

3. **Metrics CSV** (`outputs/reference/metrics.csv`): Evaluation metrics (RMSE)

## Extending

All other forecasting examples should follow this same structure:
- Use `src/loader.py` to load data
- Use `src/model.py` wrappers or create new ones following the same interface
- Use `src/evaluator.py` for evaluation
- Save plots and CSVs to `outputs/`

## Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `pmdarima` (for ARIMA)
- `signalplot` (for clean plotting)

