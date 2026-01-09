# Reference Implementation Setup

## Created Structure

 **Reference Script**: `reference_forecast.py`
- Loads time series (CSV with `date` and `value` columns)
- Splits into train/test (holds out last 20%)
- Fits ARIMA model
- Generates forecast
- Evaluates (RMSE metric)
- Saves plot and CSV to `outputs/reference/`

 **Shared Helpers in `src/`**:
- `src/loader.py`: Loads CSV with date and value columns → returns pd.Series with datetime index
- `src/model.py`: ARIMA model wrapper (fit, forecast, get_order methods)
- `src/evaluator.py`: Splits data and evaluates forecasts (RMSE metric)

 **Example Data**: `data/reference/example_series.csv`
- Format: `date,value`
- 120 daily data points (2020-01-01 to 2020-04-30)

## Ready to Run

The reference script is complete and ready to execute once dependencies are installed:

```bash
# Install dependencies (if needed)
pip install pandas numpy matplotlib pmdarima signalplot

# Run reference script
python reference_forecast.py
```

This will:
1. Load `data/reference/example_series.csv`
2. Fit ARIMA model
3. Generate forecast
4. Save results to `outputs/reference/`:
   - `forecast_plot.png` - Visualization
   - `forecast.csv` - Forecast values with confidence intervals
   - `metrics.csv` - RMSE metric

## Next Steps

1. Run the script to generate outputs
2. Commit the results (plot + CSV files)
3. Use this as the reference pattern for all other examples

## Extending

All other forecasting examples should follow this same structure:
- Use `src/loader.py` for loading data
- Use `src/model.py` wrappers (or create new ones with same interface)
- Use `src/evaluator.py` for evaluation
- Save results to `outputs/` folder

