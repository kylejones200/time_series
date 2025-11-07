# PyTimeTK for Time Series Analysis

Time series toolkit for feature engineering, visualization, and analysis using pytimetk.

## Features

- ✅ Time series visualization (Plotly and Matplotlib)
- ✅ Rolling window statistics (mean, std, custom functions)
- ✅ Fourier terms for seasonality
- ✅ Lag features
- ✅ Time-based filtering
- ✅ Feature engineering pipeline

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your data file in the shared `data/` directory
2. Update `config.yaml` with your data file name and column names
3. Run:

```bash
python main.py
```

## Configuration

Edit `config.yaml` to customize:

- **data**:
  - `filter_by_time`: Whether to filter by date range
  - `start_date` and `end_date`: Date range for filtering
- **features**:
  - `rolling_windows`: List of window sizes for rolling statistics
  - `fourier_terms`: Whether to add Fourier terms
  - `fourier_K`: Number of Fourier terms
  - `lags`: List of lag periods
- **plotting**:
  - `engine`: `"matplotlib"` or `"plotly"`

## Features Created

### Rolling Statistics
- Rolling mean and standard deviation
- Configurable window sizes
- Can add custom rolling functions

### Fourier Terms
- Seasonal decomposition
- Multiple frequencies
- Configurable number of terms

### Lag Features
- Past values at specified lags
- Useful for forecasting models

## Outputs

- `outputs/pytimetk_timeseries.png` (or `.html` for Plotly): Time series plot
- `outputs/pytimetk_features.png`: Feature visualizations
- `outputs/features.csv`: Generated features (if `save_features: true`)

## Comparison with TSFresh

- **pytimetk**: More focused on time series manipulation and visualization
- **TSFresh**: More focused on automated feature extraction
- Both useful for different purposes
- pytimetk has better time series-specific utilities

## Notes

- Great for exploratory data analysis
- Fast feature engineering
- Supports both static and interactive plots
- Can be used as preprocessing step for ML models
- Integrates well with pandas workflows

