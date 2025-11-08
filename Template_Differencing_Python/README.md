# Differencing for Stationarity

Visualize a time series alongside its first and second differences. Includes Augmented Dickey-Fuller (ADF) diagnostics to assess stationarity.

## Features

- ✅ Fetches data from a local CSV or remote URL
- ✅ Computes successive differences (up to configurable order)
- ✅ Runs ADF test after each differencing step
- ✅ Saves plots and diagnostics as CSV

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Optionally update `config.yaml` with your data source (local file or URL) and column names.
2. Run the analysis:

```bash
python main.py
```

## Configuration

`config.yaml` controls:

- **data**: `url` or `input_file`, column names, resampling rule
- **model**: number of differencing levels
- **plotting**: figure size, colors, labels

## Outputs

- `outputs/differencing_plot.png` — stacked plot of original series and differences with ADF stats
- `outputs/adf_results.csv` — table of ADF statistic, p-value, lags, observations

## Notes

- The default dataset is NASA GISTEMP global temperature anomalies.
- If you provide a local CSV, place it in the shared `data/` directory and set `url: null`.
- Differencing beyond second order rarely helps; adjust `max_difference_order` as needed.
