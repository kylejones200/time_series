# Irregular Time Series Handling

Simulate irregularly sampled data, resample to a regular grid, perform linear
interpolation, and build a Gaussian Process interpolator.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Outputs in `outputs/`:

- `irregular_time_series.png` — plot showing original points, resampled/interpolated series, and GP mean/interval
- `resampled.csv`, `interpolated.csv`, `gaussian_process.csv` — processed data for further analysis

## Configuration

`config.yaml` controls:

- simulation start timestamp, base frequency, number of points, and missingness probability
- resample rule for regularization
- Gaussian Process kernel length-scale

Adjust the settings to match your own irregular dataset if needed.
