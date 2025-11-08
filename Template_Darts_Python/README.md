# Darts: EIA Net Generation Forecasting

This template reproduces the figures and evaluation pipeline from the *Darts for Time Series Analysis in Python* article (2025‑11‑08). It evaluates multiple Darts models on the EIA net generation series and recreates the publication-ready plots.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

All parameters live in `config.yaml`:
- **data**: CSV location (`data/Net_generation_United_States_all_sectors_monthly.csv`) and column names.
- **evaluations**: Rolling-origin settings for the Tufte-style Jan–Aug 2025 comparison and the overview plot.
- **models**: Model roster per evaluation group (ARIMA, Theta, ExponentialSmoothing, NaiveSeasonal).
- **output**: File names written to `outputs/`.

## Run

```bash
python main.py
```

## Generated Assets

Running the template produces:
- `outputs/eia_darts_tbats_last_fold.png` – Tufte-style Jan–Aug 2025 comparison with optional TBATS overlay (`Template_Darts_Python/data/eia_preds_tbats.csv`).
- `outputs/eia_darts_overview_last_fold.png` – Overview of last-fold forecasts for ETS and NaiveSeasonal.
- `outputs/eia_preds_darts.csv` – ARIMA and Theta last-fold predictions aligned with ground truth.

Metrics for each model are printed to the console. Adjust `config.yaml` to swap models, horizons, or evaluation windows.*** End Patch

