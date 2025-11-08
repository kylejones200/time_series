# Forecast Error Analysis (Seasonal Naive)

This template reproduces the seasonal-naive error diagnostics from the 2025‑11‑08 article, focusing on the Jan–Aug 2025 window.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Dataset: `data/Net_generation_United_States_all_sectors_monthly.csv`.
2. Adjust horizon, number of splits, or seasonal period in `config.yaml`.
3. Run:

```bash
python main.py
```

## Outputs

- `outputs/eia_errors_last_fold.png` – Seasonal naive last-fold overlay with history and forecast.
- Console summary of MAE/MAPE/SMAPE/MASE averaged across rolling-origin splits.

The numbers match the error table in the article; tweak the config to explore alternative horizons.*** End Patch
