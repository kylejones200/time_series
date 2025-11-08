# TimesFM Forecasting Template (EIA Net Generation)

This template reproduces the TimesFM forecast from the 2025‑11‑08 article, generating the greyscale Tufte figure for Jan–Aug 2025 and printing the MAE/MAPE used in the write-up.

## Installation

```bash
pip install -r requirements.txt
```

> TimesFM runs on CPU out of the box. If you have GPU support, adjust the requirements accordingly.

## Usage

1. Ensure `data/Net_generation_United_States_all_sectors_monthly.csv` is available (already shared across templates).
2. `config.yaml` exposes:
   - `experiment.history_end` / `forecast_start` / `forecast_end`
   - TimesFM checkpoint (default `google/timesfm-1.0-200m-pytorch`)
   - Context length, horizon length, backend, and batch size
3. Run:

```bash
python main.py
```

## Outputs

- `outputs/eia_timesfm_last_fold.png` – Tufte-style Jan–Aug 2025 overlay (history, actuals, TimesFM).
- Console metrics: MAE and MAPE for the forecast window.

Metrics and the plot match the figures cited in the article; tweak the config to explore other horizons or checkpoints.*** End Patch
