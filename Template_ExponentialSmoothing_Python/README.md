# Exponential Smoothing (EIA Net Generation)

This template reproduces the ETS analyses from the 2025‑11‑08 article:
- Rolling-origin ETS evaluation (Jan–Aug 2025 focus)
- ETS vs SARIMAX last-fold comparison

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. The shared dataset lives at `data/Net_generation_United_States_all_sectors_monthly.csv`.
2. `config.yaml` controls the horizon, number of splits, and seasonal period.
3. Run:

```bash
python main.py
```

## Outputs

- `outputs/eia_expsmooth_last_fold.png` – ETS-only Tufte plot (history 2024, forecast Jan–Aug 2025).
- `outputs/eia_generation_last_fold.png` – ETS vs SARIMAX last-fold comparison.

Console output prints the rolling-origin MAE to match the numbers in the article. Adjust the config to explore different horizons or seasonalities.*** End Patch

