# Moirai Forecasting Template (EIA Net Generation)

This template reproduces the Moirai forecast from the 2025‑11‑08 article, generating the Jan–Aug 2025 greyscale figure and matching the reported numbers.

## Installation

```bash
pip install -r requirements.txt
```

> Moirai downloads from Hugging Face (`Salesforce/moirai-1.0-R-small`). Authenticate if the checkpoint requires it.

## Usage

1. Ensure the shared dataset `data/Net_generation_United_States_all_sectors_monthly.csv` exists.
2. `config.yaml` controls everything:
   - `experiment.history_end`, `forecast_start`, `forecast_end`
   - Moirai checkpoint and `context_length`, `horizon`, `num_samples`
3. Run:

```bash
python main.py
```

## Outputs

- `outputs/eia_moirai_last_fold.png` – Tufte-style overlay with history, actuals, and Moirai forecast.
- Console logs confirm the training window and forecast horizon used.

This matches the Moirai section in the article; tweak the config to explore other horizons or checkpoints.*** End Patch
