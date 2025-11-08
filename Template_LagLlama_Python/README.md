# Granite TTM Forecasting Template (EIA Net Generation)

This template reproduces the IBM Granite Time Series TTM forecast from the 2025‑11‑08 article, creating the Jan–Aug 2025 Tufte figure.

## Installation

```bash
pip install -r requirements.txt
```

> Granite downloads from Hugging Face (`ibm-granite/granite-timeseries-ttm-r2`). Authenticate if necessary.

## Usage

1. Verify `data/Net_generation_United_States_all_sectors_monthly.csv` exists.
2. Adjust `config.yaml` (context length, horizon, checkpoint) if desired.
3. Run:

```bash
python main.py
```

## Outputs

- `outputs/eia_granite_ttm_last_fold.png` – greyscale overlay with history, actuals, and Granite forecast.

The script mirrors the article’s setup; tweak the config to explore other contexts or horizons.*** End Patch
