# N-BEATS Forecasting (EIA Net Generation)

This template recreates the N-BEATS rolling-origin evaluation from the 2025‑11‑08 article and generates the Jan–Aug 2025 Tufte figure.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Dataset: `data/Net_generation_United_States_all_sectors_monthly.csv`.
2. Tweak `config.yaml` for horizon, number of splits, or chunk lengths.
3. Run:

```bash
python main.py
```

## Outputs

- `outputs/eia_nbeats_last_fold.png` – Tufte-style overlay with history, actuals, and the N-BEATS forecast.
- Console output logs the rolling-origin MAE so you can match the article’s numbers.
