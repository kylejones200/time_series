# LSTM Forecasting (EIA Net Generation)

This template recreates the LSTM rolling-origin evaluation from the 2025‑11‑08 article and outputs the Jan–Aug 2025 greyscale figure.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Dataset is shared across templates: `data/Net_generation_United_States_all_sectors_monthly.csv`.
2. Edit `config.yaml` if you’d like to adjust the LSTM chunk lengths, number of splits, or epochs.
3. Run:

```bash
python main.py
```

## Outputs

- `outputs/eia_lstm_last_fold.png` – history vs actuals vs LSTM forecast (Jan–Aug 2025) with a light uncertainty band.
- Console output prints the rolling-origin MAE to match the article.

