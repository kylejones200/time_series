# ARIMA Baselines (EIA Net Generation)

This template reproduces the classical baselines from the 2025‑11‑08 article:
- SARIMAX univariate vs calendar exogenous
- Linear calendar baseline
- ETS/SARIMAX ensemble
- Online SARIMAX streaming forecast

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Expected dataset: `data/Net_generation_United_States_all_sectors_monthly.csv`
2. `config.yaml` controls horizon, number of rolling splits, and seasonal period.
3. Run:

```bash
python main.py
```

## Generated Assets

- `outputs/eia_uni_vs_multi_last_fold.png`
- `outputs/eia_ba_baseline.png`
- `outputs/eia_ensemble_last_fold.png`
- `outputs/eia_streaming_last.png`

Console output reports the MAE for each scenario exactly as listed in the article.*** End Patch
