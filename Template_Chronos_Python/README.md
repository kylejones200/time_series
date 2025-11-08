# Chronos Transformer Forecasting (EIA Net Generation)

This template mirrors the setup used in the 2025‑11‑08 Chronos article, forecasting US EIA net generation for Jan–Aug 2025 with the `amazon/chronos-t5-tiny` pipeline.

## Features

- ✅ Loads the monthly EIA series defined in `config.yaml`
- ✅ Configurable Chronos model, context length, sampling, and dtype
- ✅ Generates both the publication Tufte-style plot and the standard forecast view
- ✅ Saves forecast CSV plus MAE/RMSE/MAPE metrics

## Installation

```bash
pip install -r requirements.txt
```

> If the chosen Chronos model requires authentication, export `HF_TOKEN=<your-token>` before running.

## Usage

1. Ensure `data/Net_generation_United_States_all_sectors_monthly.csv` is available (default path in `config.yaml`).
2. Adjust `prediction_length`, `context_length`, or switch models via `config.yaml`.
3. Run:

```bash
python main.py
```

Outputs land in `outputs/`:

- `chronos_forecast.csv`
- `chronos_metrics.yaml`
- `chronos_forecast.png`
- `eia_chronos_last_fold.png`

## Notes

- `torch_dtype: float32` is a safe default; half-precision dtypes need GPU/CPU support.
- The script compares the Chronos median forecast against the last eight observed months (Jan–Aug 2025). Adjust the window via `plotting.tufte` in `config.yaml`.
