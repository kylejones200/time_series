# Seasonal Decomposition (EIA Net Generation)

This template recreates the decomposition visuals from the 2025‑11‑08 article: full additive decomposition plus the seasonal subseries plot.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Dataset: `data/Net_generation_United_States_all_sectors_monthly.csv`
2. Set the seasonal period in `config.yaml` (default 12 months)
3. Run:

```bash
python main.py
```

## Outputs

- `outputs/eia_patterns.png` – observed/trend/seasonal/residual panels
- `outputs/eia_seasonal_subseries.png` – month-by-month seasonal subseries lines

These match the decomposition section from the article; adjust the period if you want to explore different seasonalities.*** End Patch

