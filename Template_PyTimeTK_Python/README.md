# PyTimeTK Visualisations (EIA Net Generation)

This template outputs the quick YoY/overview visuals from the 2025‑11‑08 article.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Dataset: `data/Net_generation_United_States_all_sectors_monthly.csv`
2. Run:

```bash
python main.py
```

## Outputs

- `outputs/eia_pytimetk_viz.png` – Monthly series with YoY percentage panel.
- `outputs/eia_viz.png` – Monthly line, yearly averages, and YoY overview.

These match the exploratory visuals from the article. Adjust the script if you want to experiment with different breakdowns.*** End Patch
