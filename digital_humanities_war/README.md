# Digital Humanities — War Sentiment Toolkit

This project reorganises the legacy `WIP/digital_humanities_war` assets into a reproducible, configuration-driven pipeline for analysing war-related newspaper sentiment and alliance information.

## Layout

```
digital_humanities_war/
├── env/
│   └── requirements.txt        # Python dependencies
├── configs/
│   └── war_sentiment.yaml      # Example configuration
├── data/
│   ├── raw/                    # Source CSVs
│   │   └── war_term_combined_data.csv
│   └── processed/              # Placeholder for cleaned exports
├── experiments/                # Output folders created per run
├── notebooks/
│   └── 2025-04-04 time series alliance info from correlates of war data.ipynb
├── reports/
│   ├── figures/                # Legacy plots from WIP
│   └── tables/                 # Placeholder for tabular outputs
└── src/
    ├── dataset_loaders.py
    ├── preprocessing.py
    ├── war_sentiment.py
    ├── plotting.py
    └── run_analysis.py
```

## Quick Start

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r env/requirements.txt
   ```

2. **Run the default analysis**
   ```bash
   python -m digital_humanities_war.src.run_analysis \
     --config digital_humanities_war/configs/war_sentiment.yaml \
     --project-root /Users/kylejonespatricia/time_series/digital_humanities_war
   ```
   This will aggregate TextBlob sentiment scores, export `war_sentiment_summary.csv`, and save per-term polarity plots under `experiments/war_sentiment/outputs/figures/`.

3. **Create new experiments**
   - Copy `configs/war_sentiment.yaml` and adjust paths/parameters.
   - Ensure new datasets live in `data/raw/` and document any cleaning steps by placing derived CSVs in `data/processed/`.

## Migrated Assets

- `war_term_combined_data.csv` → `data/raw/`
- PNG figures (`war_sentiment_decomposition.png`, `ww2_sentiment_trends.png`, etc.) → `reports/figures/`
- Notebook `2025-04-04 time series alliance info from correlates of war data.ipynb` → `notebooks/`

## Next Steps

- Add data provenance notes (e.g., Chronicling America, Correlates of War).
- Expand `war_sentiment.py` with additional scoring methods or topic modelling.
- Integrate processed outputs with the broader `digital_humanities/` project if shared utilities emerge.
