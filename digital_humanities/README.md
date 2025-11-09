# Digital Humanities Analysis Toolkit

A reproducible, configuration-driven workspace for historical sentiment, linguistic change, and causal inference studies.  This project consolidates the exploratory work that previously lived in `WIP/digital_humanities/` into a maintainable structure.

## Project Layout

```
digital_humanities/
├── env/                    # Environment specifications
│   └── requirements.txt
├── configs/                # YAML definitions for individual studies
│   ├── sentiment_democracy.yaml
│   └── linguistic_change.yaml
├── data/
│   ├── raw/                # Original CSVs and spreadsheets
│   └── processed/          # Cleaned datasets ready for analysis
├── experiments/            # Per-analysis result folders (created on demand)
├── notebooks/              # Cleaned notebooks importing from src/
├── reports/
│   ├── figures/            # Generated visuals
│   └── tables/             # Generated tables/CSV summaries
├── src/                    # Reusable Python modules
│   ├── dataset_loaders.py
│   ├── preprocessing.py
│   ├── sentiment_analysis.py
│   ├── linguistic_trends.py
│   ├── causal_models.py
│   ├── plotting.py
│   └── run_analysis.py
└── README.md               # You are here
```

## Getting Started

1. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r env/requirements.txt
   ```

2. **Place new datasets** in `data/raw/`.  Document provenance and any cleaning steps in `data/README.md` (to be added when the first processed dataset is generated).

3. **Create a config** by copying one of the files in `configs/` and editing the paths/parameters.  Example (sentiment):
   ```yaml
   analysis: sentiment
   sentiment:
     dataset:
       path: data/raw/Democracy yearly_sentiment 1850-1927.csv
     date_col: year
     value_col: sentiment
     smoothing_window: 5
     title: "Democracy sentiment (1850-1927)"
     output_dir: experiments/sentiment_democracy/outputs
   ```

4. **Run the pipeline**:
   ```bash
   python -m digital_humanities.src.run_analysis \
     --config digital_humanities/configs/sentiment_democracy.yaml \
     --project-root /Users/kylejonespatricia/time_series/digital_humanities
   ```
   The script writes CSV summaries and figures into the `output_dir` specified by the config.

## Migrated Assets

- All historical sentiment CSVs previously located in `WIP/digital_humanities/Digital Humanities Sentiment/` now live in `digital_humanities/data/raw/`.
- Legacy plots and animations were moved to `digital_humanities/reports/figures/`.
- Exploratory notebooks were relocated to `digital_humanities/notebooks/` for future refactoring.

## Next Steps

- Add `data/README.md` describing each dataset’s provenance and cleaning steps.
- Flesh out `src/linguistic_trends.py` and `src/causal_models.py` with production-ready implementations.
- Update notebooks to import from the new `src/` modules, replacing ad-hoc code.
- Remove or archive the now-empty `WIP/digital_humanities` folder once changes are validated and version-controlled.

## Contributing

1. Add/modify configs in `configs/`.
2. Keep reusable logic inside `src/` modules.
3. Ensure plots/metrics land in `experiments/<analysis>/outputs/`.
4. Document new workflows in this README or dedicated markdown files under `reports/`.

By enforcing this structure, the digital humanities analyses become reproducible, testable, and easier to extend.
