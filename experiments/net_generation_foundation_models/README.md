# Net Generation Foundation Model Experiments

Frozen reproduction snapshots for the **November 2025 Net Generation article**. These scripts use article-specific CSV column names and forecast windows — they are **not** the day-to-day templates.

## Canonical templates

For normal forecasting, use the `*_Python/` templates at the repo root:

| Model | Template |
|---|---|
| Chronos | `Chronos_Python/` |
| TimesFM | `TimesFM_Python/` |
| Lag-Llama | `LagLlama_Python/` |

## Experiment folders (article snapshots)

| Folder | Model | Notes |
|---|---|---|
| `chronos/` | Amazon Chronos | Article data paths and plotting |
| `timesfm/` | Google TimesFM | Article data paths and plotting |
| `granite_ttm/` | IBM Granite TTM | Renamed from `lag_llama/` (was misnamed) |
| `moirai/` | Salesforce Moirai | Renamed from `granite/` (was misnamed) |

Each folder has its own `config.yaml` and `main.py`. Run from the folder:

```bash
cd experiments/net_generation_foundation_models/chronos
python main.py
```

Data files live in `data/` and use `*_ml-ready.csv` naming from the article assets.
