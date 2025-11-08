# Aeon Clustering Template (EIA Net Generation)

This template reproduces the AEON time-series clustering workflow from the 2025‑11‑08 article, grouping annual EIA generation profiles with DTW-based `TimeSeriesKMeans`.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

`config.yaml` controls the entire pipeline:
- `data`: shared CSV in `data/Net_generation_United_States_all_sectors_monthly.csv`, plus column names and frequency.
- `clustering`: number of clusters, distance metric (`dtw`), and seasonal period (12 months).
- `plotting`: colors, figure size, and DPI for the summary figure.
- `output`: filename and directory (defaults to `outputs/eia_aeon_ts_clusters.png`).

## Run

```bash
python main.py
```

## Outputs

Running the template produces a 6-panel summary saved to `outputs/eia_aeon_ts_clusters.png`, including:
- Cluster assignments over time
- Seasonal centroids
- All annual profiles colored by cluster
- DTW distance heat map
- Representative year per cluster
- Metrics panel (inertia & silhouette)

Console output lists the cluster composition and basic dataset stats so the article text is fully reproducible.*** End Patch

