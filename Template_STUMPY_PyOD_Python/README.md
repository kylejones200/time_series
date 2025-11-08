# Anomaly Detection Template (STL + Autoencoder + STUMPY/PyOD)

The template now mirrors the 2025‑11‑08 article workflow: STL residual screening plus a lightweight residual autoencoder, with optional STUMPY matrix profile and PyOD baselines.

## Features

- ✅ STL residual z-score detection with configurable seasonal period.
- ✅ Feed-forward autoencoder on STL residual windows; prints anomaly counts and saves both the main figure and error trace.
- ✅ Optional STUMPY matrix profile and PyOD outlier detectors (disable or enable in `config.yaml`).
- ✅ Shared EIA dataset (`data/Net_generation_United_States_all_sectors_monthly.csv`) used across all templates.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Confirm the shared CSV is in `data/`.
2. Adjust `config.yaml`:
   - `methods.stl`: turn on/off STL diagnostics, set `season` and `z_threshold`.
   - `methods.autoencoder`: window length, epochs, learning rate, and output filenames.
   - `methods.stumpy` / `methods.pyod`: set `enabled: true` if you want the optional baselines.
3. Run:

```bash
python main.py
```

## Outputs

- `outputs/eia_anomaly_stl.png` – STL residual anomalies.
- `outputs/eia_anomaly_autoencoder.png` – Autoencoder anomalies on STL residuals.
- `outputs/eia_anomaly_autoencoder_error.png` – Reconstruction-error trace with threshold.
- Optional: `outputs/stumpy_matrix_profile.png`, `outputs/pyod_<method>_anomalies.png`.

Console logs list how many anomalies each method detected, matching the numbers quoted in the article.
