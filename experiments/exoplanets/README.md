# Exoplanet Experiments

Combined workspace for the exoplanet notebooks that previously lived in
`99_Misc/` (`exo conversion`, `exoplanet pca`, `Exoplanet_Analysis_Complete`).

The workflow focuses on the Kepler light-curve dataset (Kaggle’s “NASA
Exoplanet Search”). Two CSV files are required:

- `exoTrain.csv`
- `exoTest.csv`

Place both files in `data/exoplanets/`. These are not tracked in the repo.

## Usage

```bash
cd experiments/exoplanets
python analysis.py
```

The script will:

1. Summarise the train/test datasets
2. Convert the wide flux arrays to weekly resampled time series and export
   `outputs/exoplanet_weekly_timeseries.csv`
3. Run PCA (200 components) and export
   `outputs/exoplanet_pca_components.csv`
4. Save a sample light-curve plot to `outputs/sample_light_curves.png`

## Notes

- Adjust the PCA dimensionality inside `analysis.py` if needed
- If `exoTest.csv` is unavailable the script will still process the training
  data and continue
- All heavy plotting is optional; feel free to comment out sections if not
  needed
