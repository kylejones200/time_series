# Curated Time Series Datasets

This directory holds a small collection of CSV files that power the templates and experiments in this repository. Each file is intentionally lightweight so projects can be run end-to-end without hunting for raw data.

## Root CSVs

- `amtrak_ridership_time_series_data.csv` – Monthly Amtrak ridership counts used in baseline forecasting examples.
- `ercot_load_data.csv` – ERCOT demand telemetry sampled every 15 minutes. Ideal for load-forecasting demos.
- `ercot_power_data.csv` – Companion ERCOT series with power output in MW.
- `ercot_dsr_load_gen.csv` – ERCOT demand-side response dataset with load and generation columns for anomaly and regime detection.

## Reference Data (`reference/`)

- `apricot_yield_with_missing_values.csv` – Agricultural yields with intentional gaps for data-quality utilities.
- `fred_interaction_effects_monthly.csv` – Macro indicators (FRED) for econometric interaction studies.
- `hierarchical_amtrak_ridership.csv` – Multi-level ridership series for hierarchical forecasting experiments.
- `north_dakota_oil_price.csv` – Well-level pricing data for panel regression and causal analysis.
- `resampled_gdp_per_capita.csv` – GDP per capita series aligned to monthly frequency.
- `resampled_life_expectancy.csv` – Life expectancy series aligned to GDP data for joint modeling.

If you need heavier raw datasets, they remain under `WIP/` so the templates stay lean. Update this README whenever new reference files are promoted.

