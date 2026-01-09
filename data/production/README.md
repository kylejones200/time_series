# Production Data Examples

This directory contains example production time series data for testing forecasting models.

## Data Format

All production data files should follow this CSV format:

```csv
well_id,date,oil_rate,gas_rate,water_rate,cum_oil,cum_gas
WELL_001,2020-01-01,100.5,50.2,10.1,0,0
WELL_001,2020-02-01,95.3,48.1,9.8,2958,1479
...
```

### Required Columns
- `well_id`: Unique well identifier (string)
- `date`: Date of measurement (YYYY-MM-DD)
- `oil_rate`: Oil production rate (barrels/day or similar)
- `gas_rate`: Gas production rate (MCF/day or similar)
- `water_rate`: Water production rate (barrels/day or similar)

### Optional Columns
- `cum_oil`: Cumulative oil production
- `cum_gas`: Cumulative gas production
- `cum_water`: Cumulative water production

## Example Files

- `well_production.csv`: Single well production data
- `multi_well.csv`: Multiple wells in one file
- `synthetic_decline.csv`: Synthetic decline curve data for testing

## Usage

```python
from pipelines import ForecastingPipeline

# Load single well
pipeline = ForecastingPipeline(
    data_path="data/production/well_production.csv",
    target_column="oil_rate",
    forecast_horizon=12  # months
)

# Or specify well_id for multi-well file
pipeline = ForecastingPipeline(
    data_path="data/production/multi_well.csv",
    well_id="WELL_001",
    target_column="oil_rate",
    forecast_horizon=12
)
```

