# Copula Methods for Multivariate Time Series

Multivariate dependency modeling using copulas for time series forecasting and risk analysis.

## Features

- ✅ **Clayton Copula**: Lower tail dependence
- ✅ **Gumbel Copula**: Upper tail dependence
- ✅ **Frank Copula**: Symmetric dependence
- ✅ **Gaussian Copula**: Elliptical dependence
- ✅ Multivariate dependency modeling
- ✅ Joint forecasting with preserved dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your two time series data files in the shared `data/` directory
2. Update `config.yaml` with your data file names and column names
3. Run:

```bash
python main.py
```

## Configuration

Edit `config.yaml` to customize:

- **data**: 
  - `series1_file` and `series2_file`: Input file names
  - `series1_col` and `series2_col`: Column names for values
  - `difference`: Whether to difference series before analysis
- **model**:
  - `copula_type`: `"Clayton"`, `"Gumbel"`, `"Frank"`, or `"Gaussian"`
  - `n_samples`: Number of samples to generate

## Copula Types

### Clayton
- Lower tail dependence
- Good for joint downside risk
- Asymmetric dependence

### Gumbel
- Upper tail dependence
- Good for joint upside risk
- Asymmetric dependence

### Frank
- Symmetric dependence
- No tail dependence
- Good for moderate correlations

### Gaussian
- Elliptical dependence
- Symmetric
- Most flexible

## Outputs

- `outputs/copula_analysis.png`: Four-panel plot showing:
  - Original time series
  - Uniform transformed data
  - Copula samples
  - Forecasted joint distribution

## Applications

- Risk management (joint tail events)
- Portfolio optimization
- Multivariate forecasting
- Dependency structure analysis
- Stress testing

## Notes

- Separates marginal distributions from dependence structure
- Preserves correlation structure in forecasts
- Useful for non-Gaussian dependencies
- Requires rank transformation
- Best for bivariate analysis (can extend to multivariate)

