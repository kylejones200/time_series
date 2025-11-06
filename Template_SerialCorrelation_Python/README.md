# Serial Correlation Analysis for Time Series

Tests and corrections for serial correlation in time series regression models.

## Features

- ✅ Breusch-Godfrey test for serial correlation
- ✅ Ljung-Box test for residual autocorrelation
- ✅ ACF plots for visualization
- ✅ Corrections: GLS, Cochrane-Orcutt, HAC standard errors

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your data file in the shared `data/` directory
2. Update `config.yaml` with your data file name and column names
3. Run:

```bash
python main.py
```

## Configuration

Edit `config.yaml` to customize:

- **y_col**: Dependent variable
- **x_cols**: Independent variables
- **create_lags**: Automatically create lagged features
- **max_lags**: Maximum lag order
- **test_lags**: Number of lags for tests
- **apply_corrections**: Apply correction methods

## Tests

### Breusch-Godfrey Test
Tests for serial correlation in regression residuals.
- p-value < 0.05: Serial correlation detected
- p-value >= 0.05: No significant serial correlation

### Ljung-Box Test
Tests for autocorrelation in residuals at multiple lags.

## Corrections

- **GLS**: Generalized Least Squares
- **Cochrane-Orcutt**: Iterative procedure to eliminate serial correlation
- **HAC**: Heteroskedasticity and Autocorrelation Consistent standard errors

## Outputs

- `outputs/serial_correlation.png`: Residuals plot and ACF
- Console output: Test results and corrected model summaries

