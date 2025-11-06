# Econometric Time Series Analysis

Comprehensive econometric methods for causal inference, policy evaluation, and economic modeling.

## Features

- ✅ Granger Causality Tests
- ✅ Regression Discontinuity Design (RDD)
- ✅ OLS Regression
- ✅ Vector Autoregression (VAR)
- ✅ Stationarity testing
- ✅ Model diagnostics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your data file in the shared `data/` directory
2. Update `config.yaml` with your data file name, column names, and method
3. Run:

```bash
python main.py
```

## Configuration

Edit `config.yaml` to select and configure the method:

### Granger Causality
- **method**: `granger`
- **x_col**: Variable that may cause y_col
- **y_col**: Variable that may be caused by x_col
- **max_lag**: Maximum lag to test

### Regression Discontinuity Design (RDD)
- **method**: `rdd`
- **cutoff_date**: Date of the treatment/policy change
- Tests for causal effects at the cutoff point

### OLS Regression
- **method**: `ols`
- **y_col**: Dependent variable
- **x_cols**: List of independent variables

### Vector Autoregression (VAR)
- **method**: `var`
- **value_cols**: List of variables for multivariate analysis
- **var_max_lags**: Maximum lags to consider

## Methods

### Granger Causality
Tests whether one time series helps predict another. Useful for:
- Economic relationships
- Causal inference
- Predictive relationships

### Regression Discontinuity Design (RDD)
Evaluates causal effects of treatments/policies at a cutoff point. Useful for:
- Policy evaluation
- Treatment effect estimation
- Natural experiments

### OLS Regression
Standard linear regression for time series. Useful for:
- Economic relationships
- Cross-sectional analysis
- Simple causal modeling

### Vector Autoregression (VAR)
Multivariate time series modeling. Useful for:
- Multiple interdependent variables
- Economic systems
- Cross-variable relationships

## Outputs

- `outputs/rdd_analysis.png`: RDD visualization (if method is RDD)
- Console output: Model summaries, test statistics, diagnostics

## Notes

- All methods include stationarity testing where applicable
- VAR automatically handles differencing if needed
- Granger causality tests multiple lags and reports minimum p-value
- RDD includes treatment effect estimation and visualization

