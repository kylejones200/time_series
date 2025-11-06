# Sparse Regression (LASSO) for Time Series

LASSO, Ridge, and Elastic Net regression with automatic feature selection for time series forecasting.

## Features

- ✅ **LASSO**: L1 regularization for automatic feature selection
- ✅ **Ridge**: L2 regularization for multicollinearity handling
- ✅ **Elastic Net**: Combined L1/L2 regularization
- ✅ Time series cross-validation
- ✅ Automatic feature selection (sparsity)
- ✅ Lag and rolling window features
- ✅ Model evaluation metrics (MAE, RMSE, R²)
- ✅ Feature importance visualization

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

- **model**:
  - `type`: `"lasso"`, `"ridge"`, or `"elastic_net"`
  - `cv_splits`: Number of time series cross-validation splits
  - `max_iter`: Maximum iterations for optimization
  - `alphas`: Regularization path (null = auto)
- **features**:
  - `create_lags`: Whether to create lag features
  - `lags`: List of lag periods
  - `create_rolling`: Whether to create rolling window features
  - `rolling_windows`: Rolling window sizes
  - `rolling_funcs`: Functions to apply (mean, std, etc.)

## Methods

### LASSO (L1 Regularization)
- Automatically selects relevant features
- Sets irrelevant features to zero
- Good for high-dimensional data
- Sparse solutions

### Ridge (L2 Regularization)
- Shrinks coefficients but doesn't eliminate them
- Handles multicollinearity
- All features retained

### Elastic Net
- Combines L1 and L2 regularization
- Balances feature selection and coefficient shrinkage
- Good for correlated features

## Outputs

- `outputs/sparse_regression_analysis.png`: Four-panel plot showing:
  - Time series forecast
  - Selected features (coefficients)
  - Actual vs predicted scatter plot
  - Model summary (optimal α, sparsity)

## Interpretation

- **Sparsity**: Percentage of features set to zero
- **Selected features**: Features with non-zero coefficients
- **Optimal α**: Regularization parameter chosen by CV
- **Coefficient values**: Magnitude indicates feature importance

## Notes

- Best for high-dimensional feature spaces
- LASSO automatically performs feature selection
- Time series cross-validation prevents data leakage
- StandardScaler ensures fair regularization
- Can handle many lagged and rolling features

