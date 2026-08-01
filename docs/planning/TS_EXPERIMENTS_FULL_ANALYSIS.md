# Complete Time Series Experiments Analysis

## Overview

Comprehensive analysis of `WIP/2025-11-05_ipynb_processed/01_Time_Series/_Experiments/` to identify unique features that can be extracted into the codebase.

## Already Covered Templates

We already have templates for:
-  ARIMA (`ARIMA_Python/`)
-  Prophet (`Prophet_Python/`)
-  Darts (`Darts_Python/`)
-  Greykite (`Greykite_Python/`)
-  Merlion (`Merlion_Python/`)
-  VAR (`VAR_Python/`)
-  BoxJenkins (`BoxJenkins_Python/`)
-  RegimeSwitching (`RegimeSwitching_Python/`)
-  TimeSeriesDecomposition (`TimeSeriesDecomposition_Python/`)

## Unique Features to Extract

### 1. **Enhanced Seasonal Decomposition Visualization** ⭐⭐
**Source:** `2025-04-04 time series ercot data with trend, season and residual plots and prophet.ipynb`

**What it does:**
- Uses `statsmodels.tsa.seasonal.seasonal_decompose`
- Creates multi-panel plots (trend, seasonal, residual)
- Clean visualization with Tufte-style (no top/right spines)
- Feature engineering (hour, day_of_week extraction)
- Box plots by hour/day of week
- Correlation heatmaps

**Why convert:**
- We have `TimeSeriesDecomposition_Python/` but may not have these visualization patterns
- Feature engineering utilities could be useful
- Clean plotting patterns

**Conversion Plan:**
- Enhance `TimeSeriesDecomposition_Python/` with better visualizations
- Or create `src/decomposition.py` utilities
- Add feature engineering helpers to `src/`

---

### 2. **Box-Jenkins VAR Methodology** ⭐
**Source:** `2025-07-15 Box Jenkins-1.ipynb`, `box-jenkins_VAR_multivariate.ipynb`

**What it does:**
- Stationarity testing (ADF test) before VAR
- Lag order selection using AIC/BIC/FPE/HQIC
- Durbin-Watson test for residual independence
- Proper differencing and inverse transformation
- Multi-step VAR forecasting

**Why convert:**
- We have `VAR_Python/` but may not have full Box-Jenkins methodology
- Lag selection and diagnostic tests are important

**Assessment:**
- **Lower priority** - we have VAR template
- Could enhance `VAR_Python/` with Box-Jenkins diagnostics
- Or extract as utilities

---

### 3. **Time Series Cross-Validation Utilities** ⭐⭐
**Source:** `box-jenkins_VAR_multivariate.ipynb` (Cell 2)

**What it does:**
- Uses `TimeSeriesSplit` from sklearn
- Stores predictions and holdouts for visualization
- Cross-validation loop with proper time-aware splitting

**Why convert:**
- Useful utility for all templates
- Could enhance `src/evaluator.py`
- Standardized CV visualization

**Conversion Plan:**
- Add to `src/evaluator.py` or create `src/cross_validation.py`
- Time-aware CV with visualization
- Reusable across all templates

---

### 4. **Feature Engineering Utilities** ⭐
**Source:** `2025-04-04 time series ercot data with trend, season and residual plots and prophet.ipynb`

**What it does:**
- Extracts time-based features (hour, day_of_week, day_of_year)
- StandardScaler normalization
- Box plots by time features
- Correlation analysis

**Why convert:**
- Useful for all time series templates
- Could add to `src/` utilities

**Conversion Plan:**
- Create `src/feature_engineering.py`
- Time-based feature extraction
- Visualization helpers

---

## Medium-Value Features

### 5. **Employee Badge Data Analysis**
**Source:** `2025-04-04 Employee Badge in project with arima and neural network.ipynb`

**What it does:**
- Monte Carlo simulation for badge data
- Aggregation patterns
- Missing value handling

**Assessment:**
- **Lower priority** - domain-specific
- Monte Carlo utilities could be useful but we have `utils/monte_carlo_simulation.py`

---

## Low-Value / Already Covered

### 6. **ARIMA notebooks** - We have `ARIMA_Python/`
### 7. **Prophet notebooks** - We have `Prophet_Python/`
### 8. **Darts notebooks** - We have `Darts_Python/`
### 9. **Greykite notebooks** - We have `Greykite_Python/`
### 10. **Merlion notebooks** - We have `Merlion_Python/`
### 11. **Regime switching** - We have `RegimeSwitching_Python/` (already uses MarkovRegression)

---

## Recommended Extractions

### Priority 1: Time Series Cross-Validation Utilities
- Add to `src/evaluator.py` or create `src/cross_validation.py`
- Time-aware CV with visualization
- Reusable across templates

### Priority 2: Feature Engineering Utilities
- Create `src/feature_engineering.py`
- Time-based features (hour, day_of_week, etc.)
- Visualization helpers

### Priority 3: Enhanced Decomposition Visualization
- Enhance `TimeSeriesDecomposition_Python/` or create utilities
- Multi-panel plots
- Clean Tufte-style visualization

### Priority 4: Box-Jenkins VAR Diagnostics
- Enhance `VAR_Python/` with diagnostic tests
- Lag selection visualization
- Residual diagnostics

---

## Implementation Plan

### 1. Time Series Cross-Validation
**Location:** `src/cross_validation.py` or enhance `src/evaluator.py`

**Features:**
- `TimeSeriesCrossValidator` class
- Time-aware splitting
- CV visualization
- Metrics aggregation across folds

### 2. Feature Engineering
**Location:** `src/feature_engineering.py`

**Features:**
- `extract_time_features()` - hour, day_of_week, day_of_year, etc.
- `create_lag_features()` - lagged values
- `create_rolling_features()` - rolling statistics
- Visualization helpers

### 3. Enhanced Decomposition
**Location:** Enhance `TimeSeriesDecomposition_Python/` or `src/decomposition.py`

**Features:**
- Multi-panel decomposition plots
- Tufte-style visualization
- Component analysis

---

## Next Steps

1.  **Extract time series CV utilities** - Created `src/cross_validation.py`
2.  **Extract feature engineering utilities** - Created `src/feature_engineering.py`
3. ⏳ **Enhance decomposition template** - Better visualizations (optional)
4. ⏳ **Enhance VAR template** - Add Box-Jenkins diagnostics (optional)

## Implementation Status

### Completed: Time Series Cross-Validation Utilities

**Location:** `src/cross_validation.py`

**Features:**
-  `TimeSeriesCrossValidator` class
-  Time-aware splitting (prevents data leakage)
-  CV evaluation with metrics
-  Visualization of CV splits
-  Reusable across all templates

**Usage:**
```python
from src import TimeSeriesCrossValidator

cv = TimeSeriesCrossValidator(n_splits=5)
splits = cv.split(series)
metrics = cv.evaluate(X, y, model_factory, fit_func, predict_func)
fig = cv.plot_cv_splits(series)
```

---

### Completed: Feature Engineering Utilities

**Location:** `src/feature_engineering.py`

**Features:**
-  `extract_time_features()` - Time-based features (hour, day_of_week, etc.)
-  `create_lag_features()` - Lagged values
-  `create_rolling_features()` - Rolling statistics
-  `create_differenced_features()` - Differencing
-  `create_seasonal_features()` - Sine/cosine transformations
-  `prepare_features()` - Complete pipeline

**Usage:**
```python
from src import prepare_features

features_df = prepare_features(
    series,
    include_time_features=True,
    include_lags=True,
    include_rolling=True,
    lags=[1, 2, 3, 7],
    rolling_windows=[3, 7, 30],
)
```

