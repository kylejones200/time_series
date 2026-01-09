# Forecasting WIP Directory Analysis

## Overview

Analysis of `WIP/forecasting/2025-11-05_ipynb_processed/01_Time_Series/` to identify unique code that can be extracted into the main repo.

## Files Analyzed

1. `2025-04-04 time series knn and dynamic time warping.ipynb`
2. `CI in time series.ipynb` - Bootstrap confidence intervals
3. `Serial Correlation in Time Series with MICH data.ipynb`
4. `Nixtla Suite for Time Series Forecasting with Python.ipynb`
5. `Time series with fred unemployment and bollinger bands.ipynb`
6. `2025-04-04 time series moving average.ipynb`
7. `2025-04-04 time series panel regression with driscoll kraay using north dakota data.ipynb`

## Existing Templates

- ✅ `ConfidenceIntervals_Python/` - Basic confidence intervals
- ✅ `SerialCorrelation_Python/` - Serial correlation tests
- ✅ `Nixtla_Python/` - Nixtla suite
- ✅ `tslearn_Python/` - Time series learning (includes DTW)
- ✅ `BollingerBands_Python/` - Bollinger bands
- ✅ `MovingAverage_Python/` - Moving averages
- ✅ `PanelRegression_Python/` - Panel regression (already created)

## Unique Features to Extract

### 1. **Bootstrap Confidence Intervals for ARIMA** ⭐⭐
**Source:** `CI in time series.ipynb`

**What it does:**
- Bootstrap resampling for ARIMA confidence intervals
- More robust than parametric CIs
- Handles model uncertainty

**Why convert:**
- We have `ConfidenceIntervals_Python/` but may not have bootstrap method
- Bootstrap CIs are more robust
- Useful for uncertainty quantification

**Key Code:**
```python
def bootstrap_ci(model_order, data, steps=48, n_bootstraps=100, confidence=0.95):
    forecasts = []
    for i in range(n_bootstraps):
        sample = data.sample(n=len(data), replace=True).sort_index()
        model = ARIMA(sample, order=model_order).fit()
        forecasts.append(model.forecast(steps=steps).values)
    # Calculate percentiles
    return (np.mean(forecasts, axis=0),
            np.percentile(forecasts, alpha * 100, axis=0),
            np.percentile(forecasts, (1 - alpha) * 100, axis=0))
```

**Conversion Plan:**
- Enhance `ConfidenceIntervals_Python/` with bootstrap method
- Or create `src/confidence_intervals.py` utility

---

### 2. **DTW-based KNN Classification** ⭐
**Source:** `2025-04-04 time series knn and dynamic time warping.ipynb`

**What it does:**
- KNN classification using DTW distance
- Time series classification
- Uses `tslearn.neighbors.KNeighborsTimeSeriesClassifier`

**Why convert:**
- We have `tslearn_Python/` but may not have classification example
- Useful for time series classification tasks

**Assessment:**
- **Lower priority** - we have tslearn template
- Could enhance existing template or create classification variant

---

### 3. **Serial Correlation Corrections** ⭐
**Source:** `Serial Correlation in Time Series with MICH data.ipynb`

**What it does:**
- Breusch-Godfrey test
- GLS (Generalized Least Squares)
- Cochrane-Orcutt method
- Newey-West standard errors

**Why convert:**
- We have `SerialCorrelation_Python/` but may not have all correction methods
- These are important econometric techniques

**Assessment:**
- **Medium priority** - enhance existing template
- Add GLS, Cochrane-Orcutt, Newey-West methods

---

### 4. **Enhanced Nixtla Suite Examples** ⭐
**Source:** `Nixtla Suite for Time Series Forecasting with Python.ipynb`

**What it does:**
- Comprehensive examples of:
  - StatsForecast
  - MLForecast
  - NeuralForecast
  - HierarchicalForecast

**Why convert:**
- We have `Nixtla_Python/` but may not have all suite components
- HierarchicalForecast is particularly valuable

**Assessment:**
- **Medium priority** - enhance existing template
- Add HierarchicalForecast example

---

## Recommended Extractions

### Priority 1: Bootstrap Confidence Intervals ⭐⭐
- **Location:** Enhance `ConfidenceIntervals_Python/` or create utility
- **Why:** More robust than parametric CIs, handles model uncertainty
- **Features:**
  - Bootstrap resampling
  - Percentile-based confidence intervals
  - Works with ARIMA and other models

### Priority 2: Serial Correlation Corrections ⭐
- **Location:** Enhance `SerialCorrelation_Python/`
- **Why:** Important econometric corrections
- **Features:**
  - GLS estimation
  - Cochrane-Orcutt method
  - Newey-West standard errors

### Priority 3: HierarchicalForecast ⭐
- **Location:** Enhance `Nixtla_Python/`
- **Why:** Valuable for hierarchical time series
- **Features:**
  - Hierarchical structure definition
  - Bottom-up reconciliation
  - Multi-level forecasting

---

## Implementation Plan

### 1. Bootstrap Confidence Intervals
**Location:** `src/confidence_intervals.py` or enhance `ConfidenceIntervals_Python/`

**Features:**
- `bootstrap_ci()` function
- Works with any forecasting model
- Configurable bootstrap iterations
- Percentile-based intervals

### 2. Serial Correlation Corrections
**Location:** Enhance `SerialCorrelation_Python/`

**Features:**
- GLS estimation
- Cochrane-Orcutt iterative fitting
- Newey-West robust standard errors
- Comparison of methods

### 3. HierarchicalForecast
**Location:** Enhance `Nixtla_Python/` or create `HierarchicalForecast_Python/`

**Features:**
- Hierarchy definition
- Bottom-up reconciliation
- Multi-level visualization
- Coherent forecasts

---

## Next Steps

1. ⏳ **Extract bootstrap CI utility** - Create `src/confidence_intervals.py`
2. ⏳ **Enhance SerialCorrelation template** - Add correction methods
3. ⏳ **Enhance Nixtla template** - Add HierarchicalForecast

