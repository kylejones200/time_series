# Oil & Gas WIP Code Conversion Analysis

## Overview

Analysis of code in `WIP/2025-11-05_ipynb_processed/03_Oil_and_Gas/` that can be converted into time series forecasting methods for this repository.

## High-Value Conversions

### 1. **Gaussian Process Regression (GPR) for Time Series** ⭐⭐⭐
**Source:** `Gaussian Process Reservoir Modeling.ipynb`

**What it does:**
- Uses `sklearn.gaussian_process.GaussianProcessRegressor` for spatial/temporal prediction
- Implements RBF + Matern kernels
- Provides uncertainty quantification (predictive variance)
- Cross-validation framework

**Why convert:**
- **We don't have a GP template yet!** This would fill a gap
- GP is excellent for time series with uncertainty bounds
- Natural fit for production forecasting with confidence intervals

**Conversion plan:**
- Create `GaussianProcess_Python/` template
- Adapt from 3D spatial to 1D time series
- Use time as the single dimension
- Keep uncertainty quantification (key advantage of GP)
- Integrate with `src` utilities

**Key code to extract:**
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

kernel = RBF(length_scale=1.0) + Matern(length_scale=1.0, nu=1.5)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
gpr.fit(X_train, y_train)
pred, sigma = gpr.predict(X_test, return_std=True)  # Uncertainty!
```

---

### 2. **Price Regression with Granger Causality** ⭐⭐
**Source:** `Shell and Brent Crude regression.ipynb`

**What it does:**
- Multivariate regression between related time series (Shell stock vs Brent crude)
- Granger causality testing (`statsmodels.tsa.stattools.grangercausalitytests`)
- Correlation analysis
- Linear regression forecasting

**Why convert:**
- Multivariate forecasting template (we have VAR, but this is simpler)
- Granger causality is useful for feature selection
- Good for commodity price forecasting

**Conversion plan:**
- Create `GrangerCausality_Python/` or `PriceRegression_Python/` template
- Test causality between multiple series
- Use leading indicators for forecasting
- Integrate with existing regression templates

**Key code to extract:**
```python
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import LinearRegression

# Test if Brent causes Shell (or vice versa)
grangercausalitytests(df[['Shell_Close', 'Brent_Close']], maxlag=5)
```

---

### 3. **WTI Crude Oil Price Fetcher** ⭐
**Source:** `wti_crude_oil_price_fetcher.ipynb`

**What it does:**
- Fetches historical WTI prices from Yahoo Finance (`yfinance`)
- CLI interface with date ranges
- CSV export

**Why convert:**
- Useful data utility for oil/gas forecasting
- Could be part of a data loading utility
- Good example dataset for templates

**Conversion plan:**
- Add to `data/` utilities or `src/loader.py`
- Or create a simple data fetching script
- Use as example dataset for price forecasting templates

**Key code to extract:**
```python
import yfinance as yf

def fetch_wti(start: str, end: str) -> pd.DataFrame:
    ticker = yf.Ticker("CL=F")
    hist = ticker.history(start=start, end=end)
    return hist[["Date", "Close"]].rename(columns={"Close": "Price"})
```

---

## Medium-Value Conversions

### 4. **Commodity Cost Curve Analysis** ⭐
**Source:** `commodity_cost_curve_oil_analysis.ipynb`

**What it does:**
- Visualization of cost curves (production cost vs volume)
- Stacked bar charts with cost breakdowns
- Percentile analysis

**Why convert:**
- Useful for production economics analysis
- Could be a visualization utility
- Less directly related to time series forecasting

**Conversion plan:**
- Add as a utility function in `src/plotting.py` or `utils/`
- Or create a specialized analysis template if needed

---

### 5. **Decline Curve Analysis (Bakken)** 
**Source:** `Bakken production data - decline curve for MCK.ipynb`

**What it does:**
- Decline curve fitting for production data
- County-level analysis

**Why convert:**
- **We already have DCA models!** (`models/dca/`)
- This might have different fitting methods or visualizations
- Could enhance existing DCA implementation

**Conversion plan:**
- Review for any unique methods not in our DCA models
- Extract useful visualization patterns
- Integrate improvements into existing `models/dca/`

---

## Low-Value / Not Suitable

### 6. **SEGY/Kmeans Data** - Seismic data processing (not time series)
### 7. **Well Log Analysis** - Log curve analysis (not forecasting)
### 8. **AVO Attributes** - Seismic attribute analysis (not time series)
### 9. **Wavelet Estimation** - Signal processing (specialized, not general forecasting)
### 10. **PHMSA Safety** - Pipeline safety data merging (data engineering, not forecasting)
### 11. **Crude Optimization** - Scheduling optimization (operations research, not forecasting)
### 12. **PowerPoint Generator** - Report generation (not a forecasting method)

---

## Recommended Priority

1. **Gaussian Process Regression** (High) - New method, fills gap, uncertainty quantification
2. **Granger Causality / Price Regression** (Medium) - Multivariate forecasting enhancement
3. **WTI Price Fetcher** (Low) - Data utility, nice to have

---

## Implementation Notes

### Gaussian Process Template
- **Location:** `GaussianProcess_Python/`
- **Dependencies:** `scikit-learn>=1.3.0` (already in requirements)
- **Key features:**
  - Time-based kernel (RBF for smoothness)
  - Uncertainty bounds (predictive variance)
  - Hyperparameter optimization
  - Integration with `src` utilities

### Granger Causality Template
- **Location:** `GrangerCausality_Python/` or enhance `VAR_Python/`
- **Dependencies:** `statsmodels>=0.14.0` (already in requirements)
- **Key features:**
  - Causality testing between series
  - Leading indicator selection
  - Multivariate regression forecasting

---

## Next Steps

1.  **Extract GP code** from `Gaussian Process Reservoir Modeling.ipynb`
2.  **Adapt to 1D time series** (remove 3D spatial components)
3.  **Create template structure** following existing patterns
4.  **Integrate with `src` utilities** (loader, evaluator, plotting)
5. ⏳ **Test on production data** from `data/production/`
6. ⏳ **Add to template registry** in `forecast.py`

## Implementation Status

### Completed: Granger Causality Template

**Location:** `GrangerCausality_Python/`

**Features:**
-  Granger causality testing between two time series
-  Stationarity testing (Augmented Dickey-Fuller test)
-  Multivariate regression forecasting with lagged predictors
-  Leading indicator selection (automatic lag detection)
-  Correlation analysis
-  Integration with `src` utilities
-  Standardized output (CSV, plots, summary)

**Key Code:**
- Uses `grangercausalitytests` from `statsmodels.tsa.stattools`
- Tests multiple lags to find optimal leading indicator
- Fits OLS regression with lagged predictors
- Provides causality interpretation and forecasting

**Usage:**
```bash
cd GrangerCausality_Python
python main.py
```

**Outputs:**
- `granger_forecast.png` - Forecast plot with train/test/predictions
- `granger_forecast.csv` - Forecast values
- `granger_summary.csv` - Causality test results (p-values, lags, interpretation)

---

### Completed: Gaussian Process Regression Template

**Location:** `GaussianProcess_Python/`

**Features:**
-  Time series adaptation (1D time features)
-  RBF + Matern kernel combination
-  Uncertainty quantification (predictive variance)
-  Confidence intervals (95% CI)
-  Integration with `src` utilities
-  Standardized output (CSV, plots, metrics)

**Key Code:**
- Converts datetime index to numeric (days since start)
- Uses `GaussianProcessRegressor` from scikit-learn
- Provides both point forecasts and uncertainty bounds
- Configurable kernel types (RBF, Matern, or combined)

**Usage:**
```bash
cd GaussianProcess_Python
python main.py
```

**Outputs:**
- `gp_forecast.png` - Forecast plot with uncertainty bands
- `gp_forecast.csv` - Forecast values with confidence intervals
- `gp_metrics.csv` - Performance metrics (RMSE, MAE, R²)

---

## Questions to Consider

- Should GP be a standalone template or part of a "Bayesian Methods" category?
- Do we want to keep the spatial modeling code for future reservoir forecasting?
- Should Granger causality be its own template or enhance VAR?
- How should we handle the yfinance dependency (optional vs required)?

