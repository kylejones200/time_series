# Time Series Main Directory Analysis

## Overview

Analysis of `WIP/2025-11-05_ipynb_processed/01_Time_Series/` (excluding `_Experiments`) to identify code that can be extracted into the full repo/library.

## Files Analyzed

1. `2025-04-04 time series panel regression with driscoll kraay using north dakota data.py`
2. `forecasting.ipynb` - MOMENT foundation model
3. `Time Series methods with Pandas in Python.ipynb` - Pandas utilities
4. `Energy Demand forecasting using Ercot data with interactive plot.ipynb`
5. `2025-07-15 Bellevue SolarAnywhere time series.ipynb`
6. `2025-04-04 time series merged output of telem dsr load and gen from ercot.ipynb`

---

## High-Value Extractions

### 1. **Panel Regression with Driscoll-Kraay Standard Errors** ⭐⭐⭐
**Source:** `2025-04-04 time series panel regression with driscoll kraay using north dakota data.py`

**What it does:**
- Panel data regression (multiple entities over time)
- Uses `linearmodels.panel.PanelOLS`
- Driscoll-Kraay standard errors (robust to cross-sectional and temporal correlation)
- Clustered standard errors comparison
- Entity fixed effects
- Well-suited for oil/gas production data (multiple wells over time)

**Why convert:**
- **We don't have panel regression templates!**
- Very useful for production data analysis
- Driscoll-Kraay SEs are important for panel data
- Directly relevant to DCA/oil & gas use cases

**Key Code:**
```python
from linearmodels.panel import PanelOLS

panel_model = PanelOLS(data["Oil"], X_matrix, entity_effects=True).fit(
    cov_type="kernel", kernel="bartlett", bandwidth=3
)
dk_se = panel_model.std_errors
```

**Conversion Plan:**
- Create `PanelRegression_Python/` template
- Support multiple standard error types (Driscoll-Kraay, clustered, robust)
- Entity and time fixed effects
- Visualization of panel data

---

### 2. **MOMENT Foundation Model** ⭐⭐
**Source:** `forecasting.ipynb`

**What it does:**
- Uses MOMENT (foundation model for time series)
- Pre-trained transformer model
- Fine-tuning for forecasting
- Mixed precision training
- Gradient clipping

**Why convert:**
- **We don't have MOMENT template!**
- Foundation models are cutting-edge
- Pre-trained models can be powerful

**Key Code:**
```python
from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={'task_name': 'forecasting', ...}
)
```

**Assessment:**
- **Medium priority** - requires `momentfm` library
- Could create `MOMENT_Python/` template
- Useful for users who want foundation models

---

### 3. **Pandas Time Series Utilities** ⭐
**Source:** `Time Series methods with Pandas in Python.ipynb`

**What it does:**
- Basic pandas operations: lagging, rolling, resampling
- Visualization patterns

**Why convert:**
- We already have `src/feature_engineering.py` with similar functionality
- Could enhance existing utilities or create tutorial template

**Assessment:**
- **Lower priority** - mostly covered by existing utilities
- Could create `PandasTutorial_Python/` as educational template

---

### 4. **Interactive Plotting Utilities** ⭐
**Source:** `Energy Demand forecasting using Ercot data with interactive plot.ipynb`

**What it does:**
- Interactive plots (likely Plotly/Bokeh)
- Energy demand forecasting with interactive visualization

**Why convert:**
- Could add interactive plotting utilities to `src/plotting.py`
- Useful for dashboards and exploration

**Assessment:**
- **Lower priority** - could enhance plotting utilities
- Optional feature for visualization

---

## Medium-Value Features

### 5. **Solar Time Series Analysis**
**Source:** `2025-07-15 Bellevue SolarAnywhere time series.ipynb`

**What it does:**
- Solar irradiance time series analysis
- Likely similar to other energy forecasting

**Assessment:**
- **Lower priority** - domain-specific
- Could extract patterns if unique

---

## Recommended Extractions

### Priority 1: Panel Regression with Driscoll-Kraay ⭐⭐⭐
- **Location:** `PanelRegression_Python/`
- **Why:** Unique method, directly relevant to oil/gas, we don't have it
- **Features:**
  - Panel data handling
  - Driscoll-Kraay standard errors
  - Clustered standard errors
  - Entity/time fixed effects
  - Panel data visualization

### Priority 2: MOMENT Foundation Model ⭐⭐
- **Location:** `MOMENT_Python/`
- **Why:** Cutting-edge foundation model, we don't have it
- **Features:**
  - Pre-trained model loading
  - Fine-tuning
  - Forecasting with foundation models

---

## Implementation Plan

### 1. Panel Regression Template
**Location:** `PanelRegression_Python/`

**Features:**
- Load panel data (multi-index: entity, date)
- Fit PanelOLS with various standard error types
- Entity and time fixed effects
- Visualization of panel data
- Standard error comparison

**Dependencies:**
- `linearmodels` (already in requirements or add)

### 2. MOMENT Template (Optional)
**Location:** `MOMENT_Python/`

**Features:**
- Load pre-trained MOMENT model
- Fine-tuning pipeline
- Forecasting
- Integration with `src` utilities

**Dependencies:**
- `momentfm` (add to requirements)

---

## Next Steps

1.  **Create Panel Regression template** - Completed! `PanelRegression_Python/`
2. ⏳ **Create MOMENT template** - Medium value, foundation model (optional)
3. ⏳ **Enhance plotting with interactive options** - Lower priority (optional)

## Implementation Status

### Completed: Panel Regression Template

**Location:** `PanelRegression_Python/`

**Features:**
-  Panel data loading (multi-index: entity, date)
-  Driscoll-Kraay standard errors
-  Clustered standard errors
-  Robust standard errors (optional)
-  Entity and time fixed effects
-  Panel data visualization
-  Standard error comparison
-  Integration with `src` utilities

**Usage:**
```bash
cd PanelRegression_Python
python main.py
```

**Perfect for:**
- Oil/gas production data (multiple wells over time)
- Panel time series analysis
- Cross-sectional and temporal correlation handling

