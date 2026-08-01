# Time Series Experiments Analysis

## Overview

Analysis of code in `WIP/2025-11-05_ipynb_processed/01_Time_Series/_Experiments/` that can be extracted into full features for this repository.

## High-Value Conversions

### 1. **Regime-Aware LSTM** ⭐⭐⭐
**Source:** `2025_07_15_LSTM_v_regime_aware_LSTM.ipynb`, `2025-07-15 LSTM v regime aware LSTM-1.ipynb`

**What it does:**
- LSTM model that incorporates regime information via embeddings
- Compares vanilla LSTM vs regime-aware LSTM
- Uses regime embeddings to augment input features
- Shows improved performance when regimes are present

**Why convert:**
- **We have LSTM_Python but not regime-aware variant!**
- Useful for time series with structural breaks or regime changes
- Natural extension of existing LSTM template
- Demonstrates regime-aware deep learning

**Key Code:**
```python
class RegimeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_regimes):
        self.embedding = nn.Embedding(num_regimes, 4)
        self.lstm = nn.LSTM(input_dim + 4, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, regime_id):
        regime_embed = self.embedding(regime_id)
        x_augmented = torch.cat([x, regime_embed], dim=2)
        return self.fc(self.lstm(x_augmented))
```

**Conversion Plan:**
- Create `RegimeAwareLSTM_Python/` template
- Integrate with existing `RegimeSwitching_Python` for regime detection
- Or create standalone that accepts regime labels
- Compare against vanilla LSTM

---

### 2. **Prophet + DCA Integration** ⭐⭐
**Source:** `prophet for DCA.ipynb`

**What it does:**
- Uses Prophet for time series forecasting
- Fits decline curve (hyperbolic) to same data
- Compares Prophet forecast vs DCA curve
- Visualizes both on same plot

**Why convert:**
- **Direct integration of time series forecasting with DCA!**
- This is exactly what the repo was designed for
- Shows how to compare TS methods vs traditional DCA
- Could enhance existing DCA comparison examples

**Key Code:**
```python
# Prophet forecast
m = Prophet()
m.fit(df1)
forecast = m.predict(future)

# DCA hyperbolic curve
qi = get_max_initial_production(df, 5, 'Oil', 'ReportDate')
popt, pcov = curve_fit(hyperbolic_equation, df['Days_Online'], df['Oil'])
df['Hyperbolic_Predicted'] = hyperbolic_equation(df['Days_Online'], *popt)

# Compare both
timeseries(forecast, 'ds', 'yhat', 'yhat_lower', 'yhat_upper', 
           actual=df1, curve=df['Hyperbolic_Predicted'])
```

**Conversion Plan:**
- Enhance `examples/ts_vs_dca_comparison.py`
- Or create `ProphetDCA_Python/` template
- Show side-by-side comparison
- Use existing DCA models from `models/dca/`

---

### 3. **Markov Regime Switching (Statsmodels)** ⭐
**Source:** `Regime changing time series.ipynb`

**What it does:**
- Uses `statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`
- Detects regime changes in time series
- Provides smoothed probabilities for each regime
- Transition matrix for regime switching

**Why convert:**
- **We have `RegimeSwitching_Python` but should check if it uses this method**
- Markov switching is a standard approach
- Could enhance existing template or create variant

**Key Code:**
```python
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

model = MarkovRegression(data, k_regimes=2, trend='c', switching_variance=True)
result = model.fit()
smoothed_probs = result.smoothed_marginal_probabilities
predicted_regimes = np.argmax(smoothed_probs, axis=1)
```

**Assessment:**
- Check if `RegimeSwitching_Python` already uses this
- If not, enhance or create variant

---

## Medium-Value Conversions

### 4. **Box Jenkins VAR Multivariate**
**Source:** `box-jenkins_VAR_multivariate.ipynb`

**What it does:**
- Vector Autoregression (VAR) implementation
- Multivariate time series modeling

**Why convert:**
- We have `VAR_Python/` template
- Could enhance with Box-Jenkins methodology
- Or extract useful patterns

**Assessment:**
- **Lower priority** - we already have VAR template
- Could extract Box-Jenkins specific patterns if unique

---

### 5. **Prophet + ERCOT Energy Demand**
**Source:** `TS analysis for energy demand using prophet and ERCOT data.ipynb`, `2025-04-04 time series ercot data with trend, season and residual plots and prophet.ipynb`

**What it does:**
- Prophet forecasting on energy demand data
- Trend/season/residual decomposition
- Energy-specific analysis

**Why convert:**
- We have `Prophet_Python/` template
- Could extract energy-specific patterns
- Or use as example dataset

**Assessment:**
- **Lower priority** - we have Prophet template
- Could extract decomposition visualization patterns

---

## Low-Value / Already Covered

### 6. **ARIMA notebooks** - We have `ARIMA_Python/`
### 7. **Prophet notebooks** - We have `Prophet_Python/`
### 8. **Darts notebooks** - We have `Darts_Python/`
### 9. **Greykite notebooks** - We have `Greykite_Python/`
### 10. **Merlion notebooks** - We have `Merlion_Python/`

---

## Recommended Priority

1. **Regime-Aware LSTM** (High) - New variant, fills gap, useful for regime-changing series
2. **Prophet + DCA Integration** (High) - Directly aligns with repo goals
3. **Markov Regime Switching Enhancement** (Medium) - Check existing template first

---

## Implementation Notes

### Regime-Aware LSTM Template
- **Location:** `RegimeAwareLSTM_Python/` or enhance `LSTM_Python/`
- **Dependencies:** `torch>=2.0` (already in requirements)
- **Key features:**
  - Regime embedding layer
  - Augmented LSTM input
  - Comparison with vanilla LSTM
  - Integration with regime detection

### Prophet + DCA Integration
- **Location:** Enhance `examples/ts_vs_dca_comparison.py` or create `ProphetDCA_Python/`
- **Key features:**
  - Side-by-side Prophet and DCA forecasts
  - Unified visualization
  - Performance comparison
  - Use existing `models/dca/` models

---

## Next Steps

1.  **Check existing templates** - `RegimeSwitching_Python` uses MarkovRegression, `LSTM_Python` uses Darts
2.  **Extract regime-aware LSTM code** - Created `RegimeAwareLSTM_Python/` template
3.  **Integrate Prophet + DCA** - Created `ProphetDCA_Python/` template
4. ⏳ **Test on production data** - Use `data/production/` datasets

## Implementation Status

### Completed: Regime-Aware LSTM Template

**Location:** `RegimeAwareLSTM_Python/`

**Features:**
-  PyTorch-based LSTM with regime embeddings
-  Compares vanilla LSTM vs regime-aware LSTM
-  Regime detection (simple quantile-based, extensible to Markov switching)
-  Side-by-side performance comparison
-  Integration with `src` utilities

**Key Code:**
- `RegimeAwareLSTM` class with embedding layer
- Augments input features with regime embeddings
- Shows improved performance when regimes are present

**Usage:**
```bash
cd RegimeAwareLSTM_Python
python main.py
```

**Outputs:**
- `regime_lstm_comparison.png` - Comparison plots
- `regime_lstm_predictions.csv` - Predictions from both models
- `regime_lstm_metrics.csv` - Performance metrics

---

### Completed: Prophet + DCA Integration Template

**Location:** `ProphetDCA_Python/`

**Features:**
-  Direct comparison of Prophet vs DCA forecasts
-  Side-by-side visualization
-  Performance metrics comparison
-  Uses existing `models/dca/` models
-  Integration with `src` utilities

**Key Code:**
- Fits Prophet model
- Fits DCA model (hyperbolic/exponential/harmonic)
- Compares forecasts and metrics
- Unified visualization

**Usage:**
```bash
cd ProphetDCA_Python
python main.py
```

**Outputs:**
- `prophet_dca_comparison.png` - Side-by-side forecast comparison
- `prophet_dca_forecast.csv` - Forecast values from both methods
- `prophet_dca_metrics.csv` - Performance comparison metrics

