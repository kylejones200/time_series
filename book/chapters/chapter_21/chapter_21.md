# Advanced Financial Time Series Analysis

## Chapter Overview

Financial time series present unique challenges and opportunities. Unlike many other domains, financial data often exhibits non-stationarity, complex dependencies, and relationships that require specialized econometric methods. This chapter explores advanced techniques for analyzing financial time series, including cointegration analysis, Vector Autoregression (VAR), copula modeling, and causal inference methods.

### Learning Objectives

By the end of this chapter, you will be able to:

- Understand cointegration and test for long-term relationships between non-stationary series
- Build and interpret Vector Autoregression (VAR) models
- Use Granger causality tests to identify causal relationships
- Analyze impulse response functions and forecast error variance decomposition
- Model dependencies using copulas
- Apply these methods to real financial data (commodities, stocks, economic indicators)

### Why Financial Time Series Are Different

Financial time series have several distinctive characteristics:

- **Non-Stationarity**: Prices and levels are often non-stationary (unit roots)
- **Long-Term Relationships**: Some series move together in the long run (cointegration)
- **Dynamic Interactions**: Variables influence each other over time
- **Non-Linear Dependencies**: Relationships may be non-linear and asymmetric
- **High Volatility**: Financial markets exhibit volatility clustering

---

## 21.1 Cointegration Analysis

### Understanding Cointegration

**Cointegration** occurs when two or more non-stationary time series have a stationary linear combination. This means that while individual series may drift apart in the short term, they maintain a long-term equilibrium relationship.

**Key Concepts:**

- **Non-Stationary Series**: Series with unit roots (e.g., stock prices, commodity prices)
- **Stationary Combination**: A linear combination of non-stationary series that is stationary
- **Long-Term Equilibrium**: Cointegrated series cannot drift too far apart permanently

### Why Cointegration Matters

Cointegration is crucial for:

- **Pairs Trading**: Identifying assets that move together
- **Hedging Strategies**: Finding assets that maintain stable relationships
- **Risk Management**: Understanding long-term dependencies
- **Forecasting**: Using cointegration relationships to improve forecasts

### Testing for Cointegration: Engle-Granger Method

The Engle-Granger two-step method tests for cointegration:

1. **Step 1**: Estimate a linear regression between the series
2. **Step 2**: Test the residuals for stationarity using the ADF test

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.api import OLS, add_constant

# Simulate time series data
np.random.seed(42)
n = 200
t = np.arange(n)

# Non-stationary series with a unit root
y1 = np.cumsum(np.random.normal(size=n))
# Another non-stationary series with a unit root
y2 = 0.5 * np.cumsum(np.random.normal(size=n)) + 10
# Stationary series
y3 = np.sin(t / 10) + np.random.normal(scale=0.5, size=n)

# Create a DataFrame
data = pd.DataFrame({"y1": y1, "y2": y2, "y3": y3})

# Plot the series
data.plot(subplots=True, figsize=(10, 8), title="Simulated Time Series")
plt.show()

# Function to perform ADF test
def adf_test(series, name):
    result = adfuller(series)
    print(f"ADF Test for {name}:")
    print(f"Test Statistic: {result[0]:.4f}")
    print(f"P-Value: {result[1]:.4f}")
    if result[1] > 0.05:
        print(f"{name} has a unit root (non-stationary).\n")
    else:
        print(f"{name} is stationary.\n")

# Perform ADF test on individual series
adf_test(data["y1"], "y1")
adf_test(data["y2"], "y2")
adf_test(data["y3"], "y3")

# Step 1: Estimate linear regression
X = add_constant(data["y2"])
model = OLS(data["y1"], X).fit()
residuals = model.resid

# Step 2: Test residuals for stationarity (cointegration)
adf_test(residuals, "Residuals of y1 ~ y2")

# Alternatively, use the coint function for direct testing
coint_stat, p_value, critical_values = coint(data["y1"], data["y2"])
print("Engle-Granger Cointegration Test:")
print(f"Test Statistic: {coint_stat:.4f}")
print(f"P-Value: {p_value:.4f}")
print(f"Critical Values: {critical_values}")
if p_value < 0.05:
    print("y1 and y2 are cointegrated.\n")
else:
    print("y1 and y2 are not cointegrated.\n")

# Plot the residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals, label="Residuals", color="blue")
plt.axhline(0, linestyle="--", color="red", label="Zero Line")
plt.title("Residuals of Linear Regression (y1 ~ y2)")
plt.xlabel("Time")
plt.ylabel("Residual Value")
plt.legend()
plt.grid()
plt.show()
```

**Interpreting Cointegration Tests:**

- **P-Value < 0.05**: Series are cointegrated (reject null of no cointegration)
- **P-Value ≥ 0.05**: Series are not cointegrated
- **Residuals Stationary**: If residuals are stationary, the series are cointegrated

### Real-World Example: Gold and Oil

```python
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import adfuller, coint

def get_data(start_date="2020-01-01", end_date="2024-02-04"):
    """Download gold and oil futures data."""
    print("Downloading Gold data...")
    gold = yf.download('GC=F', start=start_date, end=end_date)
    print("Downloading Oil data...")
    oil = yf.download('CL=F', start=start_date, end=end_date)

    # Create DataFrame with close prices
    merged_data = pd.merge(
        gold['Close'],
        oil['Close'],
        left_index=True,
        right_index=True,
        how='inner',
        suffixes=('_Gold', '_Oil')
    )
    merged_data.columns = ['Gold', 'Oil']
    return merged_data

def adf_test(series, name):
    """Perform Augmented Dickey-Fuller test."""
    result = adfuller(series)
    print(f"ADF Test for {name}:")
    print(f"Test Statistic: {result[0]:.4f}")
    print(f"P-Value: {result[1]:.4f}")
    if result[1] > 0.05:
        print(f"{name} has a unit root (non-stationary).\n")
    else:
        print(f"{name} is stationary.\n")

# Get the data
data = get_data()

# Plot the series
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Gold'], label='Gold', color='gold')
plt.plot(data.index, data['Oil'], label='Oil', color='black')
plt.title('Gold and Oil Futures Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Test for stationarity
adf_test(data['Gold'], 'Gold')
adf_test(data['Oil'], 'Oil')

# Perform linear regression
X = add_constant(data['Oil'])
model = OLS(data['Gold'], X).fit()
residuals = model.resid

# Print regression results
print("\nRegression Results:")
print(model.summary().tables[1])

# Test residuals for cointegration
adf_test(residuals, "Residuals of Gold ~ Oil")

# Direct cointegration test
coint_stat, p_value, critical_values = coint(data['Gold'], data['Oil'])
print("\nEngle-Granger Cointegration Test:")
print(f"Test Statistic: {coint_stat:.4f}")
print(f"P-Value: {p_value:.4f}")
print("Critical Values:")
print(f"1%: {critical_values[0]:.4f}")
print(f"5%: {critical_values[1]:.4f}")
print(f"10%: {critical_values[2]:.4f}")
if p_value < 0.05:
    print("Gold and Oil are cointegrated.\n")
else:
    print("Gold and Oil are not cointegrated.\n")

# Plot residuals
plt.figure(figsize=(12, 6))
plt.plot(data.index, residuals, label='Residuals', color='blue')
plt.axhline(0, linestyle='--', color='red', label='Zero Line')
plt.title('Residuals of Linear Regression (Gold ~ Oil)')
plt.xlabel('Date')
plt.ylabel('Residual Value')
plt.legend()
plt.grid(True)
plt.show()
```

**Key Points:**

- Both Gold and Oil prices are typically non-stationary
- If they are cointegrated, there's a long-term relationship
- The residuals from the regression should be stationary if cointegrated
- Cointegration doesn't imply causation—it indicates a long-term equilibrium

---

## 21.2 Vector Autoregression (VAR)

### Understanding VAR Models

**Vector Autoregression (VAR)** models multiple time series where each variable is a linear function of past values of itself and past values of all other variables in the system.

**Key Features:**

- **Multivariate**: Models multiple time series simultaneously
- **Endogenous**: All variables are treated as endogenous (dependent)
- **Dynamic**: Captures dynamic interactions between variables
- **No Structural Assumptions**: No need to specify which variables are exogenous

### VAR Model Specification

A VAR(p) model with k variables:

\[
\mathbf{y}_t = \mathbf{c} + \mathbf{\Phi}_1 \mathbf{y}_{t-1} + \mathbf{\Phi}_2 \mathbf{y}_{t-2} + \cdots + \mathbf{\Phi}_p \mathbf{y}_{t-p} + \mathbf{\epsilon}_t
\]

Where:
- \(\mathbf{y}_t\) is a k×1 vector of variables at time t
- \(\mathbf{c}\) is a k×1 vector of constants
- \(\mathbf{\Phi}_i\) are k×k coefficient matrices
- \(\mathbf{\epsilon}_t\) is a k×1 vector of error terms

### Building a VAR Model: Gold, NEM, and GDX

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR

# Minimalist plot style
plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True
})

# Load Data
def get_data():
    start_date = datetime.now() - pd.DateOffset(years=5)
    end_date = datetime.now()

    nem = yf.download('NEM', start=start_date, end=end_date, auto_adjust=False)[['Close']]
    gold = yf.download('GC=F', start=start_date, end=end_date, auto_adjust=False)[['Close']]
    gdx = yf.download('GDX', start=start_date, end=end_date, auto_adjust=False)[['Close']]

    # Rename columns
    nem.columns = ['NEM']
    gold.columns = ['Gold']
    gdx.columns = ['GDX']

    df = pd.concat([nem, gold, gdx], axis=1).dropna()
    return df

df = get_data()

# Monthly Log Returns (make data stationary)
monthly_data = df.resample('ME').mean()
monthly_log_returns = np.log(monthly_data).diff().dropna()

# ADF Stationarity Tests
def check_stationarity(series, name):
    result = adfuller(series)
    print(f"{name}: p-value = {result[1]:.4f}")
    print(f"{name} is {'stationary' if result[1] <= 0.05 else 'NOT stationary'}.")

print("\nMonthly Stationarity Tests:")
for col in monthly_log_returns.columns:
    check_stationarity(monthly_log_returns[col], col)

# Granger Causality Test: Gold → GDX
print("\nGranger Causality Test (Gold → GDX):")
grangercausalitytests(monthly_log_returns[['GDX', 'Gold']], maxlag=3, verbose=False)

# Fit VAR Model
model = VAR(monthly_log_returns)
lag_selection = model.select_order(12)
selected_lag = lag_selection.aic
fitted_model = model.fit(selected_lag)

print("\nVAR Model Summary:")
print(fitted_model.summary())

# Impulse Response Function: Gold → GDX
irf = fitted_model.irf(12)
fig = irf.plot(impulse='Gold', response='GDX')
plt.suptitle("Impulse Response: Monthly Gold Shock → GDX", fontsize=14)
plt.tight_layout()
plt.show()

# Forecast Error Variance Decomposition
fevd = fitted_model.fevd(12)

def plot_fevd_stacked(fevd, target_var, fitted_model):
    """Plot stacked area chart for FEVD."""
    idx = fitted_model.names.index(target_var)
    decomp = fevd.decomp[:, idx, :]
    months = np.arange(1, decomp.shape[0] + 1)

    plt.figure(figsize=(10, 6))
    plt.stackplot(months, decomp.T, labels=fitted_model.names)
    plt.title(f"FEVD - Stacked Area Chart: {target_var}")
    plt.xlabel("Months Ahead")
    plt.ylabel("Fraction of Variance")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

plot_fevd_stacked(fevd, 'GDX', fitted_model)
plot_fevd_stacked(fevd, 'Gold', fitted_model)
plot_fevd_stacked(fevd, 'NEM', fitted_model)
```

**Key Components:**

1. **Stationarity**: VAR requires stationary data—use log returns or differences
2. **Lag Selection**: Use information criteria (AIC, BIC) to select optimal lag length
3. **Granger Causality**: Tests whether one variable helps predict another
4. **Impulse Response Functions (IRF)**: Show how a shock to one variable affects others over time
5. **Forecast Error Variance Decomposition (FEVD)**: Shows what fraction of forecast error variance is due to each variable

### Granger Causality

**Granger Causality** tests whether past values of one variable help predict another variable beyond what past values of that variable itself can predict.

**Interpretation:**

- **P-Value < 0.05**: Variable X Granger-causes variable Y
- **P-Value ≥ 0.05**: No Granger causality

**Important Note:** Granger causality is about **predictive causality**, not true causation. It indicates that one variable's past values contain information useful for forecasting another variable.

### Impulse Response Functions (IRF)

IRFs show how a one-time shock to one variable affects all variables in the system over time.

**Interpretation:**

- **Positive Response**: Variable increases after the shock
- **Negative Response**: Variable decreases after the shock
- **Persistence**: How long the effect lasts
- **Magnitude**: Size of the response

### Forecast Error Variance Decomposition (FEVD)

FEVD shows what fraction of the forecast error variance for each variable is due to shocks to each variable in the system.

**Use Cases:**

- Understanding which variables drive forecast uncertainty
- Identifying the most important sources of variation
- Assessing the relative importance of different shocks

---

## 21.3 Copula Modeling

### Understanding Copulas

**Copulas** are functions that link univariate marginal distributions to form multivariate distributions. They allow us to model dependencies between variables separately from their marginal distributions.

**Key Advantages:**

- **Flexibility**: Model different types of dependencies (symmetric, asymmetric, tail dependencies)
- **Separation**: Separate modeling of margins and dependence structure
- **Non-Linear**: Capture non-linear dependencies
- **Tail Dependence**: Model extreme co-movements

### Types of Copulas

1. **Gaussian Copula**: Symmetric, no tail dependence
2. **Student-t Copula**: Symmetric, tail dependence
3. **Clayton Copula**: Asymmetric, lower tail dependence
4. **Gumbel Copula**: Asymmetric, upper tail dependence

### Example: Modeling Stock Returns and Interest Rates

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from copulas.bivariate import Clayton

np.random.seed(42)

# Simulate stock returns and interest rates
time_steps = 500
stock_returns = np.random.normal(0, 1, time_steps)
interest_rates = 0.5 * stock_returns + np.random.normal(0, 1, time_steps)

data = pd.DataFrame({
    'Stock Returns': stock_returns, 
    'Interest Rates': interest_rates
})

# Transform to uniform scale (rank transformation)
u = stats.rankdata(data['Stock Returns']) / (time_steps + 1)
v = stats.rankdata(data['Interest Rates']) / (time_steps + 1)

# Fit a Clayton copula
copula = Clayton()
copula.fit(pd.DataFrame({'u': u, 'v': v}))

# Simulate future dependencies
u_future = np.random.uniform(size=100)
v_future = copula.inverse_transform(pd.DataFrame({'u': u_future}))

# Transform back to original scale
returns_forecast = np.quantile(data['Stock Returns'], u_future)
rates_forecast = np.quantile(data['Interest Rates'], v_future['v'])

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(returns_forecast, rates_forecast, alpha=0.5, color='steelblue')
plt.xlabel("Forecasted Stock Returns")
plt.ylabel("Forecasted Interest Rates")
plt.title("Stock Returns vs. Interest Rates (Copula Forecast)")
plt.tight_layout()
plt.show()
```

### Example: Inflation and Unemployment

```python
from copulas.bivariate import StudentT

np.random.seed(42)

# Simulate inflation and unemployment
inflation = np.random.normal(2, 1, time_steps)
unemployment = -0.7 * inflation + np.random.normal(0, 1, time_steps)

data = pd.DataFrame({
    'Inflation': inflation, 
    'Unemployment': unemployment
})

# Transform to uniform scale
u = stats.rankdata(data['Inflation']) / (time_steps + 1)
v = stats.rankdata(data['Unemployment']) / (time_steps + 1)

# Fit a t-Copula
copula = StudentT()
copula.fit(pd.DataFrame({'u': u, 'v': v}))

# Simulate future dependencies
u_future = np.random.uniform(size=100)
v_future = copula.inverse_transform(pd.DataFrame({'u': u_future}))

# Transform back to original scale
inflation_forecast = np.quantile(data['Inflation'], u_future)
unemployment_forecast = np.quantile(data['Unemployment'], v_future['v'])

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(inflation_forecast, unemployment_forecast, alpha=0.5, color='steelblue')
plt.xlabel("Forecasted Inflation")
plt.ylabel("Forecasted Unemployment")
plt.title("Inflation vs. Unemployment (t-Copula Forecast)")
plt.tight_layout()
plt.show()
```

**When to Use Copulas:**

- **Non-Linear Dependencies**: When relationships are non-linear
- **Tail Dependencies**: When you need to model extreme co-movements
- **Risk Management**: For portfolio risk analysis
- **Stress Testing**: Simulating extreme scenarios

---

## 21.4 Working with FRED Economic Data

### Accessing FRED Data

The Federal Reserve Economic Data (FRED) provides a vast collection of economic time series.

```python
import pandas_datareader.data as web
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Set the date range
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.today()

# Fetch data from FRED: Mortgage-Backed Securities
df = web.DataReader("DRSFRMACBS", "fred", start, end)

# Drop any missing values
df = df.dropna().reset_index()
df.columns = ["time", "value"]
df["time"] = pd.to_datetime(df["time"])

# Visualize the FRED time series
plt.figure(figsize=(10, 6))
plt.plot(df["time"], df["value"], label="FRED Series: DRSFRMACBS")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("FRED Time Series: Mortgage-Backed Securities")
plt.legend()
plt.show()
```

**FRED Series Examples:**

- **DRSFRMACBS**: Mortgage-Backed Securities
- **UNRATE**: Unemployment Rate
- **GDP**: Gross Domestic Product
- **CPIAUCSL**: Consumer Price Index
- **FEDFUNDS**: Federal Funds Rate

---

## 21.5 Best Practices for Financial Time Series

### 1. Data Preprocessing

- **Stationarity**: Always check for stationarity before modeling
- **Log Returns**: Use log returns for prices to achieve stationarity
- **Differencing**: Difference non-stationary series if needed
- **Outliers**: Handle outliers carefully (they may be real market events)

### 2. Model Selection

- **Information Criteria**: Use AIC/BIC for lag selection in VAR
- **Diagnostic Tests**: Check residuals for autocorrelation, heteroscedasticity
- **Stability**: Ensure VAR models are stable (eigenvalues < 1)

### 3. Interpretation

- **Cointegration**: Indicates long-term relationships, not causation
- **Granger Causality**: Predictive causality, not true causation
- **IRFs**: Show dynamic responses, not structural relationships
- **FEVD**: Shows variance contributions, not causal importance

### 4. Common Pitfalls

- **Spurious Regression**: Regressing non-stationary series without cointegration
- **Overfitting**: Too many lags or variables in VAR models
- **Structural Breaks**: Financial relationships may change over time
- **Data Mining**: Testing many relationships increases false discovery

---

## 21.6 Summary

This chapter introduced advanced methods for financial time series analysis:

**Key Concepts:**

1. **Cointegration**: Long-term equilibrium relationships between non-stationary series
2. **VAR Models**: Multivariate models capturing dynamic interactions
3. **Granger Causality**: Predictive causality testing
4. **Impulse Response Functions**: Dynamic responses to shocks
5. **Forecast Error Variance Decomposition**: Variance attribution
6. **Copulas**: Modeling dependencies separately from margins

**When to Use Each Method:**

- **Cointegration**: Testing long-term relationships, pairs trading
- **VAR**: Modeling multiple interacting time series
- **Granger Causality**: Identifying predictive relationships
- **Copulas**: Modeling non-linear dependencies, risk analysis

**Best Practices:**

- Always check for stationarity
- Use log returns for prices
- Select lags using information criteria
- Interpret results carefully (causation vs. correlation)
- Be aware of structural breaks

---

## Exercises

1. **Cointegration Analysis**: Test for cointegration between two commodity prices of your choice. Interpret the results.

2. **VAR Model**: Build a VAR model with three financial time series. Select optimal lags and interpret the results.

3. **Granger Causality**: Test for Granger causality between stock returns and interest rates. What do the results tell you?

4. **Impulse Response**: Generate IRFs for a VAR model. Interpret how shocks propagate through the system.

5. **Copula Modeling**: Fit a copula to model dependencies between two financial variables. Compare different copula types.

---

## References and Further Reading

- Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction: representation, estimation, and testing. Econometrica, 55(2), 251-276.
- Sims, C. A. (1980). Macroeconomics and reality. Econometrica, 48(1), 1-48.
- Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges. Publications de l'Institut de Statistique de l'Université de Paris, 8, 229-231.
- Lütkepohl, H. (2005). New introduction to multiple time series analysis. Springer.
- Nelsen, R. B. (2006). An introduction to copulas. Springer.

