# Uncertainty Quantification in Forecasting

## Chapter Overview

Forecasting is inherently uncertain. A point forecast—a single predicted value—tells only part of the story. Understanding and quantifying uncertainty is crucial for making informed decisions, managing risk, and communicating forecast reliability. This chapter explores methods for quantifying uncertainty in time series forecasts, including confidence intervals, prediction intervals, and bootstrap methods.

### Learning Objectives

By the end of this chapter, you will be able to:

- Understand the difference between confidence intervals and prediction intervals
- Generate confidence intervals for ARIMA forecasts
- Implement bootstrap methods for uncertainty quantification
- Visualize forecast uncertainty effectively
- Evaluate forecast intervals using proper metrics
- Apply uncertainty quantification to real-world forecasting problems

### Why Uncertainty Matters

Point forecasts alone are insufficient because:

- **Decision Making**: Decisions often depend on worst-case or best-case scenarios
- **Risk Management**: Understanding uncertainty helps manage risk
- **Resource Planning**: Uncertainty informs capacity planning and inventory management
- **Communication**: Stakeholders need to understand forecast reliability
- **Model Evaluation**: Uncertainty measures help evaluate model quality

---

## 20.1 Types of Forecast Intervals

### Confidence Intervals vs. Prediction Intervals

**Confidence Intervals** represent uncertainty about the **forecast mean**—the expected value of the forecast. They answer: "What is the range of likely values for the forecast mean?"

**Prediction Intervals** represent uncertainty about **individual future observations**. They answer: "What is the range of likely values for actual future observations?"

**Key Difference:**

- **Confidence Intervals**: Narrower, represent uncertainty about the mean
- **Prediction Intervals**: Wider, include both model uncertainty and observation noise

In practice, **prediction intervals** are more useful for most applications because they account for both model uncertainty and the inherent variability in future observations.

### Coverage Probability

The **coverage probability** (or confidence level) indicates the probability that the true value falls within the interval. Common choices:

- **90%**: Wider intervals, more conservative
- **95%**: Standard choice, balances precision and coverage
- **99%**: Very wide intervals, very conservative

---

## 20.2 Confidence Intervals for ARIMA Models

ARIMA models provide built-in methods for computing confidence intervals based on the model's estimated variance.

### Basic ARIMA Forecast with Confidence Intervals

```python
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from pmdarima import auto_arima

warnings.filterwarnings("ignore")

def load_data(url):
    """Load and preprocess time series data."""
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Resample to consistent frequency (hourly in this example)
    df = df.resample('h').mean().asfreq('h')
    
    # Interpolate missing values
    df['values'] = df['values'].interpolate()
    
    return df, StandardScaler()

def forecast_arima(data, order, steps=48, confidence=0.95):
    """
    Generate ARIMA forecast with confidence intervals.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    order : tuple
        ARIMA order (p, d, q)
    steps : int
        Number of steps ahead to forecast
    confidence : float
        Confidence level (e.g., 0.95 for 95%)
    
    Returns:
    --------
    forecasts : pd.Series
        Point forecasts
    lower : pd.Series
        Lower bound of confidence interval
    upper : pd.Series
        Upper bound of confidence interval
    """
    # Fit ARIMA model
    model = ARIMA(data, order=order).fit()
    
    # Get forecast with confidence intervals
    forecast_result = model.get_forecast(steps=steps)
    forecasts = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=1 - confidence)
    
    return forecasts, conf_int.iloc[:, 0], conf_int.iloc[:, 1]

# Load data
url = "https://raw.githubusercontent.com/kylejones200/time_series/refs/heads/main/ercot_load_data.csv"
df, scaler = load_data(url)

# IMPORTANT: Split data BEFORE scaling to avoid data leakage
train_raw = df['values'].iloc[:-48]
test_raw = df['values'].iloc[-48:]

# Fit scaler on training data only
train_scaled = pd.Series(
    scaler.fit_transform(train_raw.values.reshape(-1, 1)).flatten(),
    index=train_raw.index
)

# Transform test data using training scaler
test_scaled = pd.Series(
    scaler.transform(test_raw.values.reshape(-1, 1)).flatten(),
    index=test_raw.index
)

# Automatically select best ARIMA model
auto_model = auto_arima(
    train_scaled, 
    seasonal=False, 
    trace=False,
    suppress_warnings=True, 
    stepwise=True
)
best_order = auto_model.order

print(f"Best ARIMA order: {best_order}")

# Generate forecasts with confidence intervals
forecasts_scaled, lower_scaled, upper_scaled = forecast_arima(
    train_scaled, 
    best_order, 
    steps=48,
    confidence=0.95
)

# Inverse transform to original scale
def inverse_transform(data):
    return scaler.inverse_transform(np.array(data).reshape(-1, 1)).flatten()

forecasts = inverse_transform(forecasts_scaled)
lower = inverse_transform(lower_scaled)
upper = inverse_transform(upper_scaled)
test_actual = inverse_transform(test_scaled)

# Create test series for plotting
test_series = pd.Series(test_actual, index=test_raw.index)
```

**Key Points:**

1. **Data Leakage Prevention**: Split data before scaling—fit scaler on training data only
2. **Auto ARIMA**: Automatically selects optimal ARIMA parameters
3. **Confidence Intervals**: Built into ARIMA's `get_forecast()` method
4. **Inverse Transform**: Convert scaled forecasts back to original scale

### Visualizing Forecasts with Confidence Intervals

```python
def plot_forecast(historical, test, forecasts, lower, upper, title=""):
    """
    Plot historical data, test data, forecasts, and confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot historical data (lighter, for context)
    ax.plot(historical.index, historical.values, 'k-', alpha=0.3, linewidth=0.8)
    
    # Plot actual test values
    ax.plot(test.index, test.values, 'g-', linewidth=1.5, label='Actual')
    
    # Plot point forecasts
    ax.plot(test.index, forecasts, 'r-', linewidth=1.5, label='Forecast')
    
    # Fill confidence interval
    ax.fill_between(test.index, lower, upper, color='r', alpha=0.15, label='95% CI')
    
    # Vertical line separating train and test
    ax.axvline(test.index[0], color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

# Plot ARIMA forecast with confidence intervals
plot_forecast(
    df['values'], 
    test_series, 
    forecasts, 
    lower, 
    upper, 
    "ARIMA Forecast with 95% Confidence Intervals"
)
```

**Visualization Best Practices:**

- **Historical Context**: Show historical data for context
- **Clear Separation**: Mark the train/test split clearly
- **Confidence Intervals**: Use fill_between for easy interpretation
- **Legend**: Label all components clearly

---

## 20.3 Bootstrap Methods for Uncertainty Quantification

Bootstrap methods provide an alternative approach to uncertainty quantification that doesn't rely on model assumptions. They work by:

1. **Resampling**: Creating many bootstrap samples from the training data
2. **Refitting**: Fitting the model to each bootstrap sample
3. **Forecasting**: Generating forecasts from each fitted model
4. **Aggregating**: Computing intervals from the distribution of forecasts

### Bootstrap Confidence Intervals

```python
def bootstrap_ci(model_order, data, steps=48, n_bootstraps=100, confidence=0.95):
    """
    Generate bootstrap confidence intervals for ARIMA forecasts.
    
    Parameters:
    -----------
    model_order : tuple
        ARIMA order (p, d, q)
    data : pd.Series
        Training time series data
    steps : int
        Number of steps ahead to forecast
    n_bootstraps : int
        Number of bootstrap samples
    confidence : float
        Confidence level
    
    Returns:
    --------
    mean_forecasts : np.array
        Mean of bootstrap forecasts
    lower : np.array
        Lower bound of confidence interval
    upper : np.array
        Upper bound of confidence interval
    """
    forecasts = []
    
    for i in range(n_bootstraps):
        try:
            # Resample with replacement
            sample = data.sample(n=len(data), replace=True).sort_index()
            
            # Fit model to bootstrap sample
            model = ARIMA(sample, order=model_order).fit()
            
            # Generate forecast
            forecast = model.forecast(steps=steps).values
            forecasts.append(forecast)
        except:
            # Skip if model fitting fails
            continue
    
    if not forecasts:
        raise RuntimeError("All bootstrap iterations failed")
    
    # Convert to array
    forecasts = np.array(forecasts)
    
    # Compute confidence intervals
    alpha = (1 - confidence) / 2
    mean_forecasts = np.mean(forecasts, axis=0)
    lower = np.percentile(forecasts, alpha * 100, axis=0)
    upper = np.percentile(forecasts, (1 - alpha) * 100, axis=0)
    
    return mean_forecasts, lower, upper

# Generate bootstrap confidence intervals
boot_forecasts_scaled, boot_lower_scaled, boot_upper_scaled = bootstrap_ci(
    best_order, 
    train_scaled, 
    steps=48, 
    n_bootstraps=50  # Use more for better estimates (50 for speed)
)

# Inverse transform
boot_forecasts = inverse_transform(boot_forecasts_scaled)
boot_lower = inverse_transform(boot_lower_scaled)
boot_upper = inverse_transform(boot_upper_scaled)

# Plot bootstrap forecast
plot_forecast(
    df['values'], 
    test_series, 
    boot_forecasts, 
    boot_lower, 
    boot_upper, 
    "Bootstrap Forecast with 95% Confidence Intervals"
)
```

**Advantages of Bootstrap:**

- **Model-Free**: Doesn't rely on distributional assumptions
- **Flexible**: Works with any model
- **Robust**: Handles non-normal errors
- **Intuitive**: Easy to understand and explain

**Disadvantages:**

- **Computational Cost**: Requires refitting models many times
- **Time Series Challenges**: Standard bootstrap doesn't preserve temporal structure
- **Sample Size**: Requires sufficient data for reliable estimates

### Time Series Bootstrap Considerations

Standard bootstrap (sampling with replacement) breaks the temporal structure of time series. For time series, consider:

1. **Block Bootstrap**: Sample blocks of consecutive observations
2. **Residual Bootstrap**: Bootstrap residuals, then reconstruct series
3. **Moving Block Bootstrap**: Overlapping blocks

For simplicity, the example above uses standard bootstrap, but in practice, consider time series-specific bootstrap methods.

---

## 20.4 Comparing Methods

### ARIMA Confidence Intervals vs. Bootstrap

**ARIMA Confidence Intervals:**

- **Pros**: Fast, built into model, based on model assumptions
- **Cons**: Relies on model assumptions (normality, homoscedasticity)
- **Use When**: Model assumptions are reasonable, speed is important

**Bootstrap Intervals:**

- **Pros**: Model-free, robust to assumptions, flexible
- **Cons**: Computationally expensive, requires many iterations
- **Use When**: Model assumptions are questionable, you want robustness

### Visual Comparison

```python
# Compare both methods
fig, ax = plt.subplots(figsize=(14, 6))

# Historical data
ax.plot(df['values'].index, df['values'].values, 'k-', alpha=0.3, linewidth=0.8)

# Test data
ax.plot(test_series.index, test_series.values, 'g-', linewidth=2, label='Actual')

# ARIMA forecast
ax.plot(test_series.index, forecasts, 'r-', linewidth=1.5, label='ARIMA Forecast')
ax.fill_between(test_series.index, lower, upper, color='r', alpha=0.1, label='ARIMA 95% CI')

# Bootstrap forecast
ax.plot(test_series.index, boot_forecasts, 'b--', linewidth=1.5, label='Bootstrap Forecast')
ax.fill_between(test_series.index, boot_lower, boot_upper, color='b', alpha=0.1, label='Bootstrap 95% CI')

ax.axvline(test_series.index[0], color='k', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Comparison: ARIMA vs. Bootstrap Confidence Intervals')
ax.legend(frameon=False, loc='best')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
```

---

## 20.5 Evaluating Forecast Intervals

### Coverage Metrics

**Coverage** measures the proportion of actual values that fall within the forecast intervals:

```python
def calculate_coverage(actual, lower, upper):
    """
    Calculate the coverage of forecast intervals.
    
    Returns:
    --------
    coverage : float
        Proportion of actual values within intervals
    """
    within_interval = (actual >= lower) & (actual <= upper)
    coverage = within_interval.mean()
    return coverage

# Calculate coverage for both methods
arima_coverage = calculate_coverage(test_series.values, lower, upper)
bootstrap_coverage = calculate_coverage(test_series.values, boot_lower, boot_upper)

print(f"ARIMA 95% CI Coverage: {arima_coverage:.2%}")
print(f"Bootstrap 95% CI Coverage: {bootstrap_coverage:.2%}")
```

**Ideal Coverage:**

- For 95% intervals, coverage should be close to 95%
- **Under-coverage** (< 95%): Intervals are too narrow
- **Over-coverage** (> 95%): Intervals are too wide

### Interval Width

**Mean Interval Width** measures the average width of forecast intervals:

```python
def mean_interval_width(lower, upper):
    """Calculate mean width of forecast intervals."""
    return np.mean(upper - lower)

arima_width = mean_interval_width(lower, upper)
bootstrap_width = mean_interval_width(boot_lower, boot_upper)

print(f"ARIMA Mean Interval Width: {arima_width:.2f}")
print(f"Bootstrap Mean Interval Width: {bootstrap_width:.2f}")
```

**Trade-off:**

- **Narrow Intervals**: More precise but may have lower coverage
- **Wide Intervals**: Higher coverage but less informative

---

## 20.6 Best Practices

### 1. Data Preprocessing

- **Split Before Scaling**: Always split data before fitting scalers to avoid leakage
- **Handle Missing Values**: Interpolate or impute missing values appropriately
- **Check Stationarity**: Ensure data is stationary or use appropriate differencing

### 2. Model Selection

- **Auto ARIMA**: Use automatic model selection when appropriate
- **Model Diagnostics**: Check residuals for normality and homoscedasticity
- **Cross-Validation**: Use time series cross-validation for model evaluation

### 3. Uncertainty Quantification

- **Choose Method**: ARIMA intervals for speed, bootstrap for robustness
- **Adequate Bootstrap Samples**: Use at least 100-1000 bootstrap samples
- **Multiple Methods**: Compare different methods when possible

### 4. Visualization

- **Clear Labels**: Label all components clearly
- **Confidence Level**: Always indicate the confidence level
- **Context**: Show historical data for context
- **Comparison**: Compare multiple methods when relevant

### 5. Communication

- **Explain Intervals**: Help stakeholders understand what intervals mean
- **Coverage**: Report actual coverage, not just nominal coverage
- **Limitations**: Acknowledge assumptions and limitations

---

## 20.7 Real-World Applications

### Application 1: Energy Load Forecasting

Energy companies need uncertainty estimates for:
- **Capacity Planning**: Ensure sufficient generation capacity
- **Risk Management**: Understand worst-case scenarios
- **Pricing**: Set prices that account for uncertainty

### Application 2: Demand Forecasting

Retailers use uncertainty for:
- **Inventory Management**: Stock levels based on uncertainty
- **Supply Chain**: Order quantities considering uncertainty
- **Promotions**: Plan promotions accounting for forecast uncertainty

### Application 3: Financial Forecasting

Financial applications require uncertainty for:
- **Risk Assessment**: Quantify financial risk
- **Portfolio Management**: Diversify based on uncertainty
- **Regulatory Compliance**: Meet risk reporting requirements

---

## 20.8 Summary

Uncertainty quantification is essential for practical forecasting:

**Key Concepts:**

1. **Confidence Intervals**: Uncertainty about forecast mean
2. **Prediction Intervals**: Uncertainty about future observations
3. **Bootstrap Methods**: Model-free uncertainty quantification
4. **Coverage**: Proportion of actual values within intervals
5. **Interval Width**: Measure of forecast precision

**Methods:**

- **ARIMA Intervals**: Fast, model-based, assumption-dependent
- **Bootstrap Intervals**: Robust, model-free, computationally expensive

**Best Practices:**

- Always split data before scaling
- Use appropriate bootstrap methods for time series
- Evaluate intervals using coverage metrics
- Visualize uncertainty clearly
- Communicate uncertainty effectively

---

## Exercises

1. **ARIMA Confidence Intervals**: Generate ARIMA forecasts with confidence intervals for a time series of your choice. Evaluate coverage and interval width.

2. **Bootstrap Comparison**: Compare bootstrap intervals with different numbers of bootstrap samples (50, 100, 500, 1000). How does the number of samples affect the intervals?

3. **Coverage Analysis**: Calculate coverage for different confidence levels (90%, 95%, 99%). How does coverage change with confidence level?

4. **Method Comparison**: Compare ARIMA confidence intervals with bootstrap intervals on the same dataset. Which method provides better coverage?

5. **Visualization**: Create a comprehensive visualization showing historical data, point forecasts, and confidence intervals. Include coverage statistics.

---

## References and Further Reading

- Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: principles and practice. OTexts.
- Efron, B., & Tibshirani, R. J. (1994). An introduction to the bootstrap. CRC press.
- Chatfield, C. (2001). Time-series forecasting. CRC press.
- statsmodels Documentation: https://www.statsmodels.org/

