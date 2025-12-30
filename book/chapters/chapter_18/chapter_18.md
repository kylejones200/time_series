# Modern Forecasting Frameworks: The Nixtla Suite

## Chapter Overview

In this chapter, we explore the Nixtla suite of forecasting libraries, a comprehensive ecosystem for time series forecasting that spans classical statistical methods, machine learning, and deep learning approaches. The Nixtla suite provides production-ready tools that are optimized for performance, scalability, and ease of use.

### Learning Objectives

By the end of this chapter, you will be able to:

- Understand the components of the Nixtla forecasting ecosystem
- Use StatsForecast for classical time series models
- Apply MLForecast for machine learning-based forecasting
- Implement NeuralForecast for deep learning models
- Build hierarchical forecasting models with HierarchicalForecast
- Leverage parallel computing for scalable forecasting

### Why the Nixtla Suite?

Traditional time series forecasting often requires stitching together multiple libraries, each with different APIs and data formats. The Nixtla suite provides a unified interface across four powerful libraries:

1. **StatsForecast**: Fast statistical forecasting models
2. **MLForecast**: Machine learning models with automatic feature engineering
3. **NeuralForecast**: Deep learning architectures for time series
4. **HierarchicalForecast**: Coherent forecasting across aggregation levels

This unified approach reduces complexity while providing state-of-the-art performance and scalability.

---

## 18.1 Introduction to the Nixtla Ecosystem

The Nixtla suite is designed with several key principles:

- **Unified Data Format**: All libraries use the same DataFrame structure with `unique_id`, `ds` (date), and `y` (target) columns
- **Parallel Computing**: Built-in support for multi-core and distributed computing
- **Production Ready**: Optimized for speed and memory efficiency
- **Comprehensive**: Covers classical, ML, and deep learning approaches

### Installation

```bash
pip install statsforecast mlforecast neuralforecast hierarchicalforecast
```

---

## 18.2 StatsForecast: Statistical Forecasting

StatsForecast provides fast implementations of classical time series models, including ARIMA, Exponential Smoothing, and Theta methods. It's optimized for speed and can handle thousands of time series efficiently.

### Basic Usage

Let's start with a simple example using AutoARIMA:

```python
import os
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import matplotlib.pyplot as plt

# Set environment variable for unique_id column handling
os.environ['NIXTLA_ID_AS_COL'] = '1'

np.random.seed(42)

# Create sample time series data
dates = pd.date_range(start='2021-01-01', end='2022-12-31', freq='D')
values = np.cumsum(np.random.randn(len(dates))) + 100

# Prepare data in Nixtla format
df = pd.DataFrame({
    'unique_id': 'series1',
    'ds': dates,
    'y': values
})

# Initialize models
models = [AutoARIMA(season_length=7)]  # Weekly seasonality

# Create StatsForecast object with parallel processing
sf = StatsForecast(models=models, freq='D', n_jobs=-1)

# Generate forecasts
horizon = 14  # Forecast 14 days ahead
forecasts = sf.forecast(df=df, h=horizon)

# Filter for our series (if multiple series)
if 'unique_id' in forecasts.columns:
    forecasts = forecasts[forecasts['unique_id'] == 'series1']

# Visualize results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['ds'], df['y'], 'k-', linewidth=1.5, alpha=0.7, label='Historical')
ax.plot(forecasts['ds'], forecasts['AutoARIMA'], 'r-', linewidth=1.5, label='Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.legend(frameon=False, loc='best')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Evaluate forecast (using last horizon points as test set)
actual_values = df.iloc[-horizon:]
forecast_values = forecasts['AutoARIMA']
mse = np.mean((actual_values['y'].values - forecast_values.values) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(actual_values['y'].values - forecast_values.values))

print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")
```

**Key Points:**

- **Data Format**: The `unique_id`, `ds`, and `y` columns are required. This format allows handling multiple time series in a single DataFrame.
- **AutoARIMA**: Automatically selects the best ARIMA model parameters using information criteria.
- **Parallel Processing**: Setting `n_jobs=-1` uses all available CPU cores for faster computation.
- **Seasonality**: The `season_length=7` parameter indicates weekly seasonality for daily data.

### Available Models in StatsForecast

StatsForecast includes many classical models:

- **AutoARIMA**: Automatic ARIMA model selection
- **AutoETS**: Exponential Smoothing with automatic model selection
- **Theta**: Theta method for forecasting
- **Naive**: Simple naive forecast
- **SeasonalNaive**: Seasonal naive forecast
- **WindowAverage**: Moving average forecast

---

## 18.3 MLForecast: Machine Learning for Time Series

MLForecast combines machine learning models (like LightGBM, XGBoost) with automatic time series feature engineering. It automatically creates lag features, rolling statistics, and date-based features.

### Using MLForecast with LightGBM

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlforecast import MLForecast
from lightgbm import LGBMRegressor

# Create sample data
np.random.seed(42)
dates = pd.date_range(start='2021-01-01', end='2022-12-31', freq='D')
values = np.cumsum(np.random.randn(len(dates))) + 100  # Random walk

df = pd.DataFrame({
    'unique_id': 'series1',
    'ds': dates,
    'y': values
})

# Initialize MLForecast with automatic feature engineering
model = MLForecast(
    models=[LGBMRegressor()],
    freq='D',
    lags=[1, 7, 14],  # Lag features at 1, 7, and 14 days
    date_features=['dayofweek', 'day']  # Date-based features
)

# Fit the model
model.fit(df)

# Generate forecasts
horizon = 14
forecasts = model.predict(horizon)

print("Forecasts:")
print(forecasts.head())
print("\nForecast columns:")
print(forecasts.columns)

# Prepare data for visualization
historical_data = df[df['unique_id'] == 'series1']
forecast_data = forecasts[forecasts['unique_id'] == 'series1']

# Use the correct column name for predictions
prediction_col = 'LGBMRegressor'

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(historical_data['ds'], historical_data['y'], label='Historical Data')
plt.plot(forecast_data['ds'], forecast_data[prediction_col], label='Forecast', color='red')
plt.title('Time Series Forecast with MLForecast (LGBMRegressor)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Calculate and print forecast metrics
actual_values = historical_data.iloc[-horizon:]
forecast_values = forecast_data[prediction_col]
mse = np.mean((actual_values['y'].values - forecast_values.values) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(actual_values['y'].values - forecast_values.values))

print(f"\nForecast Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
```

**Key Features of MLForecast:**

1. **Automatic Feature Engineering**: Creates lag features, rolling statistics, and date features automatically
2. **Multiple Models**: Can ensemble multiple ML models
3. **Time Series Aware**: Handles time-based splits and cross-validation correctly
4. **Scalable**: Efficient for large datasets

**Feature Engineering in MLForecast:**

- **Lags**: Past values at specified time steps (e.g., `lags=[1, 7, 14]`)
- **Rolling Statistics**: Moving averages, standard deviations, etc.
- **Date Features**: Day of week, month, year, holidays, etc.
- **Custom Features**: You can add your own feature engineering functions

---

## 18.4 NeuralForecast: Deep Learning for Time Series

NeuralForecast provides state-of-the-art deep learning architectures for time series forecasting, including MLP, NHITS, N-BEATS, and Transformer-based models.

### Using NeuralForecast with MLP and NHITS

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP, NHITS
import pytorch_lightning as pl

# Disable logging to avoid verbose output
pl.utilities.rank_zero.rank_zero_only.rank = -1

# Generate sample data
np.random.seed(42)
dates = pd.date_range(start='2021-01-01', end='2022-12-31', freq='D')
values = np.cumsum(np.random.randn(len(dates))) + 100  # Random walk

df = pd.DataFrame({
    'unique_id': 'series1',
    'ds': dates,
    'y': values
})

# Set the forecast horizon
horizon = 14

# Initialize the NeuralForecast models
models = [
    MLP(h=horizon, input_size=30, trainer_kwargs={"logger": False}),
    NHITS(h=horizon, input_size=30, trainer_kwargs={"logger": False})
]

# Create the NeuralForecast object
nf = NeuralForecast(models=models, freq='D')

# Fit the model
nf.fit(df)

# Generate forecasts
forecasts = nf.predict()

# Prepare data for visualization
historical_data = df[df['unique_id'] == df['unique_id'].unique()[0]]
forecast_data = forecasts[forecasts['unique_id'] == df['unique_id'].unique()[0]]

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(historical_data['ds'], historical_data['y'], label='Historical Data')
plt.plot(forecast_data['ds'], forecast_data['MLP'], label='MLP Forecast', color='red')
plt.plot(forecast_data['ds'], forecast_data['NHITS'], label='NHITS Forecast', color='green')
plt.title('Time Series Forecast with NeuralForecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Calculate and print forecast metrics
actual_values = historical_data.iloc[-horizon:]['y'].values
mlp_forecast = forecast_data['MLP'].values
nhits_forecast = forecast_data['NHITS'].values

def calculate_metrics(actual, forecast):
    mse = np.mean((actual - forecast) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - forecast))
    return mse, rmse, mae

mlp_metrics = calculate_metrics(actual_values, mlp_forecast)
nhits_metrics = calculate_metrics(actual_values, nhits_forecast)

print("\nMLP Forecast Metrics:")
print(f"Mean Squared Error (MSE): {mlp_metrics[0]:.2f}")
print(f"Root Mean Squared Error (RMSE): {mlp_metrics[1]:.2f}")
print(f"Mean Absolute Error (MAE): {mlp_metrics[2]:.2f}")

print("\nNHITS Forecast Metrics:")
print(f"Mean Squared Error (MSE): {nhits_metrics[0]:.2f}")
print(f"Root Mean Squared Error (RMSE): {nhits_metrics[1]:.2f}")
print(f"Mean Absolute Error (MAE): {nhits_metrics[2]:.2f}")
```

**NeuralForecast Models:**

- **MLP**: Multi-layer perceptron for time series
- **NHITS**: Neural Hierarchical Interpolation for Time Series
- **N-BEATS**: Neural Basis Expansion Analysis
- **Transformer**: Transformer-based architectures
- **LSTM/GRU**: Recurrent neural networks

**Key Parameters:**

- `h`: Forecast horizon (number of steps ahead)
- `input_size`: Number of historical time steps to use as input
- `trainer_kwargs`: PyTorch Lightning trainer configuration

---

## 18.5 HierarchicalForecast: Coherent Hierarchical Forecasting

HierarchicalForecast addresses the challenge of forecasting time series that exist at multiple aggregation levels (e.g., total sales, by region, by store). It ensures that forecasts are coherent—that is, forecasts at higher levels equal the sum of forecasts at lower levels.

### Understanding Hierarchical Time Series

In many business contexts, you need to forecast at multiple levels:

- **Total**: Overall company sales
- **Region**: Sales by geographic region
- **Store**: Sales by individual store

The challenge is ensuring that the sum of store-level forecasts equals the region-level forecast, which equals the total forecast. This is called **coherence**.

### Example: Hierarchical Forecasting

```python
import pandas as pd
import numpy as np
from hierarchicalforecast.models import HierarchicalForecast
from hierarchicalforecast.utils import agg_series
from hierarchicalforecast.methods import BottomUp
from statsforecast.models import AutoARIMA
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Assume you have a DataFrame with hierarchical structure
# For demonstration, we'll create synthetic hierarchical data
df = pd.DataFrame({
    'date': pd.date_range(start='2020-01-01', end='2023-12-31', freq='Q'),
    'value': np.random.randn(16) * 100 + 1000
})

# Rename to Nixtla format
df = df.rename(columns={'date': 'ds', 'value': 'y'})
df['ds'] = pd.to_datetime(df['ds'])

# Create a simple hierarchy for demonstration
# In practice, this would come from your business structure
df['Total'] = 'GDP'
df['Region'] = np.random.choice(['East', 'West'], size=len(df))
df['State'] = np.random.choice(['State1', 'State2', 'State3', 'State4'], size=len(df))
df['unique_id'] = df['State']

# Define the hierarchy structure
# This dictionary maps parent nodes to their children
S = {
    'GDP': ['East', 'West'],
    'East': ['State1', 'State2'],
    'West': ['State3', 'State4']
}

# Visualize the hierarchy
G = nx.DiGraph(S)
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=3000, font_size=10, arrows=True)
plt.title("Hierarchical Structure of GDP")
plt.show()

# Aggregate the series to create all levels
Y_df, S_df = agg_series(df, S)

# Initialize the hierarchical forecast model
# BottomUp: Forecast at bottom level, then aggregate upward
hf_model = HierarchicalForecast(
    model=AutoARIMA(),
    reconciliation=BottomUp()
)

# Fit and forecast
hf_model.fit(Y_df=Y_df, S=S_df, freq='Q')  # Quarterly frequency
forecasts = hf_model.predict(h=4)  # Forecast 4 quarters ahead

print("Forecasts:")
print(forecasts.head())

# Plot forecasts for different levels
fig, axs = plt.subplots(3, 1, figsize=(12, 15))
levels = ['GDP', 'East', 'State1']

for i, level in enumerate(levels):
    historical = Y_df[Y_df['unique_id'] == level]
    forecast = forecasts[forecasts['unique_id'] == level]
    
    axs[i].plot(historical['ds'], historical['y'], label='Historical')
    axs[i].plot(forecast['ds'], forecast['y'], label='Forecast', color='red')
    axs[i].set_title(f'Forecast for {level}')
    axs[i].legend()
    axs[i].set_xlabel('Date')
    axs[i].set_ylabel('GDP')
    
plt.tight_layout()
plt.show()

# Evaluate model performance
def evaluate_forecast(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    return mae, rmse

# Assuming the last 4 periods are our test set
test_periods = 4
evaluation_results = {}

for level in Y_df['unique_id'].unique():
    actual = Y_df[Y_df['unique_id'] == level]['y'].iloc[-test_periods:]
    forecast = forecasts[forecasts['unique_id'] == level]['y']
    mae, rmse = evaluate_forecast(actual, forecast)
    evaluation_results[level] = {'MAE': mae, 'RMSE': rmse}

print("\nEvaluation Results:")
print(pd.DataFrame(evaluation_results).T)
```

**Reconciliation Methods:**

1. **BottomUp**: Forecast at the most disaggregated level, then sum upward
2. **TopDown**: Forecast at the top level, then disaggregate downward
3. **MiddleOut**: Forecast at a middle level, then aggregate up and disaggregate down
4. **MinT**: Minimum Trace reconciliation (optimal method)

**When to Use Hierarchical Forecasting:**

- Retail: Store → Region → Country → Global
- Finance: Product → Division → Company
- Energy: Plant → Region → Grid
- Any context with natural aggregation hierarchies

---

## 18.6 Parallel Computing and Scalability

One of the key advantages of the Nixtla suite is its built-in support for parallel computing, allowing you to forecast thousands of time series efficiently.

### Parallel Processing in StatsForecast

```python
# In StatsForecast, set n_jobs to -1 to use all available CPUs
sf = StatsForecast(df=df, models=models, freq='D', n_jobs=-1)
```

**Benefits:**

- **Speed**: Process multiple time series simultaneously
- **Scalability**: Handle large-scale forecasting tasks
- **Efficiency**: Optimal use of computational resources

### Best Practices for Large-Scale Forecasting

1. **Batch Processing**: Process time series in batches to manage memory
2. **Model Selection**: Use faster models (like Naive or SeasonalNaive) for large numbers of series
3. **Parallel Jobs**: Set `n_jobs` based on your CPU cores and memory
4. **Data Format**: Use the unified `unique_id`, `ds`, `y` format for efficient processing

---

## 18.7 Choosing the Right Tool

### When to Use StatsForecast

- **Classical Models**: When you need ARIMA, Exponential Smoothing, or Theta methods
- **Speed**: When you need fast forecasts for many time series
- **Interpretability**: When you need interpretable statistical models
- **Baseline Models**: For establishing baseline forecasts

### When to Use MLForecast

- **Rich Features**: When you have external features or complex patterns
- **Non-linear Patterns**: When relationships are non-linear
- **Feature Engineering**: When you want automatic feature creation
- **Ensemble Models**: When combining multiple ML models

### When to Use NeuralForecast

- **Complex Patterns**: When you have complex, non-linear patterns
- **Large Datasets**: When you have sufficient historical data
- **State-of-the-Art**: When you need the best possible accuracy
- **Deep Learning**: When you want to leverage deep learning architectures

### When to Use HierarchicalForecast

- **Multiple Levels**: When you need forecasts at multiple aggregation levels
- **Coherence Required**: When forecasts must sum correctly across levels
- **Business Context**: When your data has natural hierarchical structure

---

## 18.8 Integration Example: Combining Multiple Approaches

In practice, you might combine multiple Nixtla libraries for a comprehensive forecasting solution:

```python
# 1. Use StatsForecast for baseline forecasts
sf = StatsForecast(models=[AutoARIMA()], freq='D', n_jobs=-1)
baseline_forecasts = sf.forecast(df=df, h=horizon)

# 2. Use MLForecast for ML-based forecasts
ml_model = MLForecast(models=[LGBMRegressor()], freq='D', lags=[1, 7, 14])
ml_model.fit(df)
ml_forecasts = ml_model.predict(horizon)

# 3. Ensemble the forecasts
ensemble_forecast = 0.5 * baseline_forecasts['AutoARIMA'] + \
                     0.5 * ml_forecasts['LGBMRegressor']
```

---

## 18.9 Summary

The Nixtla suite provides a comprehensive, unified ecosystem for time series forecasting:

- **StatsForecast**: Fast statistical models for classical forecasting
- **MLForecast**: Machine learning with automatic feature engineering
- **NeuralForecast**: Deep learning architectures for complex patterns
- **HierarchicalForecast**: Coherent forecasting across aggregation levels

**Key Advantages:**

1. **Unified Interface**: Same data format and API across all libraries
2. **Production Ready**: Optimized for speed and scalability
3. **Comprehensive**: Covers all major forecasting approaches
4. **Parallel Computing**: Built-in support for multi-core processing

**Next Steps:**

- Experiment with different models in each library
- Combine forecasts from multiple approaches
- Apply hierarchical forecasting to your business context
- Scale to large numbers of time series using parallel processing

---

## Exercises

1. **StatsForecast Practice**: Use StatsForecast to forecast a time series of your choice. Compare AutoARIMA with AutoETS and Theta methods.

2. **MLForecast Feature Engineering**: Create an MLForecast model with custom lag features and date features. Experiment with different ML models (LightGBM, XGBoost).

3. **NeuralForecast Comparison**: Compare MLP, NHITS, and N-BEATS models on the same dataset. Which performs best?

4. **Hierarchical Forecasting**: Create a hierarchical structure for a dataset you're familiar with (e.g., sales by product category, subcategory, and item). Implement hierarchical forecasting and evaluate coherence.

5. **Ensemble Approach**: Combine forecasts from StatsForecast, MLForecast, and NeuralForecast. Experiment with different weighting schemes.

---

## References and Further Reading

- Nixtla Documentation: https://nixtla.github.io/
- StatsForecast: https://github.com/Nixtla/statsforecast
- MLForecast: https://github.com/Nixtla/mlforecast
- NeuralForecast: https://github.com/Nixtla/neuralforecast
- HierarchicalForecast: https://github.com/Nixtla/hierarchicalforecast

