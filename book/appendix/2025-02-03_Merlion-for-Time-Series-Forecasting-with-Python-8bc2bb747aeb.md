<div>

# Merlion for Time Series Forecasting with Python {#merlion-for-time-series-forecasting-with-python .p-name}

</div>

::: {.section .p-summary field="subtitle"}
Exploring a new approach/library
:::

::::::::::: {.section .e-content field="body"}
:::::: {#13b6 .section .section .section--body .section--first}
::: section-divider

------------------------------------------------------------------------
:::

:::: section-content
::: {.section-inner .sectionLayout--insetColumn}
### Merlion for Time Series Forecasting with Python {#9572 .graf .graf--h3 .graf--leading .graf--title name="9572"}

#### Exploring a new approach/library {#a8bf .graf .graf--h4 .graf-after--h3 .graf--subtitle name="a8bf"}

Merlion is an open-source Python library designed for time series
forecasting and anomaly detection. Developed by Salesforce, it
simplifies the end-to-end workflow of time series analysis by
integrating data preprocessing, model training, evaluation, and
visualization into a single framework.

Merlion supports statistical models, machine learning approaches, and
deep learning models. Merlion requires time series data in Pandas
DataFrame format with timestamps.

### Let's build an example. {#26f1 .graf .graf--h3 .graf-after--p name="26f1"}

I'm using data from Ercot on energy demand in Texas.

``` {#589f .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="bash"}
import pandas as pd
from merlion.utils.time_series import TimeSeries

# Load dataset
url = "https://raw.githubusercontent.com/kylejones200/time_series/main/ercot_load_data.csv"
df = pd.read_csv(url)
# Convert time column to datetime format
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
# Resample to hourly frequency
df = df.resample('H').mean()
df['values'] = df['values'].interpolate()
# Convert to Merlion TimeSeries format
ts = TimeSeries.from_pd(df)
print(ts)
```

Merlion provides several forecasting models, including ARIMA and
Exponential Smoothing.

``` {#f9ab .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from merlion.models.forecast.arima import Arima, ArimaConfig

# Initialize Exponential Smoothing with seasonal components
# Note: Using statsmodels as Merlion doesn't have built-in ETS
ets_model = ExponentialSmoothing(
    df['values'],
    seasonal='add',
    seasonal_periods=24  # Hourly data with daily seasonality
)
# Initialize ARIMA (manually tuned order)
arima_model = Arima(ArimaConfig(order=(2, 1, 2), target_seq_index=0))
```

Normally you would want ARIMA to be auto tuned. I had trouble getting
Merlion to do that --- a task easily done with pmdarima.

``` {#79ab .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="makefile"}
# Split data into training and test sets
train_ratio = 0.8  # 80% training, 20% testing
split_idx = int(len(df) * train_ratio)
train_data = TimeSeries.from_pd(df.iloc[:split_idx])
test_data = TimeSeries.from_pd(df.iloc[split_idx:])

# Train the models
ets_fitted = ets_model.fit()
arima_model.train(train_data)
```

Merlion has several features for measuring forecast accuracy. I'm usin
sMAPE (symetric mean absolute percentage error).

``` {#4331 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
from merlion.evaluate.forecast import ForecastMetric

# Generate forecasts
ets_forecast = ets_fitted.forecast(steps=len(test_data))
ets_forecast_ts = TimeSeries.from_pd(pd.DataFrame(ets_forecast, index=test_data.time_stamps))
arima_forecast, _ = arima_model.forecast(test_data.time_stamps)
# Compute sMAPE
ets_smape = ForecastMetric.sMAPE.value(test_data, ets_forecast_ts)
arima_smape = ForecastMetric.sMAPE.value(test_data, arima_forecast)
print(f"Exponential Smoothing sMAPE: {ets_smape:.2f}")
print(f"ARIMA sMAPE: {arima_smape:.2f}")
```

Merlion has built in visuzlation tools but I'm using matplotlob instead
so I can have more flexibility.

``` {#b8ac .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="less"}
import matplotlib.pyplot as plt

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(test_data.to_pd(), label="Actual")
plt.plot(ets_forecast_ts.to_pd(), label="Exponential Smoothing Forecast", linestyle="--")
plt.plot(arima_forecast.to_pd(), label="ARIMA Forecast", linestyle="--")
plt.legend()
plt.title("Exponential Smoothing vs ARIMA Forecasting")
plt.show()
```

### Anomaly Detection with Merlion {#cb7c .graf .graf--h3 .graf-after--pre name="cb7c"}

Merlion supports **both supervised and unsupervised** anomaly detection
models.

``` {#52ae .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="makefile"}
from merlion.models.anomaly.isolation_forest import IsolationForest, IsolationForestConfig

# Initialize an Isolation Forest model with the correct config
config = IsolationForestConfig()
anomaly_model = IsolationForest(config)

# Train the model on the dataset
anomaly_model.train(train_data)

# Generate anomaly scores
anomalies = anomaly_model.get_anomaly_label(test_data)
scores = anomaly_model.get_anomaly_score(test_data)

# Plot anomaly scores
plt.figure(figsize=(10, 6))
plt.plot(test_data.to_pd(), label="Original Data")
plt.plot(scores.to_pd(), label="Anomaly Scores", color="red", linestyle="--")
plt.legend()
plt.title("Anomaly Detection with Merlion")
plt.show()
```

<figure id="96d8" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*SBGyc_Nre3F_W4gJwn_5OA.png"
class="graf-image" data-image-id="1*SBGyc_Nre3F_W4gJwn_5OA.png"
data-width="1000" data-height="600" />
</figure>

The scores are basically zero because this is a well structured dataset.

### Model Comparison {#7a79 .graf .graf--h3 .graf-after--p name="7a79"}

Merlion simplifies benchmarking multiple models.

``` {#79de .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from merlion.models.forecast.arima import Arima, ArimaConfig

# Instantiate models correctly
ets_model = ExponentialSmoothing(
    train_data.to_pd()['values'],
    seasonal='add',
    seasonal_periods=24
)
ets_fitted = ets_model.fit()
arima_model = Arima(ArimaConfig(order=(2, 1, 2), target_seq_index=0))

# Train the models
arima_model.train(train_data)

# Compare models on performance metrics
results = []
# Exponential Smoothing forecast
ets_forecast = ets_fitted.forecast(steps=len(test_data))
ets_forecast_ts = TimeSeries.from_pd(pd.DataFrame(ets_forecast, index=test_data.time_stamps))
ets_smape = ForecastMetric.sMAPE.value(test_data, ets_forecast_ts)
results.append({"Model": "Exponential Smoothing", "sMAPE": ets_smape})

# ARIMA forecast
arima_forecast, _ = arima_model.forecast(test_data.time_stamps)
arima_smape = ForecastMetric.sMAPE.value(test_data, arima_forecast)
results.append({"Model": "Merlion ARIMA", "sMAPE": arima_smape})

# Convert results to DataFrame
comparison_df = pd.DataFrame(results)
print(comparison_df)
```

``` {#9aa0 .graf .graf--pre .graf-after--pre .graf--preV2 code-block-mode="2" spellcheck="false" code-block-lang="plaintext"}
        Model                    sMAPE
0       Merlion ARIMA           6.038623
1       Exponential Smoothing  25.010984
```

### Full implementation {#7863 .graf .graf--h3 .graf-after--pre name="7863"}

``` {#9e57 .graf .graf--pre .graf-after--h3 .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from merlion.evaluate.forecast import ForecastMetric
from merlion.models.factory import ModelFactory
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from merlion.models.forecast.arima import ArimaConfig
from merlion.models.defaults import DefaultForecasterConfig
from merlion.transform.resample import TemporalResample
from merlion.transform.sequence import TransformSequence
from merlion.utils import TimeSeries
from merlion.transform.normalize import MeanVarNormalize
from merlion.transform.moving_average import MovingAverage, DifferenceTransform
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from scipy.stats import norm

# Load ERCOT dataset with hourly resampling and outlier removal
url = "https://raw.githubusercontent.com/kylejones200/time_series/main/ercot_load_data.csv"
df = pd.read_csv(url)

# Convert time column to datetime format
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Resample to hourly frequency
df = df.resample('H').mean()
df['values'] = df['values'].interpolate()

# Remove outliers (values beyond 3 standard deviations)
df["z_score"] = (df["values"] - df["values"].mean()) / df["values"].std()
df = df[df["z_score"].abs() < 3]
df.drop(columns=["z_score"], inplace=True)

# Convert to Merlion TimeSeries format
ts = TimeSeries.from_pd(df)

# Optimize Data Splitting using TimeSeriesSplit (Cross-validation)
tscv = TimeSeriesSplit(n_splits=5)
train_idx, test_idx = list(tscv.split(df))[-1]  # Use the last split
train_data = TimeSeries.from_pd(df.iloc[train_idx])
test_data = TimeSeries.from_pd(df.iloc[test_idx])

# Function to create a model using ModelFactory
from merlion.models.factory import ModelFactory

def get_model(model_type="ets", transform=None):
    if model_type == "ets":
        # Exponential Smoothing using statsmodels
        data = train_data.to_pd()['values'] if 'train_data' in globals() else df['values']
        if transform:
            # Apply transform if provided
            data = transform.transform(data)
        return ExponentialSmoothing(data, seasonal='add', seasonal_periods=24)
    elif model_type == "arima":
        return ModelFactory.create("merlion.models.forecast.arima:Arima", 
                                   order=(2, 1, 2), target_seq_index=0)
    elif model_type == "default":
        return ModelFactory.create("merlion.models.defaults:DefaultForecaster")
    else:
        raise ValueError(f"Invalid model type: {model_type}")


# Function to evaluate and visualize forecasts
def eval_model(model, train_data, test_data, title):
    forecast_horizon = min(len(test_data), 168)  # Forecast up to 7 days (168 hours)
    t = test_data.time_stamps[:forecast_horizon]

    model.train(train_data)
    yhat_test, test_err = model.forecast(t)

    smape_value = ForecastMetric.sMAPE.value(test_data, yhat_test)

    # Confidence Intervals
    if hasattr(model, "forecast") and test_err is not None:
        ci_multiplier = 1.96  # 95% confidence
        lb = (yhat_test.to_pd() - ci_multiplier * test_err.to_pd().abs()).values.flatten()
        ub = (yhat_test.to_pd() + ci_multiplier * test_err.to_pd().abs()).values.flatten()

        # Ensure confidence intervals have the same length as timestamps
        min_length = min(len(t), len(lb), len(ub))
        t = t[:min_length]
        lb = lb[:min_length]
        ub = ub[:min_length]

    print(f"{title} - sMAPE: {smape_value:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(test_data.to_pd(), label="Actual")
    plt.plot(yhat_test.to_pd(), label="Forecast", linestyle="--")

    if hasattr(model, "forecast") and test_err is not None:
        plt.fill_between(t, lb, ub, color="gray", alpha=0.3, label="Confidence Interval")

    plt.legend()
    plt.title(f"{title} - sMAPE: {smape_value:.2f}")
    plt.show()

    return yhat_test

# Run Exponential Smoothing Model Without Transformations
print("No transform...")
ets_base = get_model("ets")
ets_base_fitted = ets_base.fit()
ets_base_forecast = ets_base_fitted.forecast(steps=len(test_data))
ets_base_ts = TimeSeries.from_pd(pd.DataFrame(ets_base_forecast, index=test_data.time_stamps))
base_smape = ForecastMetric.sMAPE.value(test_data, ets_base_ts)
base = ets_base_ts

# Note: Exponential Smoothing doesn't support Merlion transforms directly
# We'll use the base model for comparison
print("Exponential Smoothing (No Transform) - sMAPE:", base_smape)

# Run Merlion ARIMA Model
print("\n=== ARIMA Model ===")
arima_results = eval_model(get_model("arima"), train_data, test_data, title="Merlion ARIMA")

# Run Default Forecaster (Baseline Model)
print("\n=== Default Forecaster (Baseline Model) ===")
default_results = eval_model(get_model("default"), train_data, test_data, title="Default Forecaster")

# Create a table of sMAPE values
smape_values = {
    "Exponential Smoothing (No Transform)": ForecastMetric.sMAPE.value(test_data, base),
    "Merlion ARIMA": ForecastMetric.sMAPE.value(test_data, arima_results),
    "Default Forecaster": ForecastMetric.sMAPE.value(test_data, default_results),
}

# Convert to a DataFrame
smape_table = pd.DataFrame(list(smape_values.items()), columns=["Model", "sMAPE"]).sort_values(by="sMAPE")
print(smape_table)
```

<figure id="f681" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*wfbNzaGQo-fHHPSwlENrvQ.png"
class="graf-image" data-image-id="1*wfbNzaGQo-fHHPSwlENrvQ.png"
data-width="1000" data-height="600" />
</figure>

<figure id="21cb" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*_sr6eWR-iYNPV7-maalMug.png"
class="graf-image" data-image-id="1*_sr6eWR-iYNPV7-maalMug.png"
data-width="1000" data-height="600" />
</figure>

<figure id="1433" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*Vrun_oPD_w3IVBHnH_pH2Q.png"
class="graf-image" data-image-id="1*Vrun_oPD_w3IVBHnH_pH2Q.png"
data-width="1000" data-height="600" />
</figure>

<figure id="de50" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*RUsG144lAmg_C8uCCEX8og.png"
class="graf-image" data-image-id="1*RUsG144lAmg_C8uCCEX8og.png"
data-width="1000" data-height="600" />
</figure>

<figure id="6521" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*t1vcfqgU9XpYrHutL-MOyw.png"
class="graf-image" data-image-id="1*t1vcfqgU9XpYrHutL-MOyw.png"
data-width="1000" data-height="600" />
</figure>

<figure id="c12c" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*z27cFCuLuC6Mu0YEXNMKYg.png"
class="graf-image" data-image-id="1*z27cFCuLuC6Mu0YEXNMKYg.png"
data-width="1000" data-height="600" data-is-featured="true" />
</figure>

Overall, the default forecaster is the best and captures the
fluctuations in the data well. ARIMA has a low sMAPE but it clearly
doesn't fit the data well.

``` {#6c66 .graf .graf--pre .graf-after--p .graf--trailing .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="scss"}
Model                                    sMAPE
2                  Default Forecaster   4.698401
1                       Merlion ARIMA   6.038623
0   Exponential Smoothing (No Transform)  25.010984
```
:::
::::
::::::

:::::: {#c920 .section .section .section--body .section--last}
::: section-divider

------------------------------------------------------------------------
:::

:::: section-content
::: {.section-inner .sectionLayout--insetColumn}
Merlion simplifies time series forecasting and anomaly detection in
Python with built-in evaluation and visualization tools. For the ERCOT
dataset, the Default Forecaster and Merlion ARIMA deliver the best
results. Exponential Smoothing provides a baseline comparison and
handles seasonality effectively.

I spent a long time fiddling with Merlion. I wanted it to be amazing but
I don't love it and don't plan to use it for more projects.
:::
::::
::::::
:::::::::::

By [Kyle Jones](https://medium.com/@kylejones_47003){.p-author .h-card}
on [February 3, 2025](https://medium.com/p/8bc2bb747aeb).

[Canonical
link](https://medium.com/@kylejones_47003/merlion-for-time-series-forecasting-with-python-8bc2bb747aeb){.p-canonical}

Exported from [Medium](https://medium.com) on February 9, 2025.
