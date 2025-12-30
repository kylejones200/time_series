<div>

# AutoGluon for Time Series Forecasting in Python {#autogluon-for-time-series-forecasting-in-python .p-name}

</div>

::: {.section .p-summary field="subtitle"}
AutoGluon is auto ML Python library that streamlines feature
engineering, model selection, hyperparameter tuning, and evaluation.
AutoGluon...
:::

::::::: {.section .e-content field="body"}
:::::: {#901c .section .section .section--body .section--first .section--last}
::: section-divider

------------------------------------------------------------------------
:::

:::: section-content
::: {.section-inner .sectionLayout--insetColumn}
### AutoGluon for Time Series Forecasting in Python {#dc96 .graf .graf--h3 .graf--leading .graf--title name="dc96"}

**AutoGluon** is auto ML Python library that streamlines feature
engineering, model selection, hyperparameter tuning, and evaluation.
AutoGluon has a `TimeSeriesPredictor`{.markup--code .markup--p-code}
specifically made for time series forecasting.

AutoGluon's time series module automates a lot of things --- including
model selection, tuning, and feature engineering. It can create
ensembles by combining multiple models. The API is easy to use
(especially if you have built models with AWS SageMaker before).

### AutoGluon Workflow {#3b14 .graf .graf--h3 .graf-after--p name="3b14"}

The workflow with AutoGluon for time series involves preparing the
dataset, initializing the `TimeSeriesPredictor`{.markup--code
.markup--p-code} , training models automatically, and generating
forecasts.

Here is an example with synthetic data. We start by declaring:

-   [**time_column**: The timestamp for each observation.]{#a4c9}
-   [**target_column**: The value to be predicted.]{#5175}
-   [**item_id_column (optional)**: Identifies different time series in
    a dataset (for multivariate forecasting).]{#d598}

``` {#6185 .graf .graf--pre .graf-after--li .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries import TimeSeriesPredictor

# Create more complex sample data with seasonal patterns
np.random.seed(42)  # for reproducibility

# Generate timestamps for 3 years of monthly data
timestamps = pd.date_range("2020-01-01", periods=36, freq="ME")

# Function to create seasonal pattern
def create_seasonal_data(base_value, trend, seasonal_amplitude, noise_level):
    time = np.arange(len(timestamps))
    # Trend component
    trend_component = base_value + trend * time
    # Seasonal component (yearly seasonality)
    seasonal_component = seasonal_amplitude * np.sin(2 * np.pi * time / 12)
    # Random noise
    noise = np.random.normal(0, noise_level, len(time))
    return trend_component + seasonal_component + noise

# Create different patterns for different items
data = {
    "item_id": ["A"] * 36 + ["B"] * 36 + ["C"] * 36,
    "timestamp": timestamps.tolist() * 3,
    "sales": np.concatenate([
        # Item A: Strong seasonality, moderate trend, low noise
        create_seasonal_data(base_value=1000, trend=15, seasonal_amplitude=200, noise_level=30),
        # Item B: Moderate seasonality, high trend, moderate noise
        create_seasonal_data(base_value=500, trend=25, seasonal_amplitude=100, noise_level=50),
        # Item C: Weak seasonality, negative trend, high noise
        create_seasonal_data(base_value=1500, trend=-10, seasonal_amplitude=50, noise_level=100)
    ])
}

# Create DataFrame and set multi-index
df = pd.DataFrame(data)
df = df.set_index(['item_id', 'timestamp'])

# Convert to TimeSeriesDataFrame
train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column='item_id',
    timestamp_column='timestamp'
)

```

Output:

``` {#c0be .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="css"}
item_id   timestamp  sales
0       A 2020-01-31    100
1       A 2020-02-29    110
2       A 2020-03-31    120
3       A 2020-04-30    130
4       A 2020-05-31    140
```

#### Initializing and Training the Predictor {#7781 .graf .graf--h4 .graf-after--pre name="7781"}

To initialize the TimeSeriesPredictor we need to specify the target,
time, and (optionally) the item ID columns.

We don't have to do anything for the Predictor --- it will automatically
select a model for us.

``` {#cd4e .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="bash"}
# Initialize predictor
predictor = TimeSeriesPredictor(
    prediction_length=6,      # Forecast horizon
    eval_metric='MASE',      # Evaluation metric
    target='sales',          # Target variable
)

# Train the predictor
predictor.fit(train_data=train_data)
```

The predictor will automatically do feature engineering (e.g., lagged
values, seasonality features) and train multiple models, including
statistical models (ARIMA), machine learning models (LightGBM,
CatBoost), and deep learning models (N-BEATS).

Generate forecasts for the next `prediction_length`{.markup--code
.markup--p-code} time steps.

``` {#7773 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="bash"}
# Generate forecasts
forecasts = predictor.predict(train_data)
print(forecasts.head())

# Plot forecasted vs actual values
predictor.plot(train_data)

# Evaluate model performance
performance = predictor.evaluate(train_data)
print(performance)
```

Output:

``` {#7bbf .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="css"}
item_id   timestamp      sales
0       A 2022-01-31  350.1234
1       A 2022-02-28  360.5678
2       A 2022-03-31  370.9876
3       B 2022-01-31  310.4321
4       B 2022-02-28  320.8765
```

AutoGluon includes built-in visualization tools.

``` {#f34b .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="bash"}
# Plot forecasted vs actual values
predictor.plot(train_data)
```

There are three different products in our our dataset. So we get a
prediction for each one.

<figure id="4d35" class="graf graf--figure graf-after--p">
<img
src="https://cdn-images-1.medium.com/max/800/1*A2kY19b_oZKG8hKLx8u8oQ.png"
class="graf-image" data-image-id="1*A2kY19b_oZKG8hKLx8u8oQ.png"
data-width="2011" data-height="753" data-is-featured="true" />
</figure>

AutoGluon provides metrics like RMSE, MAPE, and MAE to evaluate model
performance.

``` {#f9f3 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="bash"}
# Evaluate model performance
performance = predictor.evaluate(df)
print(performance)
```

Validation score from lowest (best) to highest (worst):

<figure id="5254" class="graf graf--figure graf-after--p">
<img
src="https://cdn-images-1.medium.com/max/800/1*lBqIfVCn45djYdMvLYh5dw.png"
class="graf-image" data-image-id="1*lBqIfVCn45djYdMvLYh5dw.png"
data-width="850" data-height="528" />
</figure>

It takes a long time to run Autogluon (this run took 24 mins). You can
see the models that take a long time in the chart above --- chronos (the
LLM model) was the biggest culprit requiring 16 and a half minutes on
its own.

### Deployment and Saving Models {#5291 .graf .graf--h3 .graf-after--p name="5291"}

I don't talk about model management in most of my articles but it is
important for deploying models and using them in the real world.
autogluon lets us save the trained model for reuse. we can use this
later for testing or we can put this into a contianer and use it for
inference as new data comes in.

``` {#73fa .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
predictor.save("timeseries_model")

"""
Load the Model
Load the saved model to make predictions on new data.
"""
from autogluon.timeseries import TimeSeriesPredictor

predictor = TimeSeriesPredictor.load("timeseries_model")
new_forecasts = predictor.predict(new_data)
```

### So what? {#2ddd .graf .graf--h3 .graf-after--pre name="2ddd"}

AutoGluon is a nice library. I like the automation but sometimes it
feels like it picks really random models that I wouldn't have picked
(maybe that is a good thing). I don't love how the ensemble works and I
would prefer to use other libraries for ensembles because it just take
so long to run. If I could only use one time series library, I would not
pick AutoGluon.
:::
::::
::::::
:::::::

By [Kyle Jones](https://medium.com/@kylejones_47003){.p-author .h-card}
on [January 20, 2025](https://medium.com/p/1db0f24faa6c).

[Canonical
link](https://medium.com/@kylejones_47003/autogluon-for-time-series-forecasting-in-python-1db0f24faa6c){.p-canonical}

Exported from [Medium](https://medium.com) on February 9, 2025.
