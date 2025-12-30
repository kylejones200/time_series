<div>

# PyCaret for Low-Code Time Series Forecasting in Python {#pycaret-for-low-code-time-series-forecasting-in-python .p-name}

</div>

::: {.section .p-summary field="subtitle"}
A beginner's guide to automated time series forecasting with PyCaret
:::

::::::: {.section .e-content field="body"}
:::::: {#7a97 .section .section .section--body .section--first .section--last}
::: section-divider

------------------------------------------------------------------------
:::

:::: section-content
::: {.section-inner .sectionLayout--insetColumn}
### PyCaret for Low-Code Time Series Forecasting in Python {#c13f .graf .graf--h3 .graf--leading .graf--title name="c13f"}

#### A beginner's guide to automated time series forecasting with PyCaret {#3d65 .graf .graf--h4 .graf-after--h3 .graf--subtitle name="3d65"}

**PyCaret** is a low-code machine learning library that simplifies the
process of building and deploying machine learning models. With the
release of its time series module, PyCaret makes time series forecasting
more accessible, offering robust tools for automated model selection,
tuning, and evaluation.

PyCaret is a single framework for automation, model management, and
deployment. It supports traditional statistical models like ARIMA and
modern machine learning algorithms such as LightGBM and XGBoost.

The end-to-end workflow capabilities streamline the entire modeling
process from initial data preprocessing through feature engineering and
final deployment. You run the code and it does a LOT of stuff.

It integrates with deployment tools like Flask and Docker which helps
with the transition from development to production.

### The PyCaret Time Series Workflow {#2cf4 .graf .graf--h3 .graf-after--p name="2cf4"}

Using PyCaret for time series involves five main steps:

1.  [Load and preprocess the dataset.]{#013b}
2.  [Initialize the time series setup.]{#2734}
3.  [Compare and train models.]{#9b86}
4.  [Evaluate model performance.]{#4f07}
5.  [Deploy or save the final model.]{#fdfb}

### Example: Forecasting Sales Data {#c61e .graf .graf--h3 .graf-after--li name="c61e"}

#### Load the Dataset --- Let's use a dataset of monthly sales data. {#f8ea .graf .graf--h4 .graf-after--h3 name="f8ea"}

``` {#b84d .graf .graf--pre .graf-after--h4 .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
import pandas as pd

# Create a sample time series dataset
data = pd.Series(
    [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10, 
    name="Sales"
)
data.index = pd.date_range(start="2010-01-01", periods=len(data), freq="M")

# Convert to DataFrame
df = data.to_frame()

print(df.head())
```

Initialize PyCaret for Time Series. The `setup`{.markup--code
.markup--p-code} function initializes the PyCaret pipeline. Key
arguments in `setup`{.markup--code .markup--p-code}:

-   [`data`{.markup--code .markup--li-code}: The time series
    data.]{#9c17}
-   [`target`{.markup--code .markup--li-code}: The target column to
    forecast.]{#5693}
-   [`seasonal_period`{.markup--code .markup--li-code}: Specify
    seasonality (e.g., 12 for monthly data).]{#957c}

``` {#15af .graf .graf--pre .graf-after--li .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="bash"}
# Initialize time series experiment
exp = TSForecastingExperiment()

# Setup the environment
exp.setup(
    data=df,
    target="Sales",
    session_id=123,
    seasonal_period=12
)

# Compare models
best_model = exp.compare_models()

# Tune the best model
tuned_model = exp.tune_model(best_model)

# Make predictions
future_forecast = exp.predict_model(tuned_model, fh=12)
print("\nForecast:")
print(future_forecast)

# Plot results
exp.plot_model(tuned_model, plot="forecast")

# Get metrics
metrics = exp.pull()
print("\nMetrics:")
print(metrics)
```

<figure id="9d02" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*Zr3DsNluRRVmNDuo4ARcDg.png"
class="graf-image" data-image-id="1*Zr3DsNluRRVmNDuo4ARcDg.png"
data-width="541" data-height="623" />
</figure>

<figure id="6fb4" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*TvAXg-CGexR242F8amJhQQ.png"
class="graf-image" data-image-id="1*TvAXg-CGexR242F8amJhQQ.png"
data-width="937" data-height="315" />
</figure>

<figure id="5f6c" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*FX99tvkqMAft9KrfPI8HIQ.png"
class="graf-image" data-image-id="1*FX99tvkqMAft9KrfPI8HIQ.png"
data-width="1122" data-height="525" />
</figure>

Compare and Train Models

The `compare_models`{.markup--code .markup--p-code} function evaluates
multiple models and identifies the best-performing one.

This step automates model selection by testing various algorithms (e.g.,
ARIMA, Exponential Smoothing (ETS), SARIMA) and ranks them based on performance metrics like
MAE or RMSE.

`tune_model`{.markup--code .markup--p-code}: optimize hyperparameters
for the selected model.

`predict_model`{.markup--code .markup--p-code}: forecast future values.

`plot_model`{.markup--code .markup--p-code}: visualize the forecasts

We can save the model and load the saved model.

``` {#aa2e .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="bash"}
# Save and load model
exp.save_model(tuned_model, "time_series_model")
loaded_model = exp.load_model("time_series_model")

# Multivariate analysis
df["Marketing_Spend"] = [50 + (i % 10) for i in range(len(df))]

# New experiment for multivariate analysis
exp_multi = TSForecastingExperiment()

exp_multi.setup(
    data=df,
    target="Sales",
    session_id=123
)

best_multivariate_model = exp_multi.compare_models()
future_forecast_multivariate = exp_multi.predict_model(best_multivariate_model, fh=12)
print("\nMultivariate forecast:")
print(future_forecast_multivariate)

exp_multi.plot_model(best_multivariate_model, plot="forecast")
```

<figure id="24f8" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*YTCUhXBCz-K8YfU6DZsrHg.png"
class="graf-image" data-image-id="1*YTCUhXBCz-K8YfU6DZsrHg.png"
data-width="1122" data-height="525" data-is-featured="true" />
</figure>

#### Multivariate Time Series {#2188 .graf .graf--h4 .graf-after--figure name="2188"}

PyCaret supports multivariate time series, allowing you to include
additional features like temperature or holidays.

### Key Benefits of PyCaret for Time Series {#e88b .graf .graf--h3 .graf-after--p name="e88b"}

PyCaret automates many complex tasks in the modeling pipeline. This
automation extends from initial data preparation through model
deployment, making sophisticated time series analysis more accessible
while maintaining flexibility for customization. The framework
particularly excels in handling financial time series data where
multiple models and feature engineering approaches need to be evaluated
rapidly.

The framework's automated approach significantly reduces development
time by handling model selection, hyperparameter tuning, and evaluation
systematically. PyCaret automatically tests multiple algorithms,
including ARIMA, Exponential Smoothing (ETS), and various machine learning models, comparing
their performance using relevant metrics for time series data. This
automation extends to feature engineering, where the system
automatically generates important temporal features like lags, rolling
statistics, and seasonal indicators, crucial for financial market
analysis.

Deployment capabilities in PyCaret streamline the transition from
development to production. The framework provides built-in functions to
save models, create REST APIs, and containerize solutions, making it
easier to integrate time series models into existing trading systems.
This integration includes features for model monitoring and updating,
essential for maintaining performance in dynamic financial markets. The
combination of automation, flexibility, and deployment features makes
PyCaret particularly valuable for rapid development and implementation
of time series classification systems.

### So what? {#4a05 .graf .graf--h3 .graf-after--p name="4a05"}

PyCaret converts tedious workflows into streamlined processes. You can
do a lot of work with very little code.

Beginners will like that PyCaret provides a structured approach to time
series analysis, automating critical decisions about model selection and
feature engineering while maintaining transparency in the process.

Experienced practitioners will like the efficiency they get by
automating routine tasks. The framework's ability to handle both
traditional statistical methods and modern machine learning approaches
makes it particularly valuable in financial applications where multiple
modeling strategies often need to be evaluated.
