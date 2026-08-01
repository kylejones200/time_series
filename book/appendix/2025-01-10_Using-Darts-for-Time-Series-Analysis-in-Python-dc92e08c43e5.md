<div>

# Using Darts for Time Series Analysis in Python {#using-darts-for-time-series-analysis-in-python .p-name}

</div>

::: {.section .p-summary field="subtitle"}
Darts simplifies time series analysis by providing a unified interface
for multiple forecasting methods, from traditional statistical...
:::

:::::::: {.section .e-content field="body"}
::::::: {#a6fd .section .section .section--body .section--first .section--last}
::: section-divider

------------------------------------------------------------------------
:::

::::: section-content
:::: {.section-inner .sectionLayout--insetColumn}
### Using Darts for Time Series Analysis in Python {#7881 .graf .graf--h3 .graf--leading .graf--title name="7881"}

#### Darts simplifies time series analysis by providing a unified interface for multiple forecasting methods, from traditional statistical approaches to advanced machine learning models. {#a123 .graf .graf--h4 .graf-after--h3 .graf--subtitle name="a123"}

Darts is a time series and forecasting library that streamlines
building, evaluating, and deploying time series models. It is basically
a wrapper for a lot of other models from ARIMA to LSTM++.

Darts provides a unified API for a variety of time series models so we
can switch between models and compare performance. It has built in
preprocessing for resampling, scaling, and handling missing data. And it
comes with evaluation tools for standard metrics like MAE, MAPE, RMSE,
and cross-validation.

Let's try it out.

Install Darts with: `pip install darts`{.markup--code .markup--p-code}

#### Basic Workflow {#4027 .graf .graf--h4 .graf-after--p name="4027"}

The typical workflow in Darts involves creating a
`TimeSeries`{.markup--code .markup--p-code} object, choosing and
training a model, making predictions, and evaluating results.

Darts requires data to be in a `TimeSeries`{.markup--code
.markup--p-code} object, which wraps around Pandas DataFrames.

In this project I use data from FRED (Federal Reserve Bank of St. Louis)
on 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant
Maturity (T10Y2Y). The T10Y2Y is a financial indicator that measures the
difference between the yields of 10-year and 2-year U.S. Treasury
securities (aka the "yield spread" or the "term spread") in basis
points. Positive spread means that 10-year bonds have higher yields than
2-year bonds, which is the normal situation. Negative speard (aka yield
curve inversion) is a sign of potential economic slowdown or recession.

Economists and investors use this as an indicator of market expectations
about future economic conditions and monetary policy.

FRED provides this data by API. You can request an API key for free and
you need to update the code with that.

<figure id="5508" class="graf graf--figure graf--iframe graf-after--p">

</figure>

Darts supports traditional methods like **Exponential Smoothing** and
**ARIMA** for quick, interpretable forecasts.

#### Exponential Smoothing {#41ea .graf .graf--h4 .graf-after--p name="41ea"}

<figure id="154a" class="graf graf--figure graf--iframe graf-after--h4">

</figure>

<figure id="6587" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*NHb4A6H3s_dwmjq2v3seoA.png"
class="graf-image" data-image-id="1*NHb4A6H3s_dwmjq2v3seoA.png"
data-width="989" data-height="590" data-is-featured="true" />
</figure>

I included way too much history in this version. The forecast is just 10
steps so it is a little blue dot at the end that you can barely see.

#### ARIMA {#df99 .graf .graf--h4 .graf-after--p name="df99"}

The autoARIMA implementations in other tools like statsmodels are easier
to use than Darts. But let's still look at it.

<figure id="786a" class="graf graf--figure graf--iframe graf-after--p">

</figure>

<figure id="9912" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*OCLYp4Q22WjyY3ECFQIXtQ.png"
class="graf-image" data-image-id="1*OCLYp4Q22WjyY3ECFQIXtQ.png"
data-width="1189" data-height="590" />
</figure>

In this viz, I zoomed so it is much easier to see the
prediction --- which is basically a straight line for the next 30 days.

### Machine Learning Models in Darts {#9f0d .graf .graf--h3 .graf-after--p name="9f0d"}

But that is just the start. Darts can also do more advanced forecasting
with machine learning models like **Random Forests** and **LightGBM**.

I have another article that goes deeper into N-BEATS.

::: {#a6c8 .graf .graf--mixtapeEmbed .graf-after--p}
[**N-BEATS for Time Series Forecasting in Python**\
*N-BEATS (Neural Basis Expansion Analysis for Time Series) is a deep
learning model specifically designed for
time...*medium.com](https://medium.com/@kylejones_47003/n-beats-for-time-series-forecasting-in-python-b4a61858fe49 "https://medium.com/@kylejones_47003/n-beats-for-time-series-forecasting-in-python-b4a61858fe49"){.markup--anchor
.markup--mixtapeEmbed-anchor
data-href="https://medium.com/@kylejones_47003/n-beats-for-time-series-forecasting-in-python-b4a61858fe49"}[](https://medium.com/@kylejones_47003/n-beats-for-time-series-forecasting-in-python-b4a61858fe49){.js-mixtapeImage
.mixtapeImage .u-ignoreBlock media-id="38114c005e3f9af57969f9c93b1402d2"
thumbnail-img-id="1*blaY9wwghtAX_aGQ0NO-Pg.png"
style="background-image: url(https://cdn-images-1.medium.com/fit/c/160/160/1*blaY9wwghtAX_aGQ0NO-Pg.png);"}
:::

<figure id="185f"
class="graf graf--figure graf--iframe graf-after--mixtapeEmbed">

</figure>

### Evaluating Models {#b1ed .graf .graf--h3 .graf-after--figure name="b1ed"}

Darts makes it easy to evaluate model performance using metrics like
MAE, RMSE, and MAPE.

<figure id="0aa4" class="graf graf--figure graf--iframe graf-after--p">

</figure>

#### Real world example: Energy Load Data {#eb29 .graf .graf--h4 .graf-after--figure name="eb29"}

I pulled data for energy load in ERCOT for every 15 mins from Dec 24,
2024 to January 11, 2025. Then I repeated these steps using the new
dataset.

``` {#a59d .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.metrics import mape


# Load the ERCOT data
df = pd.read_csv("ercot_load_data.csv")
df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' is in datetime format
df['values'] = pd.to_numeric(df['values'], errors='coerce')  # Convert 'values' to numeric
df = df.sort_values('date')  # Sort by date

# Drop rows with missing or NaN values
df = df.dropna()

# Resample the data to hourly frequency
df = df.set_index('date').resample('h').mean().reset_index()  # Resample and take the mean for each hour

# Define hold-out period
hold_out_hours = 24  # 24 hours = 1 day
train = df.iloc[:-hold_out_hours]
hold_out = df.iloc[-hold_out_hours:]

# Create TimeSeries for training and hold-out data
series_train = TimeSeries.from_dataframe(train, 'date', 'values', freq="h", fill_missing_dates=True)
series_hold_out = TimeSeries.from_dataframe(hold_out, 'date', 'values', freq="h")

# Fit the Exponential Smoothing model on training data
model = ExponentialSmoothing()
model.fit(series_train)

# Forecast the hold-out period
forecast = model.predict(len(series_hold_out))

# Calculate MAPE
mape = mape(series_hold_out, forecast)

# Plot the results
plt.figure(figsize=(12, 6))

# Plot training data
series_train.plot(label="Training Data", color='blue')

# Plot hold-out data
series_hold_out.plot(label="Hold-Out Data (Actual)", color='green')

# Plot forecasted data
forecast.plot(label="Forecast", color='red')

plt.title(f"ERCOT Hourly Load Forecast with Hold-Out Data \n MAPE: {mape:.2f}%")
plt.xlabel("Date")
plt.ylabel("Load Values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ERCOT_Hourly_HoldOut_Forecast.png")
plt.show()
```

<figure id="a274" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*sv7wX0j7CAqlGbrd6SC2og.png"
class="graf-image" data-image-id="1*sv7wX0j7CAqlGbrd6SC2og.png"
data-width="1200" data-height="600" />
</figure>

The exponential smoothing works better than ARIMA on this dataset.

``` {#8f8a .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import ARIMA


# Define hold-out period
hold_out_hours = 24  # Example: 24 hours = 1 day
train = df.iloc[:-hold_out_hours]
hold_out = df.iloc[-hold_out_hours:]

# Create TimeSeries for training and hold-out data
series_train = TimeSeries.from_dataframe(train, 'date', 'values', freq="h", fill_missing_dates=True)
series_hold_out = TimeSeries.from_dataframe(hold_out, 'date', 'values', freq="h")

# Fit the ARIMA model
model = ARIMA(p=1, d=1, q=1)  # You can adjust p, d, q parameters
model.fit(series_train)

# Forecast the hold-out period
forecast = model.predict(len(series_hold_out))
# Calculate MAPE
mape_result = mape(series_hold_out, forecast)

# Plot the results
plt.figure(figsize=(12, 6))


series_train.plot(label="Training Data", color='blue')
series_hold_out.plot(label="Hold-Out Data (Actual)", color='green')
forecast.plot(label="Forecast", color='red')

plt.title(f"ERCOT Hourly Load Forecast with ARIMA and Hold-Out Period \n MAPE: {mape_result:.2f}%")
plt.xlabel("Date")
plt.ylabel("Load Values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ARIMA_Hourly_HoldOut_Forecast.png")
plt.show()
```

<figure id="57ff" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*zLA4WSSStag-79TvAYtmWg.png"
class="graf-image" data-image-id="1*zLA4WSSStag-79TvAYtmWg.png"
data-width="1200" data-height="600" />
</figure>

### But wait, there's more! {#2638 .graf .graf--h3 .graf-after--figure name="2638"}

**Darts has tools for Backtesting to** Evaluate how well a model
performs over historical data. It can do **Transformations like** Scale,
log-transform, or normalize data before modeling. And it can do
ensembling to combine multiple models for better forecasts.

<figure id="4660" class="graf graf--figure graf--iframe graf-after--p">

</figure>

### Deployment with Darts {#ef60 .graf .graf--h3 .graf-after--figure name="ef60"}

You can pickle Darts models so they are easy to containerize for
inference.

<figure id="778c" class="graf graf--figure graf--iframe graf-after--p">

</figure>

I got a little carried away and made this version that uses N-BEATS and
Fast Fourier Transforms. The FFTs do a terrible job on this dataset but
it was still fun to make.

<figure id="e464" class="graf graf--figure graf--iframe graf-after--p">

</figure>

### Key Takeaways {#a7f3 .graf .graf--h3 .graf-after--figure name="a7f3"}

Darts is a pretty good time series and forecasting library with an
intuitive API and a lot of options. It covers most of the things you
need for day to day TS work.
::::
:::::
:::::::
::::::::

By [Kyle Jones](https://medium.com/@kylejones_47003){.p-author .h-card}
on [January 10, 2025](https://medium.com/p/dc92e08c43e5).

[Canonical
link](https://medium.com/@kylejones_47003/using-darts-for-time-series-analysis-in-python-dc92e08c43e5){.p-canonical}

Exported from [Medium](https://medium.com) on February 9, 2025.
