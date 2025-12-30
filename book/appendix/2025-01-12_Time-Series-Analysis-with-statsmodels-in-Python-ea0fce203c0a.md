<div>

# Time Series Analysis with statsmodels in Python {#time-series-analysis-with-statsmodels-in-python .p-name}

</div>

::: {.section .p-summary field="subtitle"}
The statsmodels library combines traditional methods with modern Python
capabilities for business forecasting and analysis.
:::

::::::::::::::: {.section .e-content field="body"}
:::::: {#6e8f .section .section .section--body .section--first}
::: section-divider

------------------------------------------------------------------------
:::

:::: section-content
::: {.section-inner .sectionLayout--insetColumn}
### Time Series Analysis with `statsmodels`{.markup--code .markup--h3-code} in Python {#536d .graf .graf--h3 .graf--leading .graf--title name="536d"}

#### The statsmodels library combines traditional methods with modern Python capabilities for business forecasting and analysis. {#b09b .graf .graf--h4 .graf-after--h3 .graf--subtitle name="b09b"}

**`statsmodels`{.markup--code .markup--p-code}** provides a lot of tools
for statistical modeling, including time series. I've written about
other libraries so I thought I should include something about the most
popular library out there. If you are using ARIMA or SARIMA it is the
logical place to start.

`statsmodels`{.markup--code .markup--p-code} covers univariate and
multivariate time series modeling. It includes lots of statistical tests
to assess model assumptions and performance. It has "MS Excel" like
outputs of key metrics. And it works well with `pandas`{.markup--code
.markup--p-code} and `numpy`{.markup--code .markup--p-code}.

Let's check it out. We'll analyze a simulated time series dataset to
demonstrate some of the key features of `statsmodels.`{.markup--code
.markup--p-code}

Import required libraries:

``` {#70ab .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="2" spellcheck="false" code-block-lang="python"}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
```

#### Generate or Load Time Series Data {#de9f .graf .graf--h4 .graf-after--pre name="de9f"}

Simulate a time series with trend and seasonality:

``` {#a57f .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
"""
Generate or Load Time Series Data
Simulate a time series with trend and seasonality
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate Simulated Time Series Data
np.random.seed(42)
n = 200
time = pd.date_range(start="2023-01-01", periods=n, freq="D")
trend = np.linspace(10, 50, n)
seasonality = 10 * np.sin(np.linspace(0, 2 * np.pi, n))
noise = np.random.normal(0, 2, n)
data = trend + seasonality + noise

# Create a DataFrame; split the data
df = pd.DataFrame({"date": time, "value": data})
df.set_index("date", inplace=True)

hold_out_days = 30
train = df.iloc[:-hold_out_days]
hold_out = df.iloc[-hold_out_days:]

# Plot the Data
plt.figure(figsize=(10, 6))
plt.plot(df.index, df["value"], label="Full Dataset", color="Blue")
plt.plot(hold_out.index, hold_out["value"], label="Hold-Out (True Values)", color="Green")

plt.title("Simulated Time Series with Training and Hold-Out Sets")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.savefig("simulated_time_series.png")
plt.show()
```

<figure id="69c4" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*g7vZAtztFRhGKnqUcV827w.png"
class="graf-image" data-image-id="1*g7vZAtztFRhGKnqUcV827w.png"
data-width="1000" data-height="600" data-is-featured="true" />
</figure>

#### Time Series Decomposition {#e70f .graf .graf--h4 .graf-after--figure name="e70f"}

Use `seasonal_decompose`{.markup--code .markup--p-code} to split the
series into trend, seasonal, and residual components.

``` {#fc75 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
"""
Time Series Decomposition
Use seasonal_decompose to split the series into trend, seasonal, and residual components.
"""
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the time series
decomposition = seasonal_decompose(df["value"], model="additive", period=30)

# Plot the components
fig = decomposition.plot()
fig.set_size_inches(10, 8)  # Adjust the figure size
plt.suptitle("Time Series Decomposition", fontsize=16, y=0.95)  # Adjust title position
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Prevent overlap of title with subplots
plt.savefig("time_series_decomposition.png")
plt.show()
```

<figure id="da38"
class="graf graf--figure graf-after--pre graf--trailing">
<img
src="https://cdn-images-1.medium.com/max/800/1*faTpVXw4C48J8MZOJamFzw.png"
class="graf-image" data-image-id="1*faTpVXw4C48J8MZOJamFzw.png"
data-width="1000" data-height="800" />
</figure>
:::
::::
::::::

:::::: {#2a71 .section .section .section--body}
::: section-divider

------------------------------------------------------------------------
:::

:::: section-content
::: {.section-inner .sectionLayout--insetColumn}
#### Step 4: Check for Stationarity {#91d8 .graf .graf--h4 .graf--leading name="91d8"}

Use the Augmented Dickey-Fuller (ADF) test to assess stationarity.

``` {#e2cc .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
"""
Check for Stationarity
Use the Augmented Dickey-Fuller (ADF) test to assess stationarity.
"""

from statsmodels.tsa.stattools import adfuller
# Perform ADF test
result = adfuller(df["value"])
print(f"ADF Statistic: {result[0]:.4f}")
print(f"P-Value: {result[1]:.4f}")
if result[1] > 0.05:
    print("The time series is non-stationary.")
else:
    print("The time series is stationary.")
```

``` {#de13 .graf .graf--pre .graf-after--pre .graf--preV2 code-block-mode="2" spellcheck="false" code-block-lang="plaintext"}
ADF Statistic: -0.5022
P-Value: 0.8916
The time series is non-stationary.
```

#### Autocorrelation and Partial Autocorrelation {#56d4 .graf .graf--h4 .graf-after--pre name="56d4"}

Visualize the ACF and PACF to determine lag dependencies.

``` {#c26c .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
"""
Autocorrelation and Partial Autocorrelation
Visualize the ACF and PACF to determine lag dependencies.
"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_acf(df["value"], lags=30, ax=axes[0])
plot_pacf(df["value"], lags=30, ax=axes[1])
plt.suptitle("ACF and PACF Plots", fontsize=16)
plt.savefig("acf_pacf_plots.png")
plt.show()
```

<figure id="3e95" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*Liy9ky7EKhRE5CqJSlchkQ.png"
class="graf-image" data-image-id="1*Liy9ky7EKhRE5CqJSlchkQ.png"
data-width="1200" data-height="600" />
</figure>

#### Fit an ARIMA Model {#4d9e .graf .graf--h4 .graf-after--figure name="4d9e"}

Fit an ARIMA model to the data for forecasting.

``` {#d892 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
"""
Fit an ARIMA Model
Fit an ARIMA model to the data for forecasting.
"""

from statsmodels.tsa.arima.model import ARIMA
# Fit an ARIMA(2,1,2) model
model = ARIMA(df["value"], order=(2, 1, 2))
arima_result = model.fit()

print(arima_result.summary())
# Plot the residuals
arima_result.plot_diagnostics(figsize=(10, 6))
plt.savefig("arima_residuals_diagnostics.png")
plt.show()
```

<figure id="89e8"
class="graf graf--figure graf-after--pre graf--trailing">
<img
src="https://cdn-images-1.medium.com/max/800/1*4freJ8Ioo1v0lnRy9v9nrA.png"
class="graf-image" data-image-id="1*4freJ8Ioo1v0lnRy9v9nrA.png"
data-width="1000" data-height="600" />
</figure>
:::
::::
::::::

:::::: {#c6ad .section .section .section--body .section--last}
::: section-divider

------------------------------------------------------------------------
:::

:::: section-content
::: {.section-inner .sectionLayout--insetColumn}
#### Forecast Future Values {#225f .graf .graf--h4 .graf--leading name="225f"}

Use the fitted model to forecast holdout values.

``` {#fc00 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
"""
ARIMA
Forecast the 30 days that were held out
"""

# Fit ARIMA Model on Training Data
model = ARIMA(train["value"], order=(2, 1, 2), freq="D")  # Explicitly set freq="D"
arima_result = model.fit()

# Forecast Future Values for Hold-Out Period
forecast = arima_result.get_forecast(steps=hold_out_days)
forecast_index = hold_out.index  # Use the same index as the hold-out set
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Calculate MAPE on Hold-Out Set
mape = mean_absolute_percentage_error(hold_out["value"], forecast_mean)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.3%}")

# Plot the Results
plt.figure(figsize=(10, 6))
plt.plot(train.index, train["value"], label="Training Data", color="Blue")
plt.plot(hold_out.index, hold_out["value"], label="Hold-Out (True Values)", color="Green")
plt.plot(forecast_index, forecast_mean, label="Forecast", color="Red")
plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color="Red", alpha=0.2, label="Confidence Interval")
plt.title(f"ARIMA Forecast (MAPE: {mape:.3%})")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.savefig("arima_forecast_holdout.png")
plt.show()
```

<figure id="2bc8" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*Y1l-BSai4-qr4tshCoRWwA.png"
class="graf-image" data-image-id="1*Y1l-BSai4-qr4tshCoRWwA.png"
data-width="1000" data-height="600" />
</figure>

Our model predicts much slower growth than the series actually has.

#### Holt-Winters Exponential Smoothing {#fadd .graf .graf--h4 .graf-after--p name="fadd"}

``` {#dbd1 .graf .graf--pre .graf-after--h4 .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
"""
Holt-Winters Exponential Smoothing
"""

# Apply Holt-Winters Exponential Smoothing
hw_model = ExponentialSmoothing(
    train["value"], 
    seasonal="add", 
    seasonal_periods=30
).fit()

hw_forecast = hw_model.forecast(steps=hold_out_days)

# Calculate MAPE on Hold-Out Set
mape_hw = mean_absolute_percentage_error(hold_out["value"], hw_forecast)
print(f"Holt-Winters MAPE: {mape_hw:.3%}")

# Plot the Results
plt.figure(figsize=(10, 6))
plt.plot(train.index, train["value"], label="Training Data", color="Blue")
plt.plot(hold_out.index, hold_out["value"], label="Hold-Out (True Values)", color="Green")
plt.plot(hold_out.index, hw_forecast, label="Holt-Winters Forecast", color="Red")
plt.title(f"Holt-Winters Forecast \n MAPE: {mape_hw:.3%}")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.savefig("holt_winters_forecast.png")
plt.show()
```

<figure id="f90f" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*jl3f_eqX4hsu0TJQ57EHjQ.png"
class="graf-image" data-image-id="1*jl3f_eqX4hsu0TJQ57EHjQ.png"
data-width="1000" data-height="600" />
</figure>

Holt-Winters also doesn't predict the changes very accurately.

So I decided to test this with Tensorflow (Keras) using an LSTM model.

``` {#7fce .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# Prepare Data for LSTM
scaler = MinMaxScaler()
df["value"] = scaler.fit_transform(df["value"].values.reshape(-1, 1))

def create_lagged_features(data, lag):
    X, y = [], []
    for i in range(len(data) - lag):
        X.append(data[i:i+lag])
        y.append(data[i+lag])
    return np.array(X), np.array(y)

lag = 10  # Number of past observations to use for prediction
X, y = create_lagged_features(df["value"].values, lag)

X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into training and testing sets
train_size = int(0.85 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build, Fit, Predict and Evaluate the LSTM Model
model = tf.keras.Sequential([
    LSTM(50, activation='relu', input_shape=(lag, 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.1)

y_pred_lstm = model.predict(X_test)
y_pred_lstm_inverse = scaler.inverse_transform(y_pred_lstm)  # Inverse scaling for comparison
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

# Reconstruct training predictions for plotting
train_predictions = model.predict(X_train)
train_predictions_inverse = scaler.inverse_transform(train_predictions)

# Calculate MAPE for the test set
mape = mean_absolute_percentage_error(y_test_inverse, y_pred_lstm_inverse)
print(f"LSTM MAPE: {mape:.3%}")

# Plot the Results
plt.figure(figsize=(12, 8))
plt.plot(df.index, scaler.inverse_transform(df["value"].values.reshape(-1, 1)), label="Actual Data", color="Blue")
train_index = df.index[lag:train_size + lag]
plt.plot(train_index, train_predictions_inverse, label="Training Predictions", color="Orange")
test_index = df.index[train_size + lag:]
plt.plot(test_index, y_test_inverse, label="Hold-Out (True Values)", color="Green")
plt.plot(test_index, y_pred_lstm_inverse, label="Testing Predictions", color="Red")
plt.title(f'LSTM Forecast. MAPE: {mape:.3%}')
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.savefig("LSTM_forecast_with_holdout.png")
plt.show()
```

<figure id="5987" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*MdEAZ9e5VdwzZh7iUBYl1Q.png"
class="graf-image" data-image-id="1*MdEAZ9e5VdwzZh7iUBYl1Q.png"
data-width="1200" data-height="800" />
</figure>

This model did a much better job than the ARIMA and Holt-Winters models.
It still under-estimated each value but the trend is clearly closer than
the other models.

This doesn't really bother me. ARIMA has limits but I still like
statsmodels as a library --- the better prediction from Tensorflow is a
function of a different model, not a better API.

#### Real world example: ERCOT energy demand data {#1193 .graf .graf--h4 .graf-after--p name="1193"}

Let's look at actual data. This is hourly load demand data for power in
ERCOT (basically Texas) from Jan 7--11 2025. This data is available from
their website. I'm not going to repeat the code but I'll share the
graphs.

<figure id="f943" class="graf graf--figure graf-after--p">
<img
src="https://cdn-images-1.medium.com/max/800/1*7G55Vagng8-YQKuOdQHoYw.png"
class="graf-image" data-image-id="1*7G55Vagng8-YQKuOdQHoYw.png"
data-width="1000" data-height="600" />
</figure>

The data has clear patterns.

<figure id="dc87" class="graf graf--figure graf-after--p">
<img
src="https://cdn-images-1.medium.com/max/800/1*o9-uyQ7OB6UIoLY8SdOFIA.png"
class="graf-image" data-image-id="1*o9-uyQ7OB6UIoLY8SdOFIA.png"
data-width="1000" data-height="800" />
</figure>

Using ARIMA, we can forecast demand. The cone of uncertainty is large.
This could be reduced by adding in more data.

<figure id="7f72" class="graf graf--figure graf-after--p">
<img
src="https://cdn-images-1.medium.com/max/800/1*vqC6eaTUHrn3RywjFsZMxA.png"
class="graf-image" data-image-id="1*vqC6eaTUHrn3RywjFsZMxA.png"
data-width="1000" data-height="600" />
</figure>

<figure id="a78d" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*X4Gz24qulWzqirIdXJoivA.png"
class="graf-image" data-image-id="1*X4Gz24qulWzqirIdXJoivA.png"
data-width="1336" data-height="804" />
</figure>

ARIMA actually did a good job here.

<figure id="b5ee" class="graf graf--figure graf-after--p">
<img
src="https://cdn-images-1.medium.com/max/800/1*eget86zG4qqPmnGnWenlRw.png"
class="graf-image" data-image-id="1*eget86zG4qqPmnGnWenlRw.png"
data-width="1000" data-height="600" />
</figure>

Let's look at Holt-Winters.

<figure id="f4b7" class="graf graf--figure graf-after--p">
<img
src="https://cdn-images-1.medium.com/max/800/1*IynLJLmw5RyeQUTSKzsqRA.png"
class="graf-image" data-image-id="1*IynLJLmw5RyeQUTSKzsqRA.png"
data-width="1000" data-height="600" />
</figure>

Not bad. Let's look at Keras.

<figure id="d4da" class="graf graf--figure graf-after--p">
<img
src="https://cdn-images-1.medium.com/max/800/1*eAUS_Ctw_-i_NK8cG-wmBA.png"
class="graf-image" data-image-id="1*eAUS_Ctw_-i_NK8cG-wmBA.png"
data-width="1200" data-height="800" />
</figure>

The LSTM still does a better job predicting the data.

### Key Takeaways {#f041 .graf .graf--h3 .graf-after--p name="f041"}

`statsmodels`{.markup--code .markup--p-code} is the benchmark I use to
measure all other time series libraries. It isn't perfect but it is good
and if I could only use one library for the rest of time, I would choose
`statsmodels`{.markup--code .markup--p-code}.

Code for this project is available on
[GitHub](https://github.com/kylejones200/time_series/blob/main/Statsmodels.ipynb){.markup--anchor
.markup--p-anchor
data-href="https://github.com/kylejones200/time_series/blob/main/Statsmodels.ipynb"
rel="noopener" target="_blank"}.
:::
::::
::::::
:::::::::::::::

By [Kyle Jones](https://medium.com/@kylejones_47003){.p-author .h-card}
on [January 12, 2025](https://medium.com/p/ea0fce203c0a).

[Canonical
link](https://medium.com/@kylejones_47003/time-series-analysis-with-statsmodels-in-python-ea0fce203c0a){.p-canonical}

Exported from [Medium](https://medium.com) on February 9, 2025.
