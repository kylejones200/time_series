<div>

# pytimetk for Time Series Analysis in Python {#pytimetk-for-time-series-analysis-in-python .p-name}

</div>

::: {.section .p-summary field="subtitle"}
pytimetk combines intuitive feature engineering with interactive
visualizations for enhanced time series analysis.
:::

::::::: {.section .e-content field="body"}
:::::: {#20e7 .section .section .section--body .section--first .section--last}
::: section-divider

------------------------------------------------------------------------
:::

:::: section-content
::: {.section-inner .sectionLayout--insetColumn}
### `pytimetk`{.markup--code .markup--h3-code} for Time Series Analysis in Python {#30b4 .graf .graf--h3 .graf--leading .graf--title name="30b4"}

#### pytimetk combines intuitive feature engineering with interactive visualizations for enhanced time series analysis. {#bf41 .graf .graf--h4 .graf-after--h3 .graf--subtitle name="bf41"}

`pytimetk`{.markup--code .markup--p-code} is a time series analysis and
forecasting in Python inspired by the R package `timetk`{.markup--code
.markup--p-code}.

It has some nifty tools for feature engineering and visualization (be
careful, the creators spell it the British way "visuali**s**ation"). The
visualization is my favorite part. It uses plotly under the hood so the
graphs are interactive --- unfortunately the graphs are static in
medium.

<figure id="ad3f" class="graf graf--figure graf-after--p">
<img
src="https://cdn-images-1.medium.com/max/800/1*pAiqqLZjfTsFv5S_sfHf3Q.gif"
class="graf-image" data-image-id="1*pAiqqLZjfTsFv5S_sfHf3Q.gif"
data-width="1500" data-height="1000" />
</figure>

Feature Engineering is really easy in `pytimetk`{.markup--code
.markup--p-code} --- including more advanced concepts like Fast Fourier
Transforms. The time-based filtering and aggregation is ok --- but I
will probably use Pandas for this in the future.

#### Let's look at a time series dataset using `pytimetk`{.markup--code .markup--h4-code}. {#77fc .graf .graf--h4 .graf-after--p name="77fc"}

I'm using random data here.

``` {#59ee .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
import pandas as pd
import numpy as np
import pytimetk as tk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Create a simulated time series dataset
np.random.seed(42)
n = 500
time = pd.date_range(start="2020-01-01", periods=n, freq="D")
values = 100 + np.cumsum(np.random.normal(0, 1, n))
df = pd.DataFrame({"date": time, "value": values})

# Print the first few rows
print(df.head())

# Plot the time series using pytimetk's plot_timeseries method
fig = (
    tk.plot_timeseries(
        df,
        date_column='date',
        value_column='value',
        facet_ncol=1,
        x_axis_date_labels="%Y",
        engine='plotly',
        title="Simulated Time Series"
    )
)

fig.write_image("time_series_plot.png")
fig.show()
```

<figure id="6cad" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*VPZLPKwHIZASv4YdH7KOTQ.png"
class="graf-image" data-image-id="1*VPZLPKwHIZASv4YdH7KOTQ.png"
data-width="700" data-height="500" data-is-featured="true" />
<figcaption>A pretty plot</figcaption>
</figure>

#### Feature Engineering {#19b5 .graf .graf--h4 .graf-after--figure name="19b5"}

Create lagged features, rolling averages, and Fourier terms for machine
learning.

``` {#f245 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
"""
Feature Engineering
Create lagged features, rolling averages, and Fourier terms for machine learning.
"""


# Feature Engineering
# Add rolling mean and standard deviation for a window of 7 days
rolled_df_7 = tk.augment_rolling_apply(
    df,
    date_column="date",
    window=7,
    window_func=[
        ("rolling_mean_7", lambda x: x["value"].mean()),
        ("rolling_std_7", lambda x: x["value"].std()),
    ],
    center=False,
    threads=1
)

# Add rolling mean and standard deviation for a window of 14 days
rolled_df_14 = tk.augment_rolling_apply(
    rolled_df_7,
    date_column="date",
    window=14,
    window_func=[
        ("rolling_mean_14", lambda x: x["value"].mean()),
        ("rolling_std_14", lambda x: x["value"].std()),
    ],
    center=False,
    threads=1
)

# Add Fourier series for seasonality
rolled_df = tk.augment_fourier(
    rolled_df_14,
    date_column="date",  # Specify the date column
)
rolled_df.tail()
```

#### Resample and Aggregate {#ed93 .graf .graf--h4 .graf-after--pre name="ed93"}

We can use `pytimetk`{.markup--code .markup--p-code} to aggregate the
time series data by week or filter data for a specific time period.

``` {#b8ab .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
"""
Time-Based Filtering
Filter data for a specific time period.
"""

# Filter data for the year 2021
df_filtered = tk.filter_by_time(
    df,
    date_column="date",
    start_date="2021-01-01",
    end_date="2021-12-31"
)

# Plot the filtered data using pytimetk's plot_timeseries method
fig = (
    tk.plot_timeseries(
        df_filtered,
        date_column='date',
        value_column='value',
        facet_ncol=1,
        x_axis_date_labels="%b %Y",  # Adjusted for better visualization within a year
        engine='plotly',
        title="Filtered Time Series (2021)"
    )
)

fig.write_image("filtered_time_series_2021.png")
fig.show()
```

<figure id="ad61" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*NHo73ytUmyPkldO5iQisdA.png"
class="graf-image" data-image-id="1*NHo73ytUmyPkldO5iQisdA.png"
data-width="700" data-height="500" />
<figcaption>Zooming in on part of the data</figcaption>
</figure>

#### Forecast Evaluation {#74f5 .graf .graf--h4 .graf-after--figure name="74f5"}

Now let's get to some actual work. We will create sample data and
evaluate forecasting performance.

``` {#574d .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}

# Feature Engineering
# Add lagged features for 1, 2, and 3 days
df_features = tk.augment_lags(
    df,
    date_column="date",
    value_column="value",
    lags=[1, 2, 3]
)

# Add rolling mean and standard deviation for 7 and 14 days
df_features = tk.augment_rolling_apply(
    df_features,
    date_column="date",
    window=7,
    window_func=[
        ("rolling_mean_7", lambda x: x["value"].mean()),
        ("rolling_std_7", lambda x: x["value"].std())
    ],
    center=False,
    threads=1
)

df_features = tk.augment_rolling_apply(
    df_features,
    date_column="date",
    window=14,
    window_func=[
        ("rolling_mean_14", lambda x: x["value"].mean()),
        ("rolling_std_14", lambda x: x["value"].std())
    ],
    center=False,
    threads=1
)

# Add Fourier series for seasonality
df_features = tk.augment_fourier(
    df_features,
    date_column="date",  # Specify the date column
)

# Drop rows with NaN values (resulting from lagged features or rolling stats)
df_features = df_features.dropna()

# Verify the resulting DataFrame
print(df_features.head())
```

With the set up complete, we can move to the ML.

``` {#0420 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="2" spellcheck="false" code-block-lang="python"}
# Forecast Evaluation
# Split into training and testing sets
train = df_features.iloc[:-100]
test = df_features.iloc[-100:]
# Prepare features and target
X_train = train.drop(columns=["date", "value"])
y_train = train["value"]
X_test = test.drop(columns=["date", "value"])
y_test = test["value"]
# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
# Step 4: Combine Actual and Predicted Values for Plotting
test['Predicted'] = y_pred
# Melt data for pytimetk plotting (long format)
plot_df = test.melt(id_vars="date", value_vars=["value", "Predicted"], 
                    var_name="Series", value_name="Value")
# Plot using pytimetk
fig = tk.plot_timeseries(
    plot_df,
    date_column="date",
    value_column="Value",
    color_column="Series",
    title="Forecast vs Actual",
    x_axis_date_labels="%b %d, %Y",
    engine="plotly"
)
fig.write_image("forecast_vs_actual_plot.png")
fig.show()
```

<figure id="6865" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*lJ6o0eqjhL-ozV2HLywvwQ.png"
class="graf-image" data-image-id="1*lJ6o0eqjhL-ozV2HLywvwQ.png"
data-width="700" data-height="500" />
</figure>

By comparison, here is the same plot with matplotlib. It is clear which
visualization is better. There is a weird thing where
`pytimetk`{.markup--code .markup--p-code} connects the first and last
points in a straight line.

<figure id="a45c" class="graf graf--figure graf-after--p">
<img
src="https://cdn-images-1.medium.com/max/800/1*lei7UrCMO-vDzrJaaDAMlA.png"
class="graf-image" data-image-id="1*lei7UrCMO-vDzrJaaDAMlA.png"
data-width="1000" data-height="600" />
</figure>

`pytimetk`{.markup--code .markup--p-code} is based on Pandas so we can
use any machine learning models (e.g., XGBoost or LSTM) from sklearn
that we want. I like the Fourier transforms for seasonal decomposition
to analyze trends and periodic patterns.

`pytimetk`{.markup--code .markup--p-code} doesn't do ARIMA or SARIMA. I
guess this is fine since we have other tools that do this well. Just
odd.

Let's look at some real data. This data comes from ERCOT.

<figure id="e978" class="graf graf--figure graf-after--p">
<img
src="https://cdn-images-1.medium.com/max/800/1*rYO8pO_N_7S_bPsFvS3UVQ.png"
class="graf-image" data-image-id="1*rYO8pO_N_7S_bPsFvS3UVQ.png"
data-width="700" data-height="500" />
</figure>

<figure id="ccda" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*UCLs-kcfq4Mhn6TlRfkHlQ.png"
class="graf-image" data-image-id="1*UCLs-kcfq4Mhn6TlRfkHlQ.png"
data-width="1212" data-height="670" />
</figure>

We can filter the data to just values in 2025.

<figure id="46d0" class="graf graf--figure graf-after--p">
<img
src="https://cdn-images-1.medium.com/max/800/1*jx-WPCFCe8V2AxXaWS-fDA.png"
class="graf-image" data-image-id="1*jx-WPCFCe8V2AxXaWS-fDA.png"
data-width="700" data-height="500" />
</figure>

You can see a big increase in demand on Jan 7 when a large storm came
through Texas.

``` {#c409 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
import pandas as pd
import numpy as np
import pytimetk as tk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


df.sort_values(by="date", inplace=True)

# Feature Engineering
# Add lagged features for 1, 2, and 3 days
df_features = tk.augment_lags(
    df,
    date_column="date",
    value_column="values",
    lags=[1, 2, 3]
)

# Add rolling mean and standard deviation for 7 and 14 days
df_features = tk.augment_rolling_apply(
    df_features,
    date_column="date",
    window=7,
    window_func=[
        ("rolling_mean_7", lambda x: x["values"].mean()),
        ("rolling_std_7", lambda x: x["values"].std())
    ],
    center=False,
    threads=1
)

df_features = tk.augment_rolling_apply(
    df_features,
    date_column="date",
    window=14,
    window_func=[
        ("rolling_mean_14", lambda x: x["values"].mean()),
        ("rolling_std_14", lambda x: x["values"].std())
    ],
    center=False,
    threads=1
)

# Add Fourier series for seasonality
df_features = tk.augment_fourier(
    df_features,
    date_column="date",  # Specify the date column
)

# Drop rows with NaN values (resulting from lagged features or rolling stats)
df_features = df_features.dropna()

# Verify the resulting DataFrame
print(df_features.head())

# Step 3: Forecast Evaluation
# Split into training and testing sets
train = df_features.iloc[:-30]
test = df_features.iloc[-30:]
# Prepare features and target
X_train = train.drop(columns=["date", "values"])
y_train = train["values"]
X_test = test.drop(columns=["date", "values"])
y_test = test["values"]
# Train the model
model = RandomForestRegressor(random_state=123)
model.fit(X_train, y_train)
# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
# Step 4: Combine Actual and Predicted Values for Plotting
test['Predicted'] = y_pred
# Melt data for pytimetk plotting (long format)
plot_df = test.melt(id_vars="date", value_vars=["values", "Predicted"], 
                    var_name="Series", value_name="Values")
# Plot using pytimetk
fig = tk.plot_timeseries(
    plot_df,
    date_column="date",
    value_column="Values",
    color_column="Series",
    title="Forecast vs Actual",
    x_axis_date_labels="%b %d, %Y",
    engine="plotly"
)
fig.write_image("forecast_vs_actual_plot.png")
fig.show()
```

<figure id="8992" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*_GL_yzjKuhEGOxevgqugYg.png"
class="graf-image" data-image-id="1*_GL_yzjKuhEGOxevgqugYg.png"
data-width="700" data-height="500" />
</figure>

And this plot is still better than the plot from matplotlib.

``` {#9d02 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="2" spellcheck="false" code-block-lang="python"}
# Plot actual vs predicted in a basic plot to inspect
plt.figure(figsize=(10, 6))
plt.plot(test["date"], y_test, label="Actual", color="Blue")
plt.plot(test["date"], y_pred, label="Predicted", color="Red")
plt.xlabel("Date")
plt.ylabel("Values")
plt.legend()
plt.tight_layout()
plt.savefig("forecast_plot.png")
plt.show()
```

<figure id="6b56" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*G4zPBrMv8ubDHqTZ0ehdwA.png"
class="graf-image" data-image-id="1*G4zPBrMv8ubDHqTZ0ehdwA.png"
data-width="1000" data-height="600" />
</figure>

### So what? {#4ccf .graf .graf--h3 .graf-after--figure name="4ccf"}

I liked `pytimetk`{.markup--code .markup--p-code} more than I thought I
would. Maybe it is because I love R and the syntax is similar. I really
like how it creates interactive plotly graphs.

Code for this project is available on
[GitHub](https://github.com/kylejones200/time_series/blob/main/pytimekt.ipynb){.markup--anchor
.markup--p-anchor
data-href="https://github.com/kylejones200/time_series/blob/main/pytimekt.ipynb"
rel="noopener" target="_blank"}.
:::
::::
::::::
:::::::

By [Kyle Jones](https://medium.com/@kylejones_47003){.p-author .h-card}
on [January 11, 2025](https://medium.com/p/92f725352d99).

[Canonical
link](https://medium.com/@kylejones_47003/pytimetk-for-time-series-analysis-in-python-92f725352d99){.p-canonical}

Exported from [Medium](https://medium.com) on February 9, 2025.
