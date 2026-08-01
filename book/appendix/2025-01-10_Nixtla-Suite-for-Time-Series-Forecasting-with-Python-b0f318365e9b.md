<div>

# Nixtla Suite for Time Series Forecasting with Python {#nixtla-suite-for-time-series-forecasting-with-python .p-name}

</div>

::: {.section .p-summary field="subtitle"}
Nixtla brings together multiple specialized libraries for time series
analysis, from traditional statistical models to advanced neural...
:::

::::::: {.section .e-content field="body"}
:::::: {#3bc1 .section .section .section--body .section--first .section--last}
::: section-divider

------------------------------------------------------------------------
:::

:::: section-content
::: {.section-inner .sectionLayout--insetColumn}
### Nixtla Suite for Time Series Forecasting with Python {#59e7 .graf .graf--h3 .graf--leading .graf--title name="59e7"}

#### Nixtla brings together multiple specialized libraries for time series analysis, from traditional statistical models to advanced neural networks. Its modular approach allows users to choose the right tool for their forecasting needs, whether it's high-performance statistical models, machine learning algorithms, or deep learning architectures. {#00ee .graf .graf--h4 .graf-after--h3 .graf--subtitle name="00ee"}

The **Nixtla suite** is a collection of Python libraries for time series
analysis. It feels like something that will be easier to use with more
practice. Having used a lot of libraries this is the one that haunts my
thoughts --- "I wonder if I could do this with NeuralForecast? ..." But
in the end, I struggled to get NeuralForecast to do what I wanted with
N-BEATS --- a task that was easy with DARTS.

Nixtla Suite is made of several libraries, each targeting specific
forecasting need:

-   [`StatsForecast`{.markup--code .markup--li-code}: Efficient
    implementations of statistical models.]{#370e}
-   [`NeuralForecast`{.markup--code .markup--li-code}: Deep learning
    models for time series forecasting.]{#da8c}
-   [`HierarchicalForecast`{.markup--code .markup--li-code}: Tools for
    hierarchical and grouped time series.]{#84f0}
-   [`MLForecast`{.markup--code .markup--li-code}: Machine learning
    models tailored for time series data.]{#3500}

Installation: You can install the Nixtla suite libraries individually
using `pip`{.markup--code .markup--p-code}:

``` {#0619 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="2" spellcheck="false" code-block-lang="bash"}
pip install statsforecast
pip install neuralforecast
pip install mlforecast
pip install hierarchicalforecast
```

### StatsForecast: High-Performance Statistical Models {#c66d .graf .graf--h3 .graf-after--pre name="c66d"}

**StatsForecast** provides fast implementations of classic statistical
models like ARIMA, ETS, and more. Let's do a simple for forecast with
autoARIMA.

<figure id="d5e0" class="graf graf--figure graf--iframe graf-after--p">

</figure>

<figure id="5bf3" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*7uIxpLBVp2o0pGlJz9Mbog.png"
class="graf-image" data-image-id="1*7uIxpLBVp2o0pGlJz9Mbog.png"
data-width="864" data-height="432" data-is-featured="true" />
</figure>

### MLForecast: Machine Learning for Time Series {#1b7e .graf .graf--h3 .graf-after--figure name="1b7e"}

**MLForecast** simplifies the application of machine learning models to
time series data by automating feature creation and model training.
Let's try forecasting with LightGBM (version 1). This version uses data
from FRED.

<figure id="2af5" class="graf graf--figure graf--iframe graf-after--p">

</figure>

<figure id="5f3b" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*CeEpnP3DQxN7VoYccv94JQ.png"
class="graf-image" data-image-id="1*CeEpnP3DQxN7VoYccv94JQ.png"
data-width="736" data-height="387" />
</figure>

<figure id="6828"
class="graf graf--figure graf--iframe graf-after--figure">

</figure>

<figure id="4831" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*j_a7jEDIiw_hUteFD92zAg.png"
class="graf-image" data-image-id="1*j_a7jEDIiw_hUteFD92zAg.png"
data-width="724" data-height="387" />
</figure>

Give the same simulated data, LGBMRegressor did much better than a basic
regression using sklearn.

<figure id="e7ed" class="graf graf--figure graf--iframe graf-after--p">

</figure>

<figure id="0fa9" class="graf graf--figure graf-after--figure">
<img
src="https://cdn-images-1.medium.com/max/800/1*Z8GC8UIgXyMWY5E36HppTQ.png"
class="graf-image" data-image-id="1*Z8GC8UIgXyMWY5E36HppTQ.png"
data-width="724" data-height="387" />
</figure>

I was excited to try the hierarchical and grouped time series but I
couldn't get the HierarchicalForecast to work.

**Real world example: ERCOT Energy Load Data**

Back to StatsForcast. I wanted to try it with a different dataset. So in
the examples below, I use energy load data from ERCOT, the grid
balancing authority in Texas.

``` {#adbd .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
import os
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from utilsforecast.losses import mse, mape

import matplotlib.pyplot as plt

# Set environment variable
os.environ['NIXTLA_ID_AS_COL'] = '1'

# Load and preprocess the data
df = pd.read_csv("ercot_load_data.csv")
df['ds'] = pd.to_datetime(df['date'])  # Ensure 'date' is in datetime format
df['y'] = pd.to_numeric(df['values'], errors='coerce')  # Convert 'values' to numeric
df = df.sort_values('ds')  # Sort by date
df = df.dropna(subset=['ds', 'y'])  # Drop rows with missing values

# Resample the data to hourly frequency
df = df.set_index('ds').resample('h')['y'].mean().reset_index()
df["unique_id"] = "series1"  # Assign a unique ID for StatsForecast compatibility

# Split the data into training and hold-out sets
hold_out_hours = 24
train = df.iloc[:-hold_out_hours]
hold_out = df.iloc[-hold_out_hours:]

# Initialize and fit the StatsForecast model
models = [AutoARIMA(season_length=24)]  # Adjust seasonality to daily
sf = StatsForecast(models=models, freq='h', n_jobs=-1)
sf.fit(train)

# Generate forecasts for the hold-out period
horizon = len(hold_out)
forecasts = sf.predict(h=horizon)

# Add timestamps to the forecast results
forecasts['ds'] = hold_out['ds'].values

# Visualize the results
plt.figure(figsize=(12, 6))

# Plot historical data
plt.plot(df['ds'], df['y'], label='Historical Data', color='blue')

# Highlight hold-out data in green
plt.plot(hold_out['ds'], hold_out['y'], label='Hold-Out Data', color='green')

# Plot forecasted data in red
plt.plot(forecasts['ds'], forecasts['AutoARIMA'], label='Forecast', color='red')

# Add labels, title, and legend
plt.title('Time Series Forecast with AutoARIMA')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("AutoARIMA_Forecast.png")
plt.show()

# Calculate and print forecast metrics
actual_values = hold_out['y'].values
forecast_values = forecasts['AutoARIMA'].values

mse_value = np.mean((actual_values - forecast_values) ** 2)
rmse_value = np.sqrt(mse_value)
mae_value = np.mean(np.abs(actual_values - forecast_values))

print("\nForecast Metrics:")
print(f"Mean Squared Error (MSE): {mse_value:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_value:.2f}")
print(f"Mean Absolute Error (MAE): {mae_value:.2f}")
```

<figure id="e241" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*XtQX7Yi7rZbBCsi9Jq_RHQ.png"
class="graf-image" data-image-id="1*XtQX7Yi7rZbBCsi9Jq_RHQ.png"
data-width="1200" data-height="600" />
</figure>

``` {#1a26 .graf .graf--pre .graf-after--figure .graf--preV2 code-block-mode="2" spellcheck="false" code-block-lang="plaintext"}
Forecast Metrics:
Mean Squared Error (MSE): 142.56
Root Mean Squared Error (RMSE): 11.94
Mean Absolute Error (MAE): 11.00
```

Let's dive a little deeper.

``` {#6d12 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
from statsforecast.models import (
    HoltWinters,
    CrostonClassic as Croston, 
    HistoricAverage,
    DynamicOptimizedTheta as DOT,
    SeasonalNaive
)

# Create a list of models and instantiation parameters
models = [
    HoltWinters(),
    Croston(),
    SeasonalNaive(season_length=24),
    HistoricAverage(),
    DOT(season_length=24)
]
# Instantiate StatsForecast class as sf
sf = StatsForecast( 
    models=models,
    freq='h', 
    fallback_model = SeasonalNaive(season_length=7),
    n_jobs=-1,
)

forecasts_df = sf.forecast(df=train, h=48, level=[90])
forecasts_df.head()
```

<figure id="b344" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*ySGQ6_hxWUyiWp6_U6RCsw.png"
class="graf-image" data-image-id="1*ySGQ6_hxWUyiWp6_U6RCsw.png"
data-width="1128" data-height="670" />
</figure>

``` {#972f .graf .graf--pre .graf-after--figure .graf--preV2 code-block-mode="2" spellcheck="false" code-block-lang="python"}
sf.plot(df,forecasts_df)
```

<figure id="a46d" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*92ZUXFoxK7qto9-hJExsQQ.png"
class="graf-image" data-image-id="1*92ZUXFoxK7qto9-hJExsQQ.png"
data-width="1861" data-height="361" />
<figcaption>Visualization from StatForecast of ERCOT data</figcaption>
</figure>

``` {#da2b .graf .graf--pre .graf-after--figure .graf--preV2 code-block-mode="2" spellcheck="false" code-block-lang="python"}
cv_df = sf.cross_validation(
    df=df,
    h=24,
    step_size=24,
    n_windows=2
)

def evaluate_cv(df, metric):
    models = df.columns.drop(['unique_id', 'ds', 'y', 'cutoff']).tolist()
    evals = metric(df, models=models)
    evals['best_model'] = evals[models].idxmin(axis=1)
    return evals
evaluation_df = evaluate_cv(cv_df, mape)
evaluation_df.head()
```

<figure id="de17" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*M3HPX6cEm8YzXqO1_NwIIg.png"
class="graf-image" data-image-id="1*M3HPX6cEm8YzXqO1_NwIIg.png"
data-width="1732" data-height="124" />
</figure>

This is pretty cool. We can see how well each of these models does for
our dataset based on MAPE. Dynamic Optimized Theta wins!

``` {#10fc .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="2" spellcheck="false" code-block-lang="python"}
sf.plot(df, forecasts_df, models=["DynamicOptimizedTheta"],  level=[90])
```

<figure id="852c" class="graf graf--figure graf-after--pre">
<img
src="https://cdn-images-1.medium.com/max/800/1*Fan8bcFZQvlFnkESgmQIKg.png"
class="graf-image" data-image-id="1*Fan8bcFZQvlFnkESgmQIKg.png"
data-width="1926" data-height="361" />
</figure>

What about the LightGBM?

<figure id="57b5" class="graf graf--figure graf-after--p">
<img
src="https://cdn-images-1.medium.com/max/800/1*QTWc-qtU7zA3hvfs_J_44w.png"
class="graf-image" data-image-id="1*QTWc-qtU7zA3hvfs_J_44w.png"
data-width="1200" data-height="600" />
</figure>

``` {#fecb .graf .graf--pre .graf-after--figure .graf--preV2 code-block-mode="2" spellcheck="false" code-block-lang="plaintext"}
Forecast Metrics:
Mean Squared Error (MSE): 598.98
Root Mean Squared Error (RMSE): 24.47
Mean Absolute Error (MAE): 19.99
```

By comparison, ARIMA was MAE of 11.

#### So what? {#408a .graf .graf--h4 .graf-after--p name="408a"}

I feel like I've only scratched the surface of what the Nixtla suite can
do. I think I could make the analysis much faster for large datasets by
parallelizing using Nixtla. It is a little finicky and not as
straightforward as something like statsmodels but it has more
capabilities.
:::
::::
::::::
:::::::

By [Kyle Jones](https://medium.com/@kylejones_47003){.p-author .h-card}
on [January 10, 2025](https://medium.com/p/b0f318365e9b).

[Canonical
link](https://medium.com/@kylejones_47003/nixtla-suite-for-time-series-forecasting-with-python-b0f318365e9b){.p-canonical}

Exported from [Medium](https://medium.com) on February 9, 2025.
