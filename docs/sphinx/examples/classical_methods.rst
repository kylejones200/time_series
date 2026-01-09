Classical Methods
=================

Classical time series forecasting methods are well-established statistical approaches.

ARIMA
-----

Autoregressive Integrated Moving Average models.

**Template:** ``ARIMA_Python/``

**Use when:**
- Data shows trends and seasonality
- You need interpretable models
- Computational resources are limited

**Example:**

.. code-block:: python

   from src import ARIMAModel
   
   model = ARIMAModel()
   model.fit(train_series)
   forecast = model.forecast(n_periods=12)

Moving Average
--------------

Simple and exponential moving averages.

**Template:** ``MovingAverage_Python/``

**Use when:**
- Data is relatively stable
- You need a simple baseline
- Quick forecasts are needed

Exponential Smoothing
---------------------

Holt-Winters exponential smoothing.

**Template:** ``ExponentialSmoothing_Python/``

**Use when:**
- Data has trends and seasonality
- You want a simple but effective method
- Interpretability is important

VAR
---

Vector Autoregression for multivariate time series.

**Template:** ``VAR_Python/``

**Use when:**
- You have multiple related time series
- Relationships between series are important
- You need to forecast multiple variables

See individual template READMEs for detailed usage and configuration options.

