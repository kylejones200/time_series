Modern Forecasting Libraries
=============================

Modern forecasting libraries provide high-level APIs and automated model selection.

Prophet
-------

Facebook's Prophet for business time series.

**Template:** ``Prophet_Python/``

**Use when:**
- Data has strong seasonality
- Holidays/events matter
- You need robust forecasts

**Example:**

.. code-block:: python

   from prophet import Prophet
   
   model = Prophet()
   model.fit(df)
   forecast = model.predict(future_df)

Darts
-----

Unified API for multiple forecasting methods.

**Template:** ``Darts_Python/``

**Use when:**
- You want to try multiple methods easily
- Unified interface is preferred
- Multiple time series to handle

StatsForecast
-------------

Fast statistical forecasting from Nixtla.

**Template:** ``StatsForecast_Python/``

**Use when:**
- Speed is important
- Multiple series to forecast
- Statistical methods are preferred

Greykite
--------

LinkedIn's Greykite for flexible forecasting.

**Template:** ``Greykite_Python/``

**Use when:**
- You need flexible model components
- Regressors are available
- Interpretability matters

Foundation Models
-----------------

Large pre-trained models for time series.

**Chronos** (Amazon)
   Pre-trained transformer models.

**TimesFM** (Google)
   Foundation model for time series.

**LagLlama** (IBM)
   Granite TTM for time series.

**Sundial** (Moirai)
   Foundation model for forecasting.

**Use when:**
- You have limited training data
- Pre-trained knowledge helps
- Strong performance is critical

See individual template READMEs for detailed usage.

