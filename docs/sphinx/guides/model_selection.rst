Model Selection
===============

Choosing the right forecasting method depends on your data and requirements.

Data Characteristics
--------------------

**Short Series (< 50 points):**
- Moving Average
- Exponential Smoothing
- Simple ARIMA

**Seasonal Data:**
- Exponential Smoothing (Holt-Winters)
- Prophet
- Seasonal ARIMA

**Trending Data:**
- ARIMA
- Prophet
- DCA models (for production data)

**Long Series (> 100 points):**
- Deep learning methods (LSTM, N-BEATS)
- Foundation models (Chronos, TimesFM)
- Complex statistical methods

**Multiple Series:**
- VAR (Vector Autoregression)
- Multivariate methods

**Irregular/Noisy Data:**
- Robust methods (Kalman filters)
- Anomaly detection (STUMPY_PyOD)

Computational Resources
-----------------------

**Limited Resources:**
- Classical methods (ARIMA, Moving Average)
- Simple statistical methods

**Moderate Resources:**
- Prophet
- Darts
- StatsForecast

**High Resources:**
- Deep learning (LSTM, N-BEATS)
- Foundation models (Chronos, TimesFM)
- Bayesian methods (PyMC, Orbit)

Interpretability Requirements
------------------------------

**High Interpretability:**
- ARIMA
- Exponential Smoothing
- DCA models
- Linear methods

**Moderate Interpretability:**
- Prophet
- Statistical methods

**Low Interpretability:**
- Deep learning
- Foundation models
- Ensemble methods

Recommendation Workflow
-----------------------

1. Start with simple methods (Moving Average, ARIMA)
2. If seasonality is present, try Exponential Smoothing or Prophet
3. For production data, compare against DCA models
4. If you have long series and resources, try deep learning
5. Use the pipeline to compare multiple methods
6. Select based on evaluation metrics and business requirements

See :doc:`../examples/templates_overview` for a complete list of available methods.

