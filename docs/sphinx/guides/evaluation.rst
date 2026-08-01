Evaluation
===========

All models are evaluated using consistent metrics for fair comparison.

Available Metrics
-----------------

**MAE (Mean Absolute Error)**
   Average absolute difference between forecast and actual values.
   Lower is better.

**RMSE (Root Mean Squared Error)**
   Square root of average squared differences.
   Penalizes large errors more than MAE.
   Lower is better.

**MAPE (Mean Absolute Percentage Error)**
   Average percentage error.
   Useful for comparing across different scales.
   Lower is better.

**R² (Coefficient of Determination)**
   Proportion of variance explained by the model.
   Higher is better (max 1.0).

Using the Evaluator
-------------------

.. code-block:: python

   from src import Evaluator

   evaluator = Evaluator(test_size=0.2)
   train, test = evaluator.split(series)

   # After forecasting
   metrics = evaluator.evaluate(forecast, test)
   print(f"RMSE: {metrics['RMSE']:.4f}")
   print(f"MAE: {metrics['MAE']:.4f}")

Comparing Models
----------------

Use the comparison utilities:

.. code-block:: python

   from evaluation import compare_forecasts

   comparison = compare_forecasts(
       forecasts={
           "ARIMA": arima_forecast,
           "Prophet": prophet_forecast,
           "DCA": dca_forecast
       },
       actual=test_series
   )

   print(comparison)

The comparison returns a DataFrame with metrics for each model, sorted by performance.

Interpreting Results
--------------------

**RMSE vs MAE:**
- RMSE penalizes large errors more
- Use RMSE if large errors are particularly costly
- Use MAE if all errors are equally important

**MAPE:**
- Useful when series have different scales
- Be careful with values near zero (can inflate MAPE)

**R²:**
- 1.0 = perfect predictions
- 0.0 = model performs as well as predicting the mean
- Negative = model performs worse than the mean

Best Practices
--------------

1. Use a hold-out test set (don't evaluate on training data)
2. Use time-aware splits (don't shuffle time series data)
3. Compare multiple metrics (don't rely on a single metric)
4. Consider business context (some errors may be more costly)
5. Visualize forecasts alongside metrics

