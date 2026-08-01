Extending the Library
=====================

The library is designed to be easily extended with new models and functionality.

Adding a New Model
------------------

Create a model class that follows the interface:

.. code-block:: python

   class MyForecastModel:
       def fit(self, series: pd.Series) -> "MyForecastModel":
           """Fit the model to training data."""
           # Your fitting logic
           self.fitted_ = True
           return self
       
       def forecast(self, n_periods: int) -> pd.Series:
           """Generate forecast."""
           if not hasattr(self, 'fitted_'):
               raise ValueError("Model must be fitted first")
           # Your forecasting logic
           return forecast_series

Use in pipeline:

.. code-block:: python

   from pipelines import ForecastingPipeline

   pipeline = ForecastingPipeline(...)
   pipeline.add_model("My Model", MyForecastModel())

Creating a New Template
-----------------------

1. Create a new directory: ``MyMethod_Python/``
2. Add ``config.yaml`` with standard structure
3. Create ``main.py`` following the base template pattern:

.. code-block:: python

   from src import (
       load_config,
       load_time_series,
       Evaluator,
       save_plot,
   )

   def main():
       config = load_config()
       series = load_time_series(...)
       evaluator = Evaluator(test_size=0.2)
       train, test = evaluator.split(series)
       
       # Your model fitting and forecasting
       
       # Evaluate and save results
       metrics = evaluator.evaluate(forecast, test)
       save_plot(fig, "outputs/forecast.png")

4. Add ``README.md`` documenting usage
5. Add ``requirements.txt`` with dependencies

Extending the Pipeline
----------------------

Add custom evaluation metrics:

.. code-block:: python

   from evaluation.metrics import calculate_metric

   def custom_metric(forecast, actual):
       # Your metric calculation
       return metric_value

   # Use in comparison
   from evaluation.comparison import compare_forecasts
   
   comparison = compare_forecasts(
       forecasts=forecasts,
       actual=actual,
       metrics={'custom': custom_metric}
   )

Best Practices
--------------

1. Follow the existing patterns and structure
2. Use consolidated utilities from ``src/``
3. Add comprehensive docstrings
4. Include examples in README
5. Test with the reference dataset
6. Update documentation

See :doc:`../contributing` for contribution guidelines.

