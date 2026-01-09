Examples
========

Basic Forecasting
-----------------

The simplest forecasting example:

.. code-block:: python

   from src import load_time_series, ARIMAModel, Evaluator
   from src.plotting import create_forecast_plot, save_plot

   # Load data
   series = load_time_series(
       "data/reference/example_series.csv",
       date_column="date",
       value_column="value"
   )

   # Split train/test
   evaluator = Evaluator(test_size=0.2)
   train, test = evaluator.split(series)

   # Fit model
   model = ARIMAModel()
   model.fit(train)

   # Forecast
   forecast, conf_int = model.forecast(n_periods=len(test), return_conf_int=True)

   # Evaluate
   metrics = evaluator.evaluate(forecast, test)
   print(f"RMSE: {metrics['RMSE']:.4f}")

   # Plot
   fig, ax = create_forecast_plot(train, test, forecast, conf_int)
   save_plot(fig, "outputs/forecast.png")

Comparing Models
----------------

Compare multiple forecasting methods:

.. code-block:: python

   from pipelines import ForecastingPipeline
   from models.dca import ArpsExponential

   pipeline = ForecastingPipeline(
       data_path="data/production/well_production.csv",
       target_column="oil_rate",
       forecast_horizon=12
   )

   # Add multiple models
   pipeline.add_model("ARIMA", ARIMAModel())
   pipeline.add_model("Arps Exponential", ArpsExponential())

   # Run all models
   results = pipeline.run_all()

   # Compare
   comparison = pipeline.compare_models(results)
   print(comparison)

   # Get best model
   best_model = comparison.index[0]
   print(f"Best model: {best_model}")

Custom Model
------------

Create a custom forecasting model:

.. code-block:: python

   from src.model import BaseModel

   class MyCustomModel(BaseModel):
       def fit(self, series):
           # Your fitting logic
           self.fitted_ = True
           return self

       def forecast(self, n_periods):
           # Your forecasting logic
           return forecast_series

   # Use in pipeline
   pipeline.add_model("Custom", MyCustomModel())

More Examples
-------------

- See :doc:`../examples/templates_overview` for template-specific examples
- Check the ``examples/`` directory in the repository
- Review template README files in each ``*_Python/`` directory

