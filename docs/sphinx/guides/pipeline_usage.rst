Using the Pipeline
===================

The unified forecasting pipeline allows you to compare multiple models side-by-side.

Basic Usage
-----------

.. code-block:: python

   from pipelines import ForecastingPipeline
   from models.dca import ArpsExponential

   # Initialize pipeline
   pipeline = ForecastingPipeline(
       data_path="data/production/well_production.csv",
       target_column="oil_rate",
       forecast_horizon=12,
       train_size=0.8
   )

   # Add models
   pipeline.add_model("ARIMA", ARIMAModel())
   pipeline.add_model("Arps Exponential", ArpsExponential())

   # Run all models
   results = pipeline.run_all()

   # Compare results
   comparison = pipeline.compare_models(results)
   print(comparison)

Model Registry
--------------

Register models for reuse:

.. code-block:: python

   from pipelines import register_model
   from models.dca import ArpsExponential

   register_model("Arps Exponential", lambda: ArpsExponential())

   # Use registered model
   pipeline.add_model_from_registry("Arps Exponential")

Saving Results
-------------

Save all results to disk:

.. code-block:: python

   pipeline.save_results(results, "outputs/comparison/")

This creates:
- Forecast plots for each model
- CSV files with forecasts
- Comparison table
- Metrics summary

Custom Models
-------------

Add custom models that follow the interface:

.. code-block:: python

   class MyModel:
       def fit(self, series):
           # Fit logic
           return self
       
       def forecast(self, n_periods):
           # Forecast logic
           return forecast_series

   pipeline.add_model("Custom", MyModel())

