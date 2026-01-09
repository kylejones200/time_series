Quick Start
===========

This guide will get you up and running with time series forecasting in minutes.

Generate Example Data
---------------------

First, generate some example production data:

.. code-block:: bash

   python data/production/generate_example_data.py

This creates example well production data in ``data/production/``.

Run the Reference Forecast
---------------------------

The simplest way to get started is with the reference forecast script:

.. code-block:: bash

   python reference_forecast.py

This script demonstrates the standard workflow:

1. Loads time series data from CSV
2. Splits into train/test sets
3. Fits an ARIMA model
4. Generates a forecast
5. Evaluates performance
6. Saves results (plot + CSV)

The output will be in ``outputs/reference/``.

Using a Template
-----------------

Each forecasting method has its own template directory. For example, to use Prophet:

.. code-block:: bash

   cd Prophet_Python
   python main.py

The template will:
- Load data from ``config.yaml``
- Fit the model
- Generate forecasts
- Save results to ``outputs/``

Using the Pipeline
------------------

For comparing multiple models, use the unified pipeline:

.. code-block:: python

   from pipelines import ForecastingPipeline
   from models.dca import ArpsExponential, ArpsHyperbolic

   # Initialize pipeline
   pipeline = ForecastingPipeline(
       data_path="data/production/well_production.csv",
       target_column="oil_rate",
       forecast_horizon=12,
       train_size=0.8
   )

   # Add models
   pipeline.add_model("Arps Exponential", ArpsExponential())
   pipeline.add_model("Arps Hyperbolic", ArpsHyperbolic())

   # Run and compare
   results = pipeline.run_all()
   comparison = pipeline.compare_models(results)
   print(comparison)

Next Steps
----------

- See :doc:`../guides/overview` for a comprehensive overview
- Check :doc:`../examples/templates_overview` for available templates
- Read :doc:`../api/pipelines` for detailed API documentation

