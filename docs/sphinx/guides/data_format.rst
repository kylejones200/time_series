Data Format
===========

The library expects time series data in CSV format with specific column requirements.

Basic Format
------------

Minimum required columns:

.. code-block:: csv

   date,value
   2020-01-01,100.5
   2020-01-02,102.3
   2020-01-03,98.7

**Required Columns:**
- ``date``: Date in YYYY-MM-DD format (or any pandas-readable date format)
- ``value``: Numeric time series values

Production Data Format
----------------------

For production forecasting with DCA comparison:

.. code-block:: csv

   well_id,date,oil_rate,gas_rate,water_rate
   well_001,2020-01-01,100.5,50.2,10.1
   well_001,2020-01-02,98.3,49.8,9.9
   well_002,2020-01-01,200.1,100.5,20.2

**Required Columns:**
- ``well_id``: Unique well identifier
- ``date``: Date of measurement
- ``oil_rate``: Oil production rate (or your target variable)

**Optional Columns:**
- ``gas_rate``: Gas production rate
- ``water_rate``: Water production rate
- ``cum_oil``, ``cum_gas``, ``cum_water``: Cumulative production

Loading Data
------------

Use the consolidated loader:

.. code-block:: python

   from src import load_time_series

   series = load_time_series(
       "data/my_series.csv",
       date_column="date",
       value_column="value"
   )

The loader will:
- Parse dates automatically
- Set date as index
- Sort by date
- Handle missing values

Data Requirements
---------------

**Minimum Data Points:**
- Most methods require at least 20-30 data points
- Seasonal methods need at least 2 full seasons
- Deep learning methods typically need 100+ points

**Frequency:**
- Data can be daily, weekly, monthly, or irregular
- The library will attempt to infer frequency automatically
- Specify frequency in config if needed

**Missing Values:**
- Missing dates will be handled automatically
- Missing values in the series should be handled before loading
- Use forward fill, interpolation, or other methods as appropriate

Example Data Generation
------------------------

Generate example production data:

.. code-block:: bash

   python data/production/generate_example_data.py

This creates synthetic well production data following realistic decline curves.

