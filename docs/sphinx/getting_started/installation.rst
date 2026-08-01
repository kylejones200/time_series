Installation
============

Requirements
------------

Python 3.8 or higher is required.

Core Dependencies
------------------

Install the core dependencies:

.. code-block:: bash

   pip install pandas numpy scipy scikit-learn matplotlib signalplot pyyaml

Optional Dependencies
---------------------

Depending on which forecasting methods you want to use, you may need additional packages:

**Classical Methods:**
.. code-block:: bash

   pip install pmdarima statsmodels

**Prophet:**
.. code-block:: bash

   pip install prophet

**Deep Learning:**
.. code-block:: bash

   pip install torch tensorflow keras

**Modern Forecasting Libraries:**
.. code-block:: bash

   pip install darts statsforecast greykite merlion orbit

**Bayesian Methods:**
.. code-block:: bash

   pip install pymc arviz

**Specialized:**
.. code-block:: bash

   pip install tsfresh tslearn aeon stumpy pyod

Installation from Repository
----------------------------

Clone the repository:

.. code-block:: bash

   git clone <repository-url>
   cd time_series

The library is designed to be used directly from the repository. Add the repository root to your Python path or install in development mode:

.. code-block:: bash

   pip install -e .

Verification
------------

Verify the installation by running the reference forecast:

.. code-block:: bash

   python reference_forecast.py

This should generate a forecast plot and CSV output in the ``outputs/reference/`` directory.

