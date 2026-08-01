Repository Structure
=====================

The repository is organized as follows:

Core Components
---------------

::

   time_series/
   ├── src/                    # Core utilities
   │   ├── loader.py          # Data loading
   │   ├── model.py           # Model wrappers
   │   ├── evaluator.py       # Evaluation utilities
   │   ├── plotting.py        # Plotting functions
   │   ├── config.py          # Configuration loading
   │   └── base_template.py   # Base template class
   │
   ├── models/                 # Model implementations
   │   └── dca/               # Decline Curve Analysis
   │       ├── arps.py
   │       ├── exponential.py
   │       └── hyperbolic.py
   │
   ├── pipelines/              # Unified pipeline
   │   ├── forecasting_pipeline.py
   │   └── model_registry.py
   │
   ├── evaluation/             # Evaluation tools
   │   ├── metrics.py
   │   └── comparison.py
   │
   └── *_Python/               # Forecasting templates
       ├── config.yaml
       ├── main.py
       └── README.md

Data and Examples
-----------------

::

   ├── data/                   # Example data
   │   ├── reference/         # Reference dataset
   │   └── production/        # Production examples
   │
   ├── examples/               # Complete examples
   │   └── ts_vs_dca_comparison.py
   │
   └── reference_forecast.py   # Reference implementation

Documentation
-------------

::

   ├── docs/
   │   ├── sphinx/            # Sphinx documentation
   │   └── planning/          # Planning documents
   │
   └── README.md              # Main README

Each template directory contains:

- ``config.yaml``: Configuration file
- ``main.py``: Main execution script
- ``README.md``: Template-specific documentation
- ``requirements.txt``: Template dependencies
- ``outputs/``: Generated results

